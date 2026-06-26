"""Oracle-feature injection mechanism (geometric-manifold features).

Mirrors ``modular_addition/oracle/inject.py``: a *frozen* (non-trainable) vector
that is linearly ADDED into the residual stream as a fixed function of the token
id. In the RoPE-era nanochat ``GPT`` there is no additive positional encoding to
piggyback on, so the injection is itself the positional-encoding-style hook —
added in ``GPT.forward`` after the post-embedding norm and smear, just before the
``x0`` residual is saved (see the guarded block there).

What it carries here are *geometric* features: a chosen set of token ids placed
at points sampled from a manifold (ring, line, sphere, helix) embedded in a few
reserved residual dimensions. This is the natural-language analogue of the
modular-addition Fourier oracle (which is literally a ring at frequency k).

Design notes
------------
* The oracle is held OUTSIDE ``nn.Parameter`` (a plain tensor on a small carrier
  object), so it never receives gradient, is untouched by weight decay, and
  stays out of ``GPT.setup_optimizer``'s param-group partition (which asserts it
  covers every ``Parameter``). It is a true fixed feature.
* Attach AFTER the model is materialized (``to_empty`` + ``init_weights``): the
  table is real data and must live on the model's device, not on meta.
* ``attach_oracle(model, fn)`` sets ``model.oracle_fn`` and ``model.inject=True``.
  ``model.inject`` gates injection on/off — used for (a) ablation (turn off,
  measure ΔCE) and (b) delayed injection (turn on at step T).

Coordinate convention: each manifold generator returns ``(n, k)`` coordinates
that are unit-norm per row (each point sits on the unit sphere of its
``k``-dim subspace), so ``amp`` is exactly the per-token oracle norm — the
direct analogue of the Fourier oracle's per-frequency amplitude.
"""
import math

import torch


# --------------------------------------------------------------------------- #
# Carrier: a frozen per-token-id additive table, O[idx] = table[idx]
# --------------------------------------------------------------------------- #
class _TableOracle:
    """Callable frozen oracle: ``fn(idx) -> table[idx]``.

    A small object rather than a closure so the table can be moved across
    devices/dtypes after construction (``fn.to(...)``) and still be the tensor
    the call reads — a closure would capture the original tensor and ignore an
    attribute reassignment. Carries metadata (``token_ids``, ``coords``,
    ``dims``, ``amp``, ``kind``) for the uptake/analysis code.
    """

    def __init__(self, table, *, kind="table", **meta):
        self.table = table                       # (vocab_size, n_embd), no grad
        self.kind = kind
        for k, v in meta.items():
            setattr(self, k, v)

    def __call__(self, idx):
        return self.table[idx]                   # (..., n_embd)

    def to(self, device=None, dtype=None):
        self.table = self.table.to(device=device, dtype=dtype)
        return self

    def __repr__(self):
        return (f"_TableOracle(kind={self.kind!r}, table={tuple(self.table.shape)}, "
                f"device={self.table.device}, dtype={self.table.dtype})")


def make_table_oracle(vocab_size, n_embd, table, *, kind="table",
                      device=None, dtype=None, **meta):
    """Frozen per-token-id oracle from an explicit ``(vocab_size, n_embd)`` table.

    The general carrier: any frozen additive feature that is a pure function of
    the token id is a row lookup ``table[idx]``. ``make_manifold_oracle`` builds
    on this. Pass ``device``/``dtype`` to place the table (forward casts the
    looked-up rows to the activation dtype regardless, so ``dtype`` is mostly a
    memory choice).
    """
    table = torch.as_tensor(table, dtype=(dtype or torch.float32), device=device)
    assert table.shape == (vocab_size, n_embd), (
        f"table must be (vocab_size, n_embd)=({vocab_size},{n_embd}), "
        f"got {tuple(table.shape)}")
    return _TableOracle(table, kind=kind, **meta)


# --------------------------------------------------------------------------- #
# Manifold coordinate generators: each returns (n, k), unit-norm per row
# --------------------------------------------------------------------------- #
def ring_coords(n, phase=0.0):
    """``n`` points evenly spaced on the unit circle: (n, 2). The cyclic case —
    e.g. months/hours/weekdays — and the direct analogue of a Fourier mode."""
    ang = phase + 2 * math.pi * torch.arange(n, dtype=torch.float32) / n
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)


def line_coords(n, lo=-1.0, hi=1.0):
    """``n`` points evenly spaced on a segment: (n, 1). A 1-D ordinal manifold
    (e.g. a magnitude/intensity scale)."""
    return torch.linspace(lo, hi, n, dtype=torch.float32).unsqueeze(1)


def sphere_coords(n):
    """``n`` points on the unit 2-sphere via the Fibonacci spiral: (n, 3).
    Near-uniform coverage for a 2-D manifold with no boundary."""
    i = torch.arange(n, dtype=torch.float32)
    z = 1.0 - 2.0 * (i + 0.5) / n                     # latitudes in (-1, 1)
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    theta = math.pi * (3.0 - math.sqrt(5.0)) * i      # golden angle
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=1)


def helix_coords(n, turns=2.0):
    """``n`` points on a helix (circle × line), unit-norm per row: (n, 3).
    A cyclic feature with a slowly varying second axis (e.g. day-of-year)."""
    ang = 2 * math.pi * turns * torch.arange(n, dtype=torch.float32) / max(n - 1, 1)
    z = torch.linspace(-1.0, 1.0, n, dtype=torch.float32)
    raw = torch.stack([torch.cos(ang), torch.sin(ang), z], dim=1)
    return raw / raw.norm(dim=1, keepdim=True)


# --------------------------------------------------------------------------- #
# Manifold oracle: place chosen token ids at manifold points in reserved dims
# --------------------------------------------------------------------------- #
def make_manifold_oracle(vocab_size, n_embd, token_ids, coords, *, dims=None,
                         amp=1.0, kind="manifold", device=None, dtype=None):
    """Frozen oracle placing ``token_ids`` at ``coords`` in reserved ``dims``.

    For token ``token_ids[i]`` we write ``amp * coords[i]`` into the residual
    dimensions ``dims`` (one dim per coordinate). All other token rows are zero,
    so only the chosen ids carry an oracle feature. With unit-norm ``coords``
    (the generators above) the per-token oracle norm is exactly ``amp``.

    Returns a ``_TableOracle`` with ``.token_ids``, ``.coords``, ``.dims``,
    ``.amp`` set for the analysis code.
    """
    coords = torch.as_tensor(coords, dtype=torch.float32)
    assert coords.ndim == 2, "coords must be (n_points, k)"
    n_pts, k = coords.shape
    token_ids = torch.as_tensor(list(token_ids), dtype=torch.long)
    assert token_ids.shape == (n_pts,), (
        f"need one coord row per token id ({n_pts} coords vs {token_ids.numel()} ids)")
    if dims is None:
        dims = list(range(k))
    dims = list(dims)
    assert len(dims) == k, f"need one residual dim per coordinate (k={k}, dims={dims})"
    assert max(dims) < n_embd, f"oracle dims {dims} exceed n_embd={n_embd}"

    dtype = dtype or torch.float32
    table = torch.zeros(vocab_size, n_embd, dtype=dtype, device=device)
    rows = token_ids.to(device=device)[:, None]                   # (n_pts, 1)
    cols = torch.as_tensor(dims, dtype=torch.long, device=device)[None, :]  # (1, k)
    table[rows, cols] = (amp * coords).to(dtype=dtype, device=device)

    return _TableOracle(table, kind=kind, token_ids=token_ids,
                        coords=coords, dims=dims, amp=float(amp))


def make_ring_oracle(vocab_size, n_embd, token_ids, *, dims=(0, 1), amp=1.0,
                     phase=0.0, device=None, dtype=None):
    """Convenience: place ``token_ids`` on a unit ring in two reserved dims.

    Equal-norm, evenly-spaced cyclic features — the geometric-manifold sibling of
    ``modular_addition.oracle.inject.make_fourier_oracle`` for a single pair.
    """
    coords = ring_coords(len(list(token_ids)), phase=phase)
    return make_manifold_oracle(vocab_size, n_embd, token_ids, coords, dims=dims,
                                amp=amp, kind="ring", device=device, dtype=dtype)


# --------------------------------------------------------------------------- #
# Attach / freeze helpers
# --------------------------------------------------------------------------- #
def attach_oracle(model, oracle_fn, *, inject=True):
    """Attach a frozen oracle to a materialized model and gate it on.

    Sets ``model.oracle_fn`` (read by the hook in ``GPT.forward``) and
    ``model.inject``. Moves the oracle table onto the model's device so the
    ``table[idx]`` lookup stays on-device. Call AFTER ``init_weights``.
    """
    if oracle_fn is not None and hasattr(oracle_fn, "to"):
        oracle_fn.to(device=model.get_device())
    model.oracle_fn = oracle_fn
    model.inject = inject
    return model


def detach_oracle(model):
    """Remove the oracle (forward reverts to the stock no-oracle path)."""
    model.oracle_fn = None
    return model


def freeze_params(model, patterns):
    """Hold parameters fixed at init by clearing ``requires_grad``.

    A parameter is frozen if any pattern in ``patterns`` is a substring of its
    ``named_parameters()`` name (e.g. ``["wte"]`` freezes the token embedding,
    ``["value_embeds"]`` the value embeddings). Returns the frozen names; raises
    if a non-empty ``patterns`` matches nothing, so a typo fails loudly instead
    of silently training everything. Mirrors
    ``modular_addition.oracle.sweep.freeze_params``.
    """
    patterns = list(patterns or [])
    frozen = []
    for name, p in model.named_parameters():
        if any(pat in name for pat in patterns):
            p.requires_grad_(False)
            frozen.append(name)
    if patterns and not frozen:
        raise ValueError(f"freeze patterns {patterns} matched no parameters")
    return frozen
