"""Direction-subspace ablation: the two arms beyond on/off.

The four arms under study are {oracle ON, oracle OFF, direction ablated L0, direction
ablated ALL}. on/off are a loudness dial the existing harness already provides. The two
ablation arms are this module: they delete the subspace the oracle WRITES INTO from the
residual stream, and ask whether the model's concept behaviour was routing through it.

  x <- x - (x @ Q^T) @ Q          Q = orthonormal basis of rowspace(direction)

Applied via forward hooks on the transformer blocks, because that is exactly where a
site fires: `GPT.forward` runs `x = block(...)` and then applies any site registered for
that block index, so a block hook lands at the same point in the stream as the injection.

`L0` means the injecting block only (read from the site's own `after_block`, not
hardcoded). `ALL` means every block, which matters because `x0_lambdas` re-adds the
pre-trunk embedding at every layer — a single-point ablation does not stay ablated, and
the difference between the two arms is precisely how much the model re-derives.

EFFECTIVE RANK IS NOT r. The sphere arm's direction is a frozen 4-point circle: four
rows that lie in a 2-plane, so its row space is rank 2, not 4. Building the basis with
QR and assuming r rows would ablate two spurious directions and overstate the damage.
`direction_basis` uses an SVD with a relative tolerance and returns only the singular
vectors that carry real weight — always check the reported rank against r.
"""
import contextlib

import numpy as np
import torch


def direction_basis(direction, tol=1e-6):
    """Orthonormal basis of the direction's ROW space, as [k, n_embd].

    ``direction`` is the site's [r, n_embd] map from activation channels into the
    residual stream. k is the effective rank at relative tolerance ``tol`` and may be
    < r (see the module docstring on the sphere arm)."""
    D = torch.as_tensor(np.asarray(direction), dtype=torch.float32)
    assert D.ndim == 2, f"direction must be [r, n_embd], got {tuple(D.shape)}"
    _, S, Vh = torch.linalg.svd(D, full_matrices=False)
    if S.numel() == 0 or float(S[0]) == 0.0:
        return torch.zeros((0, D.shape[1]), dtype=torch.float32)
    k = int((S > tol * S[0]).sum())
    return Vh[:k].contiguous()


def project_out(x, Q):
    """x - (x @ Q^T) @ Q, computed in fp32 and cast back. Q is [k, n_embd] orthonormal."""
    if Q.numel() == 0:
        return x
    Qd = Q.to(device=x.device, dtype=torch.float32)
    xf = x.float()
    return (xf - (xf @ Qd.T) @ Qd).to(x.dtype)


def resolve_blocks(model, scope, after_block):
    """Block indices to ablate at. scope: 'none' | 'L0' | 'all'.

    'L0' is the SITE's block, not literally block 0 — `after_block` comes from the
    arm's own saved config so this stays correct if a run ever injects elsewhere."""
    n_layer = len(model.transformer.h)
    if scope == "none":
        return []
    if scope == "L0":
        assert 0 <= after_block < n_layer, f"after_block {after_block} out of range"
        return [after_block]
    if scope == "all":
        return list(range(n_layer))
    raise ValueError(f"unknown ablation scope {scope!r} (expected none/L0/all)")


@contextlib.contextmanager
def ablated(model, basis, blocks):
    """Project ``basis`` out of the residual stream after each block in ``blocks``.

    A no-op (and registers nothing) when ``blocks`` is empty or the basis is rank 0, so
    the 'none' scope is bit-identical to an unhooked forward. Hooks are always removed,
    including on exception — a leaked hook would silently corrupt every later arm."""
    handles = []
    if blocks and basis is not None and basis.numel() > 0:
        Q = basis.detach().clone()

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return (project_out(output[0], Q),) + output[1:]
            return project_out(output, Q)

        for i in blocks:
            handles.append(model.transformer.h[i].register_forward_hook(hook))
    try:
        yield len(handles)
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def ablated_map(model, basis_by_block):
    """Like `ablated`, but projects a possibly-DIFFERENT basis out after each block.

    ``basis_by_block`` is {block index: [k, n_embd] orthonormal basis}. A single shared
    basis (injected arm, one write direction) is expressed as {b: Q for b in blocks};
    a site-less baseline's 'ablate all' passes each layer's OWN seasonal basis. Empty
    map / all-empty bases register no hooks (exact no-op). Hooks always removed."""
    handles = []
    for i, basis in (basis_by_block or {}).items():
        if basis is None or basis.numel() == 0:
            continue
        Q = basis.detach().clone()

        def make_hook(Qb):
            def hook(_module, _inputs, output):
                if isinstance(output, tuple):
                    return (project_out(output[0], Qb),) + output[1:]
                return project_out(output, Qb)
            return hook

        handles.append(model.transformer.h[i].register_forward_hook(make_hook(Q)))
    try:
        yield len(handles)
    finally:
        for h in handles:
            h.remove()


def arm_plan(model, meta, site_name, scope):
    """(basis, blocks, info) for one ablation arm.

    Reads the direction off the loaded arm's own site, so each model is ablated along
    ITS OWN learned directions — the sphere and trainable arms do not share a subspace
    and must not share a basis."""
    sites = getattr(model, "injection_sites", None)
    if not sites or site_name not in sites:
        # the baseline has no site: there is no direction to ablate, so every
        # ablation arm degenerates to the plain model. Report it rather than crash.
        return None, [], {"scope": scope, "rank": 0, "r": 0, "blocks": [],
                          "note": "arm has no injection site (baseline)"}
    site = sites[site_name]
    D = site.direction.detach().float().cpu().numpy()
    basis = direction_basis(D)
    after_block = int(site.cfg.after_block)
    blocks = resolve_blocks(model, scope, after_block)
    info = {"scope": scope, "rank": int(basis.shape[0]), "r": int(D.shape[0]),
            "after_block": after_block, "blocks": blocks,
            "rank_deficient": bool(basis.shape[0] < D.shape[0])}
    return basis, blocks, info
