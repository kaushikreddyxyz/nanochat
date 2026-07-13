"""Generalized injection sites for nanochat pretraining (v2 of the oracle
coord injection; supersedes the single hard-coded inline site in gpt.py's
first patch).

An *injection* adds a feature signal into the residual stream after a chosen
block. It decomposes into exactly three parts with fixed optimizability rules:

  gate       -- loudness dial. Injected per-token RMS = gate * per-token
                RMS(residual), so gate=1.0 is "as loud as the stream itself",
                gate=0.05 reproduces the old inject_beta, gate=0 is exactly
                off (forward no-op AND no gradient reaches the direction).
                NEVER optimizable: registered as a parameter that is excluded
                from every optimizer group, so autograd still ASSIGNS a
                gradient every backward (loggable want-signal: dL/d gate),
                but no update step ever changes the value.
  activation -- the content: an (B, T, r) tensor per batch. Determined by a
                pluggable source (frozen Qwen encoder coords, gold probe
                scores, or any other function of the data). NEVER optimizable;
                providers must produce it without grad.
  direction  -- (r, n_embd) map from activation channels into the residual
                stream. The ONLY optionally-trainable part; controlled purely
                by freezing/unfreezing (`requires_grad`). Frozen + orthonormal
                init reproduces the old fixed-P behavior ("tabular
                injection"); unfrozen lets the model learn where the feature
                lives ("free injection").

Site math (per token):
    z      = a @ D                                   # (B,T,n_embd)
    z_hat  = z / rms(z)                              # direction only; D's
                                                     # scale can never fight
                                                     # the gate for loudness
    x      = x + gate * rms(x).detach() * z_hat
rms(x) is detached: it is a measurement of the stream used to calibrate
amplitude, not a path through which the injection should shape x's own norm
gradients. Zero activation rows (BOS / missing doc) give z == 0 and the
rms(z) clamp keeps the added term exactly 0 -- same no-op invariant as v1,
no branching, torch.compile-friendly.

Wiring: ``GPT.setup_injection_sites(cfgs)`` builds ``self.injection_sites =
build_sites(cfgs, n_embd)`` (an nn.ModuleDict, so trainable directions are
registered for the optimizer, DDP, and checkpoints) and ``GPT.forward`` takes
``acts: dict[name, (B,T,r)]`` from the ride-along dataloader; each site fires
after its own block index. Multiple sites may share or differ in block,
source, gate, and trainability.

Optimizer contract (GPT.setup_optimizer side):
  - params with `_never_optimize = True` (gates) go in NO param group;
  - frozen directions (requires_grad=False) are skipped by the usual filter;
  - trainable directions join the AdamW group by default (embedding-like map;
    override with cfg.optim="muon" per site if desired).
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn


def _rms(v: torch.Tensor) -> torch.Tensor:
    return v.pow(2).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()


def orthonormal_direction(r: int, n_embd: int, seed: int = 1337) -> torch.Tensor:
    """Fixed random orthonormal (r, n_embd) direction bank (rows orthonormal),
    the v1 P matrix transposed. Deterministic in seed: bit-identical to
    ``coords_store.make_orthonormal_P(n_embd, r, seed).T`` (same generator,
    same fp64 QR)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n_embd, r, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(a, mode="reduced")           # (n_embd, r), Q^T Q = I
    return q.t().contiguous().to(torch.float32)          # (r, n_embd)


@dataclass
class InjectionCfg:
    name: str                      # key into the dataloader's acts dict
    r: int                         # activation channels
    after_block: int               # inject after this block index
    gate: float = 0.05             # loudness (fraction of residual RMS); 0 = off
    trainable_direction: bool = False
    direction_init: str = "orthonormal"   # "orthonormal" | "zeros" | "randn"
    direction_seed: int = 1337
    channel_weights: list = field(default_factory=list)  # optional fixed per-
    # channel mix applied to the activation BEFORE projection (e.g. mute one
    # coord). Fixed buffer, never optimizable; empty = all-ones.
    optim: str = "adamw"           # param group for a trainable direction


class InjectionSite(nn.Module):
    def __init__(self, cfg: InjectionCfg, n_embd: int):
        super().__init__()
        self.cfg = cfg
        self.after_block = int(cfg.after_block)

        # --- gate: parameter with grads assigned, never in an optimizer group.
        self.gate = nn.Parameter(torch.tensor(float(cfg.gate)))
        self.gate._never_optimize = True

        # --- direction: the only optionally-trainable component.
        if cfg.direction_init == "orthonormal":
            d0 = orthonormal_direction(cfg.r, n_embd, cfg.direction_seed)
        elif cfg.direction_init == "zeros":
            # only sensible for a trainable direction (frozen zeros = dead site)
            d0 = torch.zeros(cfg.r, n_embd)
        elif cfg.direction_init == "randn":
            g = torch.Generator().manual_seed(cfg.direction_seed)
            d0 = torch.randn(cfg.r, n_embd, generator=g) / n_embd ** 0.5
        else:
            raise ValueError(f"unknown direction_init {cfg.direction_init!r}")
        self.direction = nn.Parameter(d0, requires_grad=bool(cfg.trainable_direction))

        # --- fixed per-channel mix (buffer: never a parameter).
        w = torch.tensor(cfg.channel_weights, dtype=torch.float32) \
            if cfg.channel_weights else torch.ones(cfg.r)
        assert w.numel() == cfg.r, "channel_weights must have r entries"
        self.register_buffer("channel_weights", w, persistent=True)

    def freeze(self):
        self.direction.requires_grad_(False)

    def unfreeze(self):
        self.direction.requires_grad_(True)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """x: (B,T,n_embd) residual; a: (B,T,r) activation (no-grad content)."""
        a = (a.detach() * self.channel_weights).to(x.dtype)   # content is data
        z = a @ self.direction.to(x.dtype)                    # (B,T,n_embd)
        z_hat = z / _rms(z)                                   # unit per-token RMS
        return x + self.gate.to(x.dtype) * _rms(x).detach() * z_hat


def build_sites(cfgs, n_embd: int) -> nn.ModuleDict:
    """cfgs: list[InjectionCfg] (or dicts). Returns ModuleDict name->site.
    Attach to the GPT module so directions are checkpointed / DDP-synced."""
    sites = nn.ModuleDict()
    for c in cfgs:
        if isinstance(c, dict):
            c = InjectionCfg(**c)
        if c.name in sites:
            raise ValueError(f"duplicate injection name {c.name!r}")
        sites[c.name] = InjectionSite(c, n_embd)
    return sites


def sites_by_block(sites: nn.ModuleDict):
    """{block_idx: [site, ...]} for the forward loop."""
    out = {}
    for s in sites.values():
        out.setdefault(s.after_block, []).append(s)
    return out


def optimizer_param_split(sites: nn.ModuleDict):
    """(adamw_params, muon_params) of TRAINABLE direction matrices only.
    Gates are never returned (never optimizable); frozen directions excluded."""
    adamw, muon = [], []
    for s in sites.values():
        if s.direction.requires_grad:
            (muon if s.cfg.optim == "muon" else adamw).append(s.direction)
    return adamw, muon


def reassert_optimizability(sites: nn.ModuleDict):
    """Re-impose each site's optimizability contract from its cfg.

    ``load_state_dict(..., assign=True)`` (used by base_train resume and
    checkpoint_manager.build_model) REPLACES the Parameter objects, which
    drops the gate's ``_never_optimize`` stamp and can change ``requires_grad``
    to that of the (detached) checkpoint tensors. Call this after ANY
    state-dict load into a model with injection sites, BEFORE setup_optimizer."""
    for s in sites.values():
        s.gate.requires_grad_(True)          # grads assigned (loggable), never stepped
        s.gate._never_optimize = True
        s.direction.requires_grad_(bool(s.cfg.trainable_direction))


# --------------------------------------------------------------------------- #
# Activation sources (loader-side). All produce numpy/torch WITHOUT grad; the
# site additionally detaches, so optimizability is impossible by construction.
#   * Qwen coords  -> CoordSource in nanochat.oracle.coords_store (doc-hash
#     keyed int8 store precomputed by scripts/precompute_coords.py). r = 14.
#   * probe scores -> a climbmix-scored store carrying gold-probe scores per
#     GEMMA token; injecting them per NANOCHAT token needs the same offline
#     repackaging precompute_coords.py does for Qwen (nanochat-tokenizer
#     alignment + doc-hash index). Build it as a precompute mode before first
#     use; it then plugs in through the same CoordSource duck-type.
#   * fn           -> any callable(text, n_tokens) -> (n_tokens, r) for
#     synthetic/control injections (positional ramps, random features, ...).
# --------------------------------------------------------------------------- #
class FnActivation:
    """Arbitrary-function activation source with the CoordSource duck-type:
    lookup(text, n_tokens) -> ((n_tokens, r) float32, key). Deterministic in
    doc content; never sees gradients (numpy-side)."""

    def __init__(self, fn, r: int):
        self.fn, self.r = fn, int(r)

    def lookup(self, text: str, n_tokens: int):
        out = self.fn(text, n_tokens)
        assert out.shape == (n_tokens, self.r)
        return out, 0

    def add_noise(self, z, key):
        return z
