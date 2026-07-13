"""Injection sites: add an activation signal into the residual stream after a
chosen transformer block. A site decomposes into gate (loudness, NEVER
optimizable), activation (per-token content from an ActivationSource, NEVER
optimizable), and direction (the (r, n_embd) map, the only optionally-trainable
part). Site math per token:

    z = a @ D;   x = x + gate * rms(x).detach() * z / rms(z)

Design rationale and invariants: nanochat/injection/README.md.
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn


def _rms(v: torch.Tensor) -> torch.Tensor:
    return v.pow(2).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()


def orthonormal_direction(r: int, n_embd: int, seed: int = 1337) -> torch.Tensor:
    """Seeded random orthonormal (r, n_embd) direction bank. Bit-identical to
    ``sources.make_orthonormal_P(n_embd, r, seed).T`` (same generator, fp64 QR)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n_embd, r, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(a, mode="reduced")
    return q.t().contiguous().to(torch.float32)


@dataclass
class InjectionCfg:
    name: str                      # key into the dataloader's acts dict
    r: int                         # activation channels
    after_block: int               # inject after this block index
    gate: float = 1.0              # injected RMS as a fraction of residual RMS; 0 = off (the v1 run used 0.05)
    trainable_direction: bool = False
    direction_init: str = "orthonormal"   # "orthonormal" | "zeros" | "randn"
    direction_seed: int = 1337
    channel_weights: list = field(default_factory=list)  # fixed per-channel mix, never optimizable; empty = all-ones
    optim: str = "adamw"           # param group for a trainable direction ("adamw" | "muon")


class InjectionSite(nn.Module):
    def __init__(self, cfg: InjectionCfg, n_embd: int):
        super().__init__()
        self.cfg = cfg
        self.after_block = int(cfg.after_block)

        # Gate: a Parameter so autograd assigns it a gradient (loggable
        # want-signal), but _never_optimize keeps it out of every optimizer group.
        self.gate = nn.Parameter(torch.tensor(float(cfg.gate)))
        self.gate._never_optimize = True

        if cfg.direction_init == "orthonormal":
            d0 = orthonormal_direction(cfg.r, n_embd, cfg.direction_seed)
        elif cfg.direction_init == "zeros":
            d0 = torch.zeros(cfg.r, n_embd)  # only sensible trainable (frozen zeros = dead site)
        elif cfg.direction_init == "randn":
            g = torch.Generator().manual_seed(cfg.direction_seed)
            d0 = torch.randn(cfg.r, n_embd, generator=g) / n_embd ** 0.5
        else:
            raise ValueError(f"unknown direction_init {cfg.direction_init!r}")
        self.direction = nn.Parameter(d0, requires_grad=bool(cfg.trainable_direction))

        w = torch.tensor(cfg.channel_weights, dtype=torch.float32) \
            if cfg.channel_weights else torch.ones(cfg.r)
        assert w.numel() == cfg.r, "channel_weights must have r entries"
        self.register_buffer("channel_weights", w, persistent=True)

    def freeze(self):
        self.direction.requires_grad_(False)

    def unfreeze(self):
        self.direction.requires_grad_(True)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """x: (B,T,n_embd) residual; a: (B,T,r) activation (no-grad content).
        Zero activation rows (BOS / missing doc) inject exactly 0 via the rms
        clamp — no branch, no NaN. rms(x) is detached: it calibrates amplitude,
        it is not a gradient path into x's own norm."""
        a = (a.detach() * self.channel_weights).to(x.dtype)
        z = a @ self.direction.to(x.dtype)
        z_hat = z / _rms(z)  # D's scale can never fight the gate for loudness
        return x + self.gate.to(x.dtype) * _rms(x).detach() * z_hat


def build_sites(cfgs, n_embd: int) -> nn.ModuleDict:
    """cfgs: list[InjectionCfg | dict] -> ModuleDict name->site. Attach to the
    GPT module so directions are checkpointed / DDP-synced."""
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
    """(adamw_params, muon_params) of TRAINABLE directions only. Gates are
    never returned; frozen directions excluded."""
    adamw, muon = [], []
    for s in sites.values():
        if s.direction.requires_grad:
            (muon if s.cfg.optim == "muon" else adamw).append(s.direction)
    return adamw, muon


def reassert_optimizability(sites: nn.ModuleDict):
    """Re-impose each site's optimizability contract from its cfg. Required
    after ANY load_state_dict(..., assign=True): assign replaces the Parameter
    objects, dropping the gate's _never_optimize stamp and the direction's
    requires_grad. Call before setup_optimizer."""
    for s in sites.values():
        s.gate.requires_grad_(True)  # grads assigned (loggable), never stepped
        s.gate._never_optimize = True
        s.direction.requires_grad_(bool(s.cfg.trainable_direction))
