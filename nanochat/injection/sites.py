"""Injection sites: add an activation signal into the residual stream after a
chosen transformer block. A site decomposes into gate (loudness, NEVER
optimizable), activation (per-token content from an ActivationSource, NEVER
optimizable), and direction (the (r, n_embd) map, the only optionally-trainable
part). Site math per token:

    z = a @ D;   x = x + gate * rms(x).detach() * z / rms(z)

``gate`` is a scalar OR a length-r vector (one loudness per activation channel).
A vector gate scales channels before projection; its overall loudness is
rms(gate) and an all-zero vector is an exact no-op. A scalar broadcasts and is
byte-identical to the old scalar-only path. Neither form is ever optimized.

Design rationale and invariants: nanochat/injection/README.md.
"""
from dataclasses import dataclass

import numpy as np
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
    gate: float = 0.05             # scalar loudness (fraction of residual RMS) OR a length-r list of per-channel gates; 0 = off
    trainable_direction: bool = False
    direction_init: str = "orthonormal"   # "orthonormal" | "zeros" | "randn"
    direction_seed: int = 1337
    optim: str = "adamw"           # param group for a trainable direction ("adamw" | "muon")


class InjectionSite(nn.Module):
    def __init__(self, cfg: InjectionCfg, n_embd: int):
        super().__init__()
        self.cfg = cfg
        self.after_block = int(cfg.after_block)

        # Gate: a Parameter so autograd assigns it a gradient (loggable
        # want-signal), but _never_optimize keeps it out of every optimizer group.
        # Scalar (0-dim) OR length-r vector (per-channel loudness). "auto" must be
        # resolved to a concrete vector upstream (injection_train) before we get here.
        assert not isinstance(cfg.gate, str), \
            f"gate={cfg.gate!r} must be resolved to a scalar/list before InjectionSite (see injection_train auto-calibration)"
        g = torch.as_tensor(cfg.gate, dtype=torch.float32)
        assert g.ndim == 0 or (g.ndim == 1 and g.numel() == cfg.r), \
            f"gate must be a scalar or a length-{cfg.r} vector, got shape {tuple(g.shape)}"
        self.gate = nn.Parameter(g.clone())
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

    def freeze(self):
        self.direction.requires_grad_(False)

    def unfreeze(self):
        self.direction.requires_grad_(True)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """x: (B,T,n_embd) residual; a: (B,T,r) activation (no-grad content).
        Zero activation rows (BOS / missing doc) inject exactly 0 via the rms
        clamp — no branch, no NaN. rms(x) is detached: it calibrates amplitude,
        it is not a gradient path into x's own norm. gate.ndim is fixed per
        site, so the branch is a compile-time constant."""
        a = a.detach().to(x.dtype)
        g = self.gate.to(x.dtype)
        if g.ndim == 0:                       # scalar loudness (channels unweighted)
            a_scaled, overall = a, g
        else:                                 # per-channel: loudness rms(gate), mix gate/rms(gate)
            overall = g.pow(2).mean().sqrt()  # all-zero gate -> overall 0 -> exact no-op
            a_scaled = a * (g / overall.clamp_min(1e-8))
        z = a_scaled @ self.direction.to(x.dtype)
        z_hat = z / _rms(z)  # D's scale can never fight the gate for loudness
        return x + overall * _rms(x).detach() * z_hat


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


# --------------------------------------------------------------------------- #
# Auto gate calibration: turn a source's per-channel activation statistics into
# a per-channel gate vector that (a) equalizes each channel's typical RMS
# contribution and (b) has overall loudness rms(gate) == target. Deterministic
# in (source, seed); the resolved vector is stored in the cfg/checkpoint meta so
# resumes never recalibrate.
# --------------------------------------------------------------------------- #
def parse_gate_spec(spec, default_target=0.05):
    """'auto' | 'auto:0.1' -> (True, target); a number/list -> (False, value).
    injection_train resolves the auto case against the site's source."""
    if isinstance(spec, str) and spec.startswith("auto"):
        rest = spec[len("auto"):].lstrip(":")
        return True, (float(rest) if rest else default_target)
    return False, spec


def calibrate_auto_gate(src, target=0.05, k=256, seed=0, min_docs=16, floor=1e-3):
    """Per-channel gate from a source's sampled activation statistics.

    gate_c ∝ 1/rms_c on channels that are active (rms above ``floor`` and firing
    at all), 0 on dead channels, then scaled so rms(gate) == ``target``. Fails
    loudly if the source has < ``min_docs`` docs. Returns (gate_list, meta)."""
    rms, nz, n_docs, n_tokens = src.sample_activation_stats(k, seed)
    rms = np.asarray(rms, np.float64); nz = np.asarray(nz, np.float64)
    if n_docs < min_docs:
        raise RuntimeError(f"auto-gate: source {getattr(src, 'name', '?')!r} sampled only "
                           f"{n_docs} docs (< min_docs={min_docs}); too few to calibrate")
    active = (rms > floor) & (nz > 0.0)
    w = np.zeros(rms.shape[0], np.float64)
    w[active] = 1.0 / rms[active]
    wr = float(np.sqrt(np.mean(w ** 2)))
    if wr <= 0.0:
        raise RuntimeError(f"auto-gate: source {getattr(src, 'name', '?')!r} has no active "
                           f"channels (all rms <= floor={floor})")
    gate = (w * (target / wr)).astype(np.float32)
    meta = {"target": float(target), "k": int(k), "seed": int(seed),
            "n_docs": int(n_docs), "n_tokens": int(n_tokens),
            "n_active": int(active.sum()), "rms_gate": float(np.sqrt(np.mean(gate.astype(np.float64) ** 2))),
            "channel_rms": rms.astype(np.float32).tolist(), "nonzero_rate": nz.astype(np.float32).tolist()}
    return gate.tolist(), meta
