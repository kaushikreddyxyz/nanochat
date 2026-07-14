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
import hashlib
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


# --------------------------------------------------------------------------- #
# Donor-matched gate: instead of merely equalizing channels (auto), scale the
# gate so each concept's per-token loudness MATCHES how loud that concept is
# NATIVELY in the donor model (gemma-2-2b), read from loudness.json
# (attribution/measure_loudness.py). loudness.json interprets probe z-scores as
# fractions of gemma's residual-stream norm — the same units as the gate:
#   λ_c / ℓ_c   per-concept loudness (fraction of ‖x‖); active_loudness.ridge[L].p50
#   ℓ_tot       subspace loudness (the whole-packet gate analogue); subspace_total.ridge[L]
# Overall target = subspace_total.ridge[L][stat]; per-channel weight
# g_c ∝ active_loudness.ridge[L].p50[c] / rms_c. Sibling to parse_gate_spec
# (whose (bool, value) contract is unchanged) + calibrate_auto_gate.
# --------------------------------------------------------------------------- #
def parse_donor_gate_spec(spec, default_stat="p50"):
    """'donor' | 'donor:p95' -> (True, stat); anything else -> (False, None).
    stat ∈ {p50 (default), p90, p95, p99} selects the subspace_total quantile the
    overall gate loudness targets."""
    if isinstance(spec, str) and spec.startswith("donor"):
        rest = spec[len("donor"):].lstrip(":")
        stat = rest or default_stat
        if stat not in ("p50", "p90", "p95", "p99"):
            raise ValueError(f"--gate donor stat must be one of p50/p90/p95/p99, got {stat!r}")
        return True, stat
    return False, None


def donor_source_layer_concepts(src):
    """(gemma_layer, concept_column_names) the donor gate needs. Runtime/live
    probe-score sources expose ``.layer`` + ``.concepts``; a probe-scores store
    exposes them via meta.json (``layer``/``gemma_layer`` + ``concepts``/
    ``columns``). Raises a clear error for sources with no gemma-layer / concept
    identity (FnSource; a Qwen store whose meta lacks a layer field)."""
    layer = getattr(src, "layer", None)
    concepts = getattr(src, "concepts", None)
    if layer is not None and concepts is not None:
        return int(layer), list(concepts)
    meta = getattr(src, "meta", None)
    if isinstance(meta, dict):
        layer = meta.get("layer", meta.get("gemma_layer"))
        concepts = meta.get("concepts", meta.get("columns"))
        if layer is not None and concepts is not None:
            return int(layer), list(concepts)
        missing = [f for f, v in (("layer/gemma_layer", layer), ("concepts/columns", concepts)) if v is None]
        raise ValueError(
            f"donor gate: source {getattr(src, 'name', '?')!r} store meta.json lacks {missing} — "
            f"cannot identify which gemma layer's probe scores it predicts")
    raise ValueError(
        f"donor gate requires a probe-score source (exposing a gemma layer + concept columns); "
        f"source {getattr(src, 'name', '?')!r} ({type(src).__name__}) exposes neither")


def validate_donor_concepts(loudness_concepts, source_concepts):
    """HARD: loudness.json 'concepts' must EQUAL the source's column names exactly
    (order included) — the permutation lesson. Refuse to start otherwise."""
    if list(loudness_concepts) != list(source_concepts):
        raise ValueError(
            "donor gate: loudness.json 'concepts' != source column names (order included). "
            f"loudness[:5]={list(loudness_concepts)[:5]} source[:5]={list(source_concepts)[:5]}. "
            "Refusing to start — a permutation here silently mis-weights every channel.")


def calibrate_donor_gate(src, donor_loudness, target, k=256, seed=0, min_docs=16, floor=1e-3):
    """Per-channel donor-matched gate: gate_c ∝ donor_loudness[c]/rms_c on active
    channels (dead / zero-loudness -> 0), scaled so rms(gate) == target.
    ``donor_loudness[c]`` = the concept's native active ridge loudness
    (active_loudness.ridge[L].p50[c]); ``target`` = subspace_total.ridge[L][stat].
    ``rms_c`` from the source's sampled activation stats — the SAME independent
    sampling as auto (never the training buffer). Deterministic in (source, seed)
    so every DDP rank agrees. Returns (gate_list, meta)."""
    rms, nz, n_docs, n_tokens = src.sample_activation_stats(k, seed)
    rms = np.asarray(rms, np.float64); nz = np.asarray(nz, np.float64)
    donor = np.asarray(donor_loudness, np.float64)
    if donor.shape[0] != rms.shape[0]:
        raise ValueError(f"donor gate: loudness has {donor.shape[0]} concepts, source r={rms.shape[0]}")
    if n_docs < min_docs:
        raise RuntimeError(f"donor gate: source {getattr(src, 'name', '?')!r} sampled only {n_docs} docs "
                           f"(< min_docs={min_docs}); too few to calibrate")
    active = (rms > floor) & (nz > 0.0) & (donor > 0.0)
    w = np.zeros(rms.shape[0], np.float64)
    w[active] = donor[active] / rms[active]
    wr = float(np.sqrt(np.mean(w ** 2)))
    if wr <= 0.0:
        raise RuntimeError(f"donor gate: source {getattr(src, 'name', '?')!r} has no active channels "
                           f"(rms<=floor or zero donor loudness everywhere)")
    gate = (w * (target / wr)).astype(np.float32)
    meta = {"mode": "donor", "target": float(target), "k": int(k), "seed": int(seed),
            "n_docs": int(n_docs), "n_tokens": int(n_tokens), "n_active": int(active.sum()),
            "rms_gate": float(np.sqrt(np.mean(gate.astype(np.float64) ** 2))),
            "channel_rms": rms.astype(np.float32).tolist(),
            "donor_loudness": donor.astype(np.float32).tolist()}
    return gate.tolist(), meta


def donor_gate_from_loudness(src, loudness, stat="p50", k=256, seed=0, min_docs=16, floor=1e-3):
    """End-to-end donor-gate resolution from a loaded loudness.json + a source:
    resolve (layer, concepts) from the source, HARD-validate concept order, pull
    the per-concept active ridge loudness (p50) + the subspace_total target for
    ``stat`` at the source's gemma layer, and calibrate. Returns (gate_list, meta)."""
    layer, src_concepts = donor_source_layer_concepts(src)
    validate_donor_concepts(loudness["concepts"], src_concepts)
    Lk = str(int(layer))
    st = loudness.get("subspace_total", {}).get("ridge", {}).get(Lk)
    if st is None:
        raise ValueError(f"donor gate: loudness.json has no subspace_total.ridge[{Lk}] for the source's "
                         f"gemma layer {layer}; layers present: {list(loudness.get('subspace_total', {}).get('ridge', {}))}")
    if stat not in st:
        raise ValueError(f"donor gate: subspace_total.ridge[{Lk}] has no stat {stat!r} (have {list(st)})")
    target = float(st[stat])
    act = loudness["ridge"]["active_loudness"].get(Lk)
    if act is None:
        raise ValueError(f"donor gate: loudness.json has no ridge.active_loudness[{Lk}]")
    gate, meta = calibrate_donor_gate(src, act["p50"], target, k=k, seed=seed,
                                      min_docs=min_docs, floor=floor)
    meta.update({"stat": stat, "layer": int(layer)})
    return gate, meta


def discover_loudness_json(src, override, log, load_fn, fallback_repo="kaushikreddyxyz/climbmix-scored"):
    """Locate + load loudness.json. Precedence (always logs which artifact + from
    where — NO silent defaulting):
      1. ``override`` (--loudness-json): a local file, a local dir, or an HF
         dataset repo id;
      2. the source's own store root (score_loc / store_dir);
      3. ``fallback_repo``, with a LOUD log — valid because loudness is a property
         of gemma+probes+corpus, not of the scoring source.
    ``load_fn(loc)`` -> parsed dict for a location (injected for testing).
    Returns (loudness_dict, source_desc)."""
    if override:
        log(f"[donor-gate] loudness.json <- --loudness-json {override}")
        return load_fn(override), f"override:{override}"
    loc = getattr(src, "score_loc", None) or getattr(src, "store_dir", None)
    if loc is not None:
        try:
            d = load_fn(loc)
            log(f"[donor-gate] loudness.json <- source store root {loc}")
            return d, f"store:{loc}"
        except Exception as e:  # noqa: BLE001 — absent at store root -> fall through to fallback
            log(f"[donor-gate] no loudness.json at source store root {loc} "
                f"({type(e).__name__}: {e}); falling back")
    log("!" * 80)
    log(f"[donor-gate] FALLBACK: loudness.json from {fallback_repo}. VALID — loudness is a property "
        f"of gemma+probes+corpus, not of the scoring source.")
    log("!" * 80)
    return load_fn(fallback_repo), f"fallback:{fallback_repo}"


def gate_vector_hash(gate):
    """Stable content hash of a resolved gate (scalar or vector). DDP ranks
    compare this to assert bit-identical calibration."""
    a = np.ascontiguousarray(np.asarray(gate, np.float64).ravel())
    return hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest()


def assert_gate_identical_across_ranks(gate, name, log, all_gather_hash=None):
    """Assert every DDP rank computed a bit-identical gate. ``all_gather_hash(h)``
    -> list of every rank's hash (None => single process, trivially identical).
    Cheap + testable with simulated ranks. Returns the gate hash."""
    h = gate_vector_hash(gate)
    if all_gather_hash is None:
        return h
    hashes = list(all_gather_hash(h))
    if len(set(hashes)) != 1:
        raise RuntimeError(f"donor gate {name!r}: DDP ranks disagree on the calibrated gate "
                           f"(hashes {sorted(set(hashes))}) — calibration is not deterministic")
    log(f"[donor-gate] {name!r}: gate hash {h} identical across {len(hashes)} rank(s)")
    return h
