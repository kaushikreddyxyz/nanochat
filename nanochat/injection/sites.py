"""Injection sites: add an activation signal into the residual stream after a
chosen transformer block. Dose-responsive math, per token — score MAGNITUDE
survives, so a 4σ token injects proportionally more than a 2.1σ one:

    a_eff = relu(a - z0)
    x     = x + rms(x).detach() * ((a_eff * channel_scale) @ D/rms_rows(D))

A site decomposes into loudness (``channel_scale`` + ``threshold``, NEVER
optimizable), activation (per-token content from an ActivationSource, NEVER
optimizable), and direction (the (r, n_embd) map, the only optionally-trainable
part). ``channel_scale`` is resolved at startup by calibrate_dose_gate.

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
    after_block: int = 0           # inject after this block index
    loudness: object = 1.0         # trainer-level spec: a plain number is a DIAL in donor units (dial 1.0 = the median firing event injects at gemma's own median per-concept active loudness for THIS site's concepts); "abs:<n>" targets that median directly as a fraction of residual RMS. Resolved into channel_scale at startup.
    trainable_direction: bool = False
    direction_init: str = "orthonormal"   # "orthonormal" | "zeros" | "randn" | "file:<path.npy|.npz>"
    direction_seed: int = 1337
    optim: str = "adamw"           # param group for a trainable direction ("adamw" | "muon")
    threshold: object = 2.0        # relu knee z0 in σ units, scalar or length-r
    channel_scale: object = None   # frozen length-r loudness per unit of relu(a-z0); calibrate_dose_gate fills it


def initial_direction(cfg: InjectionCfg, n_embd: int) -> torch.Tensor:
    """The site's (r, n_embd) direction at init, per cfg.direction_init. Split out
    so dose calibration can measure the ACTUAL D before the site is built."""
    if cfg.direction_init == "orthonormal":
        return orthonormal_direction(cfg.r, n_embd, cfg.direction_seed)
    if cfg.direction_init == "zeros":
        return torch.zeros(cfg.r, n_embd)  # only sensible trainable (frozen zeros = dead site)
    if cfg.direction_init == "randn":
        g = torch.Generator().manual_seed(cfg.direction_seed)
        return torch.randn(cfg.r, n_embd, generator=g) / n_embd ** 0.5
    if cfg.direction_init.startswith("file:"):
        # (r, n_embd) direction from a .npy/.npz (npz: key "D" if present, else its
        # sole array). Rows are taken verbatim; the site renormalizes them to unit
        # RMS. Frozen unless trainable_direction=True. Relative paths resolve from
        # the launch CWD (run from the nanochat repo root).
        path = cfg.direction_init[len("file:"):]
        loaded = np.load(path)
        arr = ((loaded["D"] if "D" in loaded.files else loaded[loaded.files[0]])
               if hasattr(loaded, "files") else loaded)
        d0 = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        assert tuple(d0.shape) == (cfg.r, n_embd), \
            f"direction file {path!r}: shape {tuple(d0.shape)} != (r={cfg.r}, n_embd={n_embd})"
        return d0
    raise ValueError(f"unknown direction_init {cfg.direction_init!r}")


class InjectionSite(nn.Module):
    def __init__(self, cfg: InjectionCfg, n_embd: int):
        super().__init__()
        self.cfg = cfg
        self.after_block = int(cfg.after_block)

        # channel_scale / threshold are Parameters so autograd assigns them a gradient
        # (loggable want-signal), but _never_optimize keeps them out of every optimizer
        # group. The loudness spec must be resolved upstream (injection_train) first.
        assert cfg.channel_scale is not None, \
            (f"site {cfg.name!r}: channel_scale is unresolved — loudness={cfg.loudness!r} must be "
             f"calibrated by calibrate_dose_gate before the site is built (see injection_train)")
        cs = torch.as_tensor(cfg.channel_scale, dtype=torch.float32)
        assert cs.ndim == 1 and cs.numel() == cfg.r, \
            f"channel_scale must be a length-{cfg.r} vector, got shape {tuple(cs.shape)}"
        th = torch.as_tensor(cfg.threshold, dtype=torch.float32)
        assert th.ndim == 0 or (th.ndim == 1 and th.numel() == cfg.r), \
            f"threshold must be a scalar or a length-{cfg.r} vector, got shape {tuple(th.shape)}"
        self.channel_scale = nn.Parameter(cs.clone())
        self.channel_scale._never_optimize = True
        self.threshold = nn.Parameter(th.clone())
        self.threshold._never_optimize = True

        self.direction = nn.Parameter(initial_direction(cfg, n_embd),
                                      requires_grad=bool(cfg.trainable_direction))

    def freeze(self):
        self.direction.requires_grad_(False)

    def unfreeze(self):
        self.direction.requires_grad_(True)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """x: (B,T,n_embd) residual; a: (B,T,r) activation (no-grad content).
        Sub-threshold / zero / negative rows relu to exactly 0 and inject bitwise 0 —
        no branch, no NaN, torch.compile-friendly. rms(x) is detached: it calibrates
        amplitude, it is not a gradient path into x's own norm."""
        a = a.detach().to(x.dtype)
        u = (a - self.threshold.to(x.dtype)).clamp_min(0) * self.channel_scale.to(x.dtype)
        d = self.direction.to(x.dtype)
        # Unit-RMS rows reparameterized here, not by a post-step hook: checkpoint-safe
        # and self-enforcing. RMS (not L2) so channel_scale reads directly as a fraction
        # of residual RMS, with no sqrt(n_embd) factor. D's scale is a pure gauge.
        return x + _rms(x).detach() * (u @ (d / _rms(d)))


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
    """(adamw_params, muon_params) of TRAINABLE directions only. Loudness params are
    never returned; frozen directions excluded."""
    adamw, muon = [], []
    for s in sites.values():
        if s.direction.requires_grad:
            (muon if s.cfg.optim == "muon" else adamw).append(s.direction)
    return adamw, muon


def reassert_optimizability(sites: nn.ModuleDict):
    """Re-impose each site's optimizability contract from its cfg. Required
    after ANY load_state_dict(..., assign=True): assign replaces the Parameter
    objects, dropping the loudness params' _never_optimize stamp and the direction's
    requires_grad. Call before setup_optimizer."""
    for s in sites.values():
        for p in (s.channel_scale, s.threshold):
            p.requires_grad_(True)  # grads assigned (loggable), never stepped
            p._never_optimize = True
        s.direction.requires_grad_(bool(s.cfg.trainable_direction))


# --------------------------------------------------------------------------- #
# Loudness grammar. Exactly two forms — the relu does the gating, this only sets
# overall volume:
#   number    -> ("dial", float)  in DONOR units; dial 1.0 = the median firing event
#                injects at gemma's own median per-concept active loudness for THIS
#                site's concepts. Needs loudness.json + a source gemma layer.
#   "abs:<n>" -> ("abs", float)   absolute target for that median, as a fraction of
#                residual RMS. The escape hatch for sources with no gemma identity.
# --------------------------------------------------------------------------- #
def classify_loudness_spec(spec):
    """Loudness spec -> (mode, value). Raises on anything outside the two forms."""
    if isinstance(spec, str):
        if spec.startswith("abs"):
            rest = spec[len("abs"):].lstrip(":")
            if not rest:
                raise ValueError("--loudness abs needs a number, e.g. abs:0.03 (fraction of residual RMS)")
            v = float(rest)
            if v < 0:
                raise ValueError(f"--loudness abs must be >= 0, got {v}")
            return "abs", v
        spec = float(spec)       # plain numeric string -> dial; junk raises ValueError
    if isinstance(spec, (list, tuple)):
        raise ValueError(f"--loudness no longer takes a per-channel vector ({list(spec)[:4]}...); "
                         "per-event equalization derives the per-channel mix. Use a dial or abs:<n>.")
    d = float(spec)
    if d < 0:
        raise ValueError(f"loudness dial must be >= 0, got {d}")
    return "dial", d


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
            f"donor dial: source {getattr(src, 'name', '?')!r} store meta.json lacks {missing} — "
            f"cannot identify which gemma layer's probe scores it predicts. A plain-number --loudness "
            f"is a donor dial and needs that identity; use --loudness abs:<number> (a fraction of "
            f"residual RMS) for this source")
    raise ValueError(
        f"a donor-loudness dial requires a probe-score source (exposing a gemma layer + concept "
        f"columns); source {getattr(src, 'name', '?')!r} ({type(src).__name__}) exposes neither. "
        f"Use --loudness abs:<number> (a fraction of residual RMS) for this source")


def validate_donor_concepts(loudness_concepts, source_concepts):
    """Map each source column to its loudness entry BY NAME -> (r,) int index array.
    Real sites are SUBSETS of the 54 (seasons r=4, weekdays r=7), so equality was too
    strict; indexing by name makes positional order irrelevant by construction, which
    is a stronger form of the permutation lesson than the old order check. HARD error
    on any unknown or duplicated name."""
    pos = {}
    for i, c in enumerate(loudness_concepts):
        if c in pos:
            raise ValueError(f"donor gate: loudness.json 'concepts' lists {c!r} twice — "
                             f"ambiguous, cannot index by name")
        pos[c] = i
    src = list(source_concepts)
    missing = [c for c in src if c not in pos]
    if missing:
        raise ValueError(
            f"donor gate: {len(missing)} source concept(s) absent from loudness.json 'concepts': "
            f"{missing[:10]}. loudness lists {len(pos)}, source needs {len(src)}. Refusing to "
            "start — an unknown concept would silently mis-weight its channel.")
    return np.asarray([pos[c] for c in src], np.int64)


def _per_concept_loudness(loudness, Lk, idx, what="dose gate"):
    """active_loudness.ridge[L].p50 sliced to the source's own concepts, by name."""
    act = loudness.get("ridge", {}).get("active_loudness", {}).get(Lk)
    if act is None or "p50" not in act:
        raise ValueError(f"{what}: loudness.json has no ridge.active_loudness[{Lk}].p50")
    p50 = np.asarray(act["p50"], np.float64)
    if p50.shape[0] != len(loudness["concepts"]):
        raise ValueError(f"{what}: active_loudness[{Lk}].p50 has {p50.shape[0]} entries but "
                         f"'concepts' lists {len(loudness['concepts'])} — cannot index by name")
    return p50[idx]


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
    try:
        return load_fn(fallback_repo), f"fallback:{fallback_repo}"
    except Exception as e:  # noqa: BLE001 — HARD: never silently fall back to absolute semantics
        raise RuntimeError(
            f"loudness.json unavailable (--loudness-json not given; not at the source store root; "
            f"fallback {fallback_repo} failed: {type(e).__name__}: {e}). A plain-number --loudness is "
            f"a donor dial and REQUIRES loudness.json — pass --loudness-json PATH, or use "
            f"--loudness abs:<number> for a fraction of residual RMS.") from e


# --------------------------------------------------------------------------- #
# Dose calibration: resolve amplitude="dose"'s frozen channel_scale. Two stages:
# PER-EVENT equalization (channel_scale_c ∝ 1/E_c, E_c = rms of channel c's
# activation OVER ITS FIRING TOKENS ONLY) so every concept's typical firing event
# injects equally regardless of how often it fires; then one global rescale so the
# MEDIAN injected loudness over firing tokens is dial × L_ref — the same donor
# reference the constant-mode dial uses. Deterministic in (source, seed).
# --------------------------------------------------------------------------- #
def _injected_loudness(a_eff, w, d_hat, chunk=4096):
    """Per-token rms(u @ D_hat) for u = a_eff * w, chunked (n × n_embd is large)."""
    out = np.empty(a_eff.shape[0], np.float64)
    for i in range(0, a_eff.shape[0], chunk):
        z = (a_eff[i:i + chunk] * w) @ d_hat
        out[i:i + chunk] = np.sqrt((z ** 2).mean(1))
    return out


def calibrate_dose_gate(src, loudness, direction, dial=1.0, abs_target=None, threshold=2.0, k=256,
                        seed=0, min_events=50, min_docs=16, log=print):
    """Frozen per-channel scale for a site. ``direction`` is the site's ACTUAL initial D
    (rows are non-orthogonal in the sphere arm, so a quadrature estimate of loudness
    would be wrong). ``abs_target`` sets the median injected loudness directly and needs
    no loudness.json; otherwise the dial reference is the SUBSET's own median per-concept
    active loudness. Returns (channel_scale_list, meta)."""
    rows, n_docs, n_tokens = src.sample_activation_rows(k, seed)
    if n_docs < min_docs:
        raise RuntimeError(f"dose gate: source {getattr(src, 'name', '?')!r} sampled only {n_docs} docs "
                           f"(< min_docs={min_docs}); too few to calibrate")
    layer = idx = donor_pc = L_ref = L_total = None
    src_concepts = None
    if abs_target is None:
        layer, src_concepts = donor_source_layer_concepts(src)
        idx = validate_donor_concepts(loudness["concepts"], src_concepts)
        Lk = str(int(layer))
        st = loudness.get("subspace_total", {}).get("ridge", {}).get(Lk)
        if not st or "p50" not in st:
            raise ValueError(f"dose gate: loudness.json has no subspace_total.ridge[{Lk}].p50 for the "
                             f"source's gemma layer {layer}; layers present: "
                             f"{list(loudness.get('subspace_total', {}).get('ridge', {}))}")
        L_total = float(st["p50"])        # WHOLE 54-concept subspace — logged for contrast, not used
        # The reference is the SUBSET's own median per-concept active loudness: concepts
        # add in quadrature, so the 54-subspace total would over-inject an r-concept site
        # by ~sqrt(54/r). dial 1.0 = "each firing concept about as loud as it is in gemma".
        donor_pc = _per_concept_loudness(loudness, Lk, idx)
        L_ref = float(np.median(donor_pc))
        if L_ref <= 0.0:
            raise ValueError(f"dose gate: median per-concept active loudness at layer {layer} is {L_ref} "
                             f"over the source's {len(src_concepts)} concepts — nothing to dial against")
    dial = float(dial)
    if dial < 0:
        raise ValueError(f"dose dial must be >= 0, got {dial}")
    a = np.asarray(rows, np.float64)
    r = a.shape[1]
    if src_concepts is not None and len(src_concepts) != r:
        raise ValueError(f"dose gate: source names {len(src_concepts)} concepts but yields r={r} channels")
    names = list(src_concepts) if src_concepts is not None else [f"ch{i}" for i in range(r)]
    z0 = np.broadcast_to(np.asarray(threshold, np.float64), (r,)).astype(np.float64)
    a_eff = np.maximum(a - z0, 0.0)              # the SAME relu the site applies
    n_events = (a_eff > 0.0).sum(0)
    fired = n_events > 0
    E = np.zeros(r, np.float64)                  # rms CONDITIONAL on firing (per-event, not per-total)
    E[fired] = np.sqrt((a_eff ** 2).sum(0)[fired] / n_events[fired])

    w = np.zeros(r, np.float64)
    weak = fired & (n_events < min_events)
    good = fired & ~weak
    w[good] = 1.0 / E[good]
    E_typical = None
    if weak.any():
        if not good.any():
            raise RuntimeError(f"dose gate: every firing channel fired < min_events={min_events} times; "
                               f"no well-measured channel to impute from (raise --gate-k or lower it)")
        # Per-event equalization means "equally loud when it fires", so a channel we
        # cannot measure is treated as a TYPICAL channel — same units (1/σ), not donor loudness.
        E_typical = float(np.median(E[good]))
        E[weak] = E_typical
        w[weak] = 1.0 / E_typical
        log("!" * 80)
        log(f"[dose-gate] FALLBACK: {int(weak.sum())} channel(s) fired < min_events={min_events} times; "
            f"imputing the median well-measured E_c={E_typical:.4f}: "
            f"{[(names[i], int(n_events[i])) for i in np.flatnonzero(weak)]}")
        log("!" * 80)
    if (~fired).any():
        log("!" * 80)
        log(f"[dose-gate] DEAD: {int((~fired).sum())} channel(s) never crossed the threshold -> "
            f"channel_scale 0: {[names[i] for i in np.flatnonzero(~fired)]}")
        log("!" * 80)

    d = direction.detach().cpu().numpy() if hasattr(direction, "detach") else direction
    d = np.ascontiguousarray(d, np.float64)
    if d.shape[0] != r:
        raise ValueError(f"dose gate: direction has {d.shape[0]} rows, source r={r}")
    d_hat = d / np.sqrt(np.maximum((d ** 2).mean(1, keepdims=True), 1e-8))   # _rms's clamp, in numpy
    base = _injected_loudness(a_eff, w, d_hat)
    firing = base > 0.0    # tokens that ACTUALLY inject (a token firing only on a zeroed channel does not)
    if not firing.any():
        raise RuntimeError(f"dose gate: source {getattr(src, 'name', '?')!r} injects on 0 of {n_tokens} "
                           f"sampled tokens at threshold={threshold}; nothing to calibrate against")
    target = float(abs_target) if abs_target is not None else dial * L_ref
    s = target / float(np.median(base[firing]))
    scale = (w * s).astype(np.float32)
    ladder = base[firing] * s
    meta = {"mode": "abs" if abs_target is not None else "dial",
            "dial": None if abs_target is not None else dial,
            "layer": None if layer is None else int(layer), "L_ref": L_ref,
            "L_ref_subspace_total": L_total,
            "donor_per_concept": None if donor_pc is None else donor_pc.astype(np.float32).tolist(),
            "concepts": names,
            "concept_index": None if idx is None else idx.astype(np.int64).tolist(),
            "target_median": float(target), "threshold": z0.astype(np.float32).tolist(),
            "k": int(k), "seed": int(seed), "n_docs": int(n_docs), "n_tokens": int(n_tokens),
            "min_events": int(min_events), "n_events": n_events.astype(np.int64).tolist(),
            "event_rms": E.astype(np.float32).tolist(),   # weak channels carry the imputed median
            "event_rms_imputed": E_typical,
            "firing_rate": (n_events / max(n_tokens, 1)).astype(np.float32).tolist(),
            "n_imputed": int(weak.sum()), "n_dead": int((~fired).sum()),
            "n_firing_tokens": int(firing.sum()), "firing_token_rate": float(firing.mean()),
            "channel_scale": scale.tolist(),
            "injected_loudness": {"p50": float(np.percentile(ladder, 50)),
                                  "p90": float(np.percentile(ladder, 90)),
                                  "p95": float(np.percentile(ladder, 95)),
                                  "p99": float(np.percentile(ladder, 99)),
                                  "max": float(ladder.max())},
            "donor_loudness": ({q: float(v) for q, v in st.items() if isinstance(v, (int, float))}
                               if abs_target is None else None)}
    return scale.tolist(), meta


# --------------------------------------------------------------------------- #
# Realized-loudness check. calibrate_dose_gate measures PRE-alignment (gemma-grid)
# scores; the site sees POST-alignment rows, and mean-pooling alignment gives
# mean(relu(·)) >= relu(mean(·)) — the relu is convex, so calibration can only
# UNDER-estimate wherever a nanochat token pools several gemma tokens (and is exact
# wherever one gemma token broadcasts to several). The size of that gap is an
# empirical property of tokenizer granularity, so we measure it on the real stream
# rather than assume it. Reports a correction factor; never applies one.
# --------------------------------------------------------------------------- #
def _loudness_ladder(v):
    return {"p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),
            "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)),
            "max": float(v.max())}


def realized_loudness_report(act_batches, threshold, channel_scale, direction, target,
                             min_firing_tokens=50_000, tol=0.25):
    """Injected-loudness ladder over the REAL post-alignment activation stream vs the
    pre-alignment calibration target. Pulls from ``act_batches`` ((n, r) or (B, T, r))
    only until ``min_firing_tokens`` inject, so the caller can replay exactly what was
    consumed. Returns meta; ``within_tol`` False means the dial is off by ``correction``."""
    d = direction.detach().cpu().numpy() if hasattr(direction, "detach") else direction
    d = np.ascontiguousarray(d, np.float64)
    d_hat = d / np.sqrt(np.maximum((d ** 2).mean(1, keepdims=True), 1e-8))
    w = np.asarray(channel_scale, np.float64)
    z0 = np.asarray(threshold, np.float64)
    kept, n_tokens, n_batches, n_fire = [], 0, 0, 0
    for a in act_batches:
        a = a.detach().cpu().numpy() if hasattr(a, "detach") else a
        a = np.ascontiguousarray(a, np.float64).reshape(-1, w.shape[0])
        v = _injected_loudness(np.maximum(a - z0, 0.0), w, d_hat)
        v = v[v > 0.0]
        kept.append(v)
        n_tokens += a.shape[0]
        n_batches += 1
        n_fire += v.size
        if n_fire >= min_firing_tokens:
            break
    fired = np.concatenate(kept) if kept else np.zeros(0, np.float64)
    if fired.size == 0:
        return {"n_batches": n_batches, "n_tokens": n_tokens, "n_firing_tokens": 0,
                "ladder": None, "target": float(target), "realized_p50": None,
                "rel_dev": None, "correction": None, "within_tol": False, "tol": float(tol)}
    ladder = _loudness_ladder(fired)
    p50 = ladder["p50"]
    rel = (p50 - target) / target if target > 0 else float("inf")
    return {"n_batches": n_batches, "n_tokens": n_tokens, "n_firing_tokens": int(fired.size),
            "firing_token_rate": float(fired.size / max(n_tokens, 1)), "ladder": ladder,
            "target": float(target), "realized_p50": p50, "rel_dev": float(rel),
            "correction": float(target / p50), "within_tol": bool(abs(rel) <= tol),
            "tol": float(tol)}


def log_realized_loudness(rep, name, calib_ladder, donor_ladder, log=print):
    """Log the realized ladder beside the calibration-time and donor ladders, and WARN
    loudly (never auto-correct) when realized p50 misses the target by > tol."""
    if rep["ladder"] is None:
        log("!" * 80)
        log(f"[dose-check] {name!r}: NO token injected over {rep['n_tokens']} sampled tokens "
            f"({rep['n_batches']} batches) — the site is silent on the real stream. Check the threshold.")
        log("!" * 80)
        return
    log(f"[dose-check] {name!r}: realized over {rep['n_firing_tokens']} firing / {rep['n_tokens']} tokens "
        f"({100 * rep['firing_token_rate']:.2f}%, {rep['n_batches']} peeked batches, replayed into training)")
    for q in ("p50", "p90", "p95", "p99", "max"):
        c = f"{calib_ladder[q]:.4f}" if calib_ladder and q in calib_ladder else "-"
        dv = f"{donor_ladder[q]:.4f}" if donor_ladder and q in donor_ladder else "-"
        log(f"[dose-check] {name!r} ladder: {q:>4} realized={rep['ladder'][q]:.4f} calib={c} donor={dv}")
    if rep["within_tol"]:
        log(f"[dose-check] {name!r}: realized p50 {rep['realized_p50']:.4f} vs target {rep['target']:.4f} "
            f"({100 * rep['rel_dev']:+.1f}%, within +/-{100 * rep['tol']:.0f}%)")
        return
    log("!" * 80)
    log(f"[dose-check] WARNING {name!r}: realized p50 {rep['realized_p50']:.4f} misses the target "
        f"{rep['target']:.4f} by {100 * rep['rel_dev']:+.1f}% (tol +/-{100 * rep['tol']:.0f}%). Calibration "
        f"samples PRE-alignment gemma rows; the site sees POST-alignment rows, and mean-pooling makes "
        f"mean(relu) >= relu(mean). NOT auto-corrected — multiply the dial by {rep['correction']:.4f} to hit "
        f"the target.")
    log("!" * 80)


def loudness_vector_hash(vec):
    """Stable content hash of a resolved channel_scale. DDP ranks compare this to
    assert bit-identical calibration."""
    a = np.ascontiguousarray(np.asarray(vec, np.float64).ravel())
    return hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest()


def assert_loudness_identical_across_ranks(vec, name, log, all_gather_hash=None):
    """Assert every DDP rank computed a bit-identical channel_scale. ``all_gather_hash(h)``
    -> list of every rank's hash (None => single process, trivially identical).
    Cheap + testable with simulated ranks. Returns the hash."""
    h = loudness_vector_hash(vec)
    if all_gather_hash is None:
        return h
    hashes = list(all_gather_hash(h))
    if len(set(hashes)) != 1:
        raise RuntimeError(f"site {name!r}: DDP ranks disagree on the calibrated channel_scale "
                           f"(hashes {sorted(set(hashes))}) — calibration is not deterministic")
    log(f"[dose-gate] {name!r}: channel_scale hash {h} identical across {len(hashes)} rank(s)")
    return h
