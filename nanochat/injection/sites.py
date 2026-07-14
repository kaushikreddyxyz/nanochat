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
    gate: float = 1.0              # trainer-level spec: a plain number is a loudness DIAL in donor units (rms(gate) = dial × L_ref; 1.0 = gemma-native packet loudness, resolved to absolute at startup). At the SITE level always absolute: a scalar fraction of residual RMS or a length-r vector; 0 = off. Absolute escapes: "abs:<n>", "auto[:t]".
    trainable_direction: bool = False
    direction_init: str = "orthonormal"   # "orthonormal" | "zeros" | "randn" | "file:<path.npy|.npz>"
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
        elif cfg.direction_init.startswith("file:"):
            # (r, n_embd) direction from a .npy/.npz (npz: key "D" if present,
            # else its sole array). Rows are taken verbatim; the site's z/rms(z)
            # renorm supplies the per-token scale. Frozen unless
            # trainable_direction=True. Relative paths resolve from the launch
            # CWD (run from the nanochat repo root).
            _path = cfg.direction_init[len("file:"):]
            _loaded = np.load(_path)
            _arr = ((_loaded["D"] if "D" in _loaded.files else _loaded[_loaded.files[0]])
                    if hasattr(_loaded, "files") else _loaded)
            d0 = torch.from_numpy(np.ascontiguousarray(_arr, dtype=np.float32))
            assert tuple(d0.shape) == (cfg.r, n_embd), \
                f"direction file {_path!r}: shape {tuple(d0.shape)} != (r={cfg.r}, n_embd={n_embd})"
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
# Donor-loudness gate: the plain-number gate is a DIAL in donor units — the
# absolute injected loudness resolves at startup to rms(gate) = dial × L_ref,
# where L_ref = gemma-2-2b's native median 54-concept packet loudness at the
# source's gemma layer (loudness.json subspace_total.ridge[L].p50, from
# attribution/measure_loudness.py; z-scores as fractions of gemma's residual
# stream — the site's own unit). dial 1.0 = "standard loudness" (donor-native).
# Per-channel mix is donor-proportional: g_c ∝ active_loudness.ridge[L].p50[c]
# / rms_c, scaled to the target. Site math is untouched — only how the CLI
# number becomes the absolute target. parse_gate_spec's contract is unchanged.
# --------------------------------------------------------------------------- #
def parse_donor_gate_spec(spec, default_stat="p50"):
    """'donor' | 'donor:p95' -> (True, stat); anything else -> (False, None).
    Alias of the dial: donor == dial 1.0; donor:<stat> == dial (stat/p50) at the
    source's layer (targets subspace_total.ridge[L][stat] directly)."""
    if isinstance(spec, str) and spec.startswith("donor"):
        rest = spec[len("donor"):].lstrip(":")
        stat = rest or default_stat
        if stat not in ("p50", "p90", "p95", "p99"):
            raise ValueError(f"--gate donor stat must be one of p50/p90/p95/p99, got {stat!r}")
        return True, stat
    return False, None


def classify_gate_spec(spec):
    """Gate-spec grammar -> (mode, value):
      number / "1.0" -> ("dial", float)   loudness dial in DONOR units (default 1.0;
                        0 = exactly off; requires loudness.json + a source gemma layer)
      "abs:<n>"      -> ("abs", float)    raw stream fraction (absolute; pre-dial semantics)
      "auto[:t]"     -> ("auto", float)   absolute target, channel-equalized calibration
      "donor[:stat]" -> ("donor", stat)   dial alias (donor == 1.0; donor:p95 == p95/p50)
      list/tuple     -> ("vector", list)  explicit absolute per-channel vector (resume/meta)
    """
    is_auto, val = parse_gate_spec(spec)
    if is_auto:
        return "auto", float(val)
    if isinstance(spec, str):
        is_donor, stat = parse_donor_gate_spec(spec)
        if is_donor:
            return "donor", stat
        if spec.startswith("abs"):
            rest = spec[len("abs"):].lstrip(":")
            if not rest:
                raise ValueError("--gate abs needs a number, e.g. abs:0.05 (raw stream fraction)")
            return "abs", float(rest)
        spec = float(spec)   # plain numeric string -> dial; junk raises ValueError
    if isinstance(spec, (list, tuple)):
        return "vector", list(spec)
    d = float(spec)
    if d < 0:
        raise ValueError(f"gate dial must be >= 0, got {d}")
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
            f"donor/dial gate: source {getattr(src, 'name', '?')!r} store meta.json lacks {missing} — "
            f"cannot identify which gemma layer's probe scores it predicts. A plain-number --gate is a "
            f"donor-loudness dial and needs that identity; use --gate abs:<number> (raw stream fraction) "
            f"or --gate auto[:target] for this source")
    raise ValueError(
        f"a donor-loudness dial gate requires a probe-score source (exposing a gemma layer + concept "
        f"columns); source {getattr(src, 'name', '?')!r} ({type(src).__name__}) exposes neither. "
        f"Use --gate abs:<number> (raw stream fraction) or --gate auto[:target] for this source")


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


def dial_gate_from_loudness(src, loudness, dial=None, stat=None, k=256, seed=0,
                            min_docs=16, floor=1e-3):
    """End-to-end dial-gate resolution from a loaded loudness.json + a source.
    Exactly one of ``dial`` / ``stat``: a dial targets rms(gate) = dial × L_ref
    (L_ref = subspace_total.ridge[<source layer>].p50); a stat (the donor[:stat]
    alias) targets subspace_total.ridge[L][stat] directly and records the
    equivalent dial. dial=0 -> exact-off scalar 0.0 WITHOUT touching src or
    loudness (loudness may be None). Resolves (layer, concepts) from the source,
    HARD-validates concept order, and calibrates the donor-proportional
    per-channel mix. Returns (gate_scalar_or_list, meta) — meta carries dial,
    L_ref, target_abs (the resolved absolute rms(gate)), stat, layer."""
    assert (dial is None) != (stat is None), "pass exactly one of dial= / stat="
    if stat is None:
        dial = float(dial)
        if dial < 0:
            raise ValueError(f"gate dial must be >= 0, got {dial}")
        if dial == 0.0:  # exact off — never needs the loudness artifact
            return 0.0, {"mode": "dial", "dial": 0.0, "L_ref": None, "target_abs": 0.0,
                         "stat": None, "layer": None}
    layer, src_concepts = donor_source_layer_concepts(src)
    validate_donor_concepts(loudness["concepts"], src_concepts)
    Lk = str(int(layer))
    st = loudness.get("subspace_total", {}).get("ridge", {}).get(Lk)
    if not st or "p50" not in st:
        raise ValueError(f"dial gate: loudness.json has no subspace_total.ridge[{Lk}].p50 for the "
                         f"source's gemma layer {layer}; layers present: "
                         f"{list(loudness.get('subspace_total', {}).get('ridge', {}))}")
    L_ref = float(st["p50"])
    if stat is not None:
        if stat not in st:
            raise ValueError(f"dial gate: subspace_total.ridge[{Lk}] has no stat {stat!r} (have {list(st)})")
        target = float(st[stat])
        dial = target / L_ref
    else:
        target = dial * L_ref
    act = loudness["ridge"]["active_loudness"].get(Lk)
    if act is None:
        raise ValueError(f"dial gate: loudness.json has no ridge.active_loudness[{Lk}]")
    gate, meta = calibrate_donor_gate(src, act["p50"], target, k=k, seed=seed,
                                      min_docs=min_docs, floor=floor)
    meta.update({"mode": "dial", "dial": float(dial), "L_ref": L_ref,
                 "target_abs": float(target), "stat": stat, "layer": int(layer)})
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
    try:
        return load_fn(fallback_repo), f"fallback:{fallback_repo}"
    except Exception as e:  # noqa: BLE001 — HARD: never silently fall back to absolute semantics
        raise RuntimeError(
            f"loudness.json unavailable (--loudness-json not given; not at the source store root; "
            f"fallback {fallback_repo} failed: {type(e).__name__}: {e}). A plain-number --gate is a "
            f"donor-loudness dial and REQUIRES loudness.json — pass --loudness-json PATH, or use "
            f"--gate abs:<number> for a raw stream fraction.") from e


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
