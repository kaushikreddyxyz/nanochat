"""Injection-on-vs-off eval harness (foundation imported by run_evals/causal/open_gen):
load_model, attach_site, GemmaScorer, build_acts, forward_metrics, ce_report. Shared by
every concept family; PINNED interface — the sibling drivers and tests code against it.
"""
import os
import json
import sys
import contextlib

import numpy as np
import torch
import torch.nn.functional as F

# --- repo wiring -----------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.dirname(_HERE)                         # runs/lib
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))  # nanochat repo root

GEMMA_LAYER = 8
GEMMA_MODEL = "google/gemma-2-2b"
DEFAULT_SCORE_STORE = "kaushikreddyxyz/climbmix-scored"
DEFAULT_THRESHOLD = 2.0        # the site's relu knee, in sigma units


def _load_shared(name, filename):
    """Load a runs/lib module by ABSOLUTE file path (collision-proof, CWD-independent
    — mirrors the config's file-path 'class' form; an installed PyPI ``runs`` package
    shadows the local runs/ dir)."""
    import importlib.util
    path = os.path.abspath(os.path.join(_LIB_DIR, filename))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_source_class():
    concepts = _load_shared("_concepts_eval", "concepts.py")
    sys.modules["concepts"] = concepts         # probe_source.py imports it by bare name
    return _load_shared("_probe_source_eval", "probe_source.py").ConceptProbeScoreSource, concepts


try:
    ConceptProbeScoreSource, CONCEPTS = _load_source_class()
except Exception:  # pragma: no cover — optional for pure-numpy users of ce_report etc.
    ConceptProbeScoreSource = CONCEPTS = None


def family_concepts(family):
    """A family's activation-channel order (runs/lib/concepts.py store order) — what a
    channel index MEANS. The single source; never re-list concepts in an eval module."""
    assert CONCEPTS is not None, "runs/lib/concepts.py failed to import"
    return list(CONCEPTS.get_family(family).store_order)


# --- injected nanochat tokenizer (module-level default) --------------------
_NANO_ENC = None


def set_nano_tokenizer(nano_enc):
    """Register the nanochat tiktoken Encoding once; ``build_acts`` uses it when
    ``nano_enc`` is not passed explicitly. On the pod this is
    ``get_tokenizer().enc``; in tests it is a byte-level fake."""
    global _NANO_ENC
    _NANO_ENC = nano_enc


def _resolve_nano_enc(nano_enc):
    """Nano tokenizer resolution: explicit arg -> module default -> pod fallback
    (lazily construct get_tokenizer().enc). Tests ALWAYS pass an explicit fake, so
    the rustbpe fallback never fires off-pod. A RustBPETokenizer wrapper is
    unwrapped to its tiktoken ``.enc`` (which has encode_ordinary +
    decode_single_token_bytes)."""
    enc = nano_enc if nano_enc is not None else _NANO_ENC
    if enc is None:  # pod convenience: siblings call build_acts without injecting
        from nanochat.tokenizer import get_tokenizer
        enc = get_tokenizer()
    return getattr(enc, "enc", enc)  # unwrap RustBPETokenizer -> tiktoken Encoding


# --------------------------------------------------------------------------- #
# 1. Checkpoint loading
#
# The three helpers below are inlined (byte-for-byte) from
# nanochat.checkpoint_manager so load_model has ZERO rustbpe coupling: that
# module imports nanochat.tokenizer -> `import rustbpe` at module load, which is
# unavailable off-pod. The tokenizer is an injected dependency here, so we never
# need get_tokenizer / build_model.
# --------------------------------------------------------------------------- #
def _load_tensorfile(path, map_location):
    import io
    import gzip
    if os.path.exists(path):
        return torch.load(path, map_location=map_location)
    gz_path = path + ".gz"
    if os.path.exists(gz_path):
        with open(gz_path, "rb") as f:
            buf = io.BytesIO(gzip.decompress(f.read()))
        return torch.load(buf, map_location=map_location)
    raise FileNotFoundError(f"Checkpoint not found: {path} (or {gz_path})")


def _patch_missing_config_keys(cfg_kwargs):
    cfg_kwargs.setdefault("window_pattern", "L")  # old models: full context


def _patch_missing_keys(model_data, model_config):
    n_layer = model_config.n_layer
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)


def _download(hf_repo, arm, step):
    """(model_data path base, meta dict) for an arm. Handles gz/plain .pt."""
    from huggingface_hub import hf_hub_download
    stem = f"{arm}/model_{step:06d}.pt"
    try:
        gz = hf_hub_download(hf_repo, stem + ".gz", repo_type="model")
        model_path = gz[:-3]                      # _load_tensorfile re-adds .gz
    except Exception:
        model_path = hf_hub_download(hf_repo, stem, repo_type="model")
    meta_path = hf_hub_download(hf_repo, f"{arm}/meta_{step:06d}.json", repo_type="model")
    with open(meta_path) as f:
        meta = json.load(f)
    return model_path, meta


def load_model(arm, device, hf_repo, step):
    """Download + build an arm's checkpoint with injection sites rebuilt from the
    saved meta. Returns (model, meta). The baseline arm has no
    injection_sites_config, so it loads with NO site (vanilla). Injected arms
    rebuild their site (including the calibrated channel_scale the trainer froze
    into the meta); sites stay dormant (eval is activations-off unless forward
    gets acts=...).

    Mirrors checkpoint_manager.build_model's model-building steps EXCEPT:
      * no get_tokenizer() (the tokenizer is an injected dependency; rustbpe is
        off-pod-unavailable);
      * any ``direction_init: "file:..."`` in the meta (the sphere arm) is
        rewritten to "zeros" before setup_injection_sites, because the site's
        direction is IMMEDIATELY overwritten by the strict assign-load of the
        checkpoint's saved direction. This makes loading self-contained (no
        dependency on a family's *.npz nor a specific CWD).
    """
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.injection.sites import reassert_optimizability

    device = torch.device(device) if not isinstance(device, torch.device) else device
    model_path, meta = _download(hf_repo, arm, step)
    model_data = _load_tensorfile(model_path, map_location=device)
    if device.type in {"cpu", "mps"}:
        model_data = {k: (v.float() if v.dtype == torch.bfloat16 else v)
                      for k, v in model_data.items()}
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    cfg_kwargs = dict(meta["model_config"])
    _patch_missing_config_keys(cfg_kwargs)
    config = GPTConfig(**cfg_kwargs)
    _patch_missing_keys(model_data, config)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()  # (re)inits rotary caches; params overwritten by the load

    isc = meta.get("injection_sites_config")
    if isc:
        build_cfgs = [dict(c) for c in isc]
        for c in build_cfgs:  # file:-init overwritten by the assign-load anyway
            if isinstance(c.get("direction_init"), str) and c["direction_init"].startswith("file:"):
                c["direction_init"] = "zeros"
        model.setup_injection_sites(build_cfgs)
    model.load_state_dict(model_data, strict=True, assign=True)
    if isc:
        reassert_optimizability(model.injection_sites)  # assign-load dropped the stamps
    model.eval()
    return model, meta


def site_params(model, name):
    """A loaded arm's site, as attach_site kwargs: direction, the CALIBRATED
    channel_scale, threshold and after_block. There is no hand-set loudness constant
    to reuse any more — the trainer calibrated channel_scale at startup and froze it
    into the checkpoint, so the faithful negative control copies it from the arm."""
    site = model.injection_sites[name]
    return {"direction": site.direction.detach().float().cpu().numpy(),
            "channel_scale": site.channel_scale.detach().float().cpu().numpy(),
            "threshold": site.threshold.detach().float().cpu().numpy(),
            "after_block": int(site.cfg.after_block)}


def attach_site(model, direction, channel_scale, threshold=DEFAULT_THRESHOLD,
                after_block=0, name="acts", dial=1.0):
    """Bolt an inference-time injection site onto the BASELINE model (the causal
    negative control: a site the baseline never trained with). ``direction`` is an
    [r, n_embd] float array; ``channel_scale``/``threshold``/``after_block`` come from
    the donor arm's own site (site_params), so the control is as loud as the arm it
    controls for. ``dial`` multiplies that calibrated loudness (1.0 = match the arm).

    The site is built with a placeholder direction then the requested direction is
    copied in verbatim. Intended for a model with no existing ``name`` site; asserts so.
    """
    from nanochat.injection.sites import InjectionCfg

    direction = np.ascontiguousarray(direction, dtype=np.float32)
    r, n_embd = direction.shape
    assert n_embd == model.config.n_embd, \
        f"attach_site: direction n_embd {n_embd} != model {model.config.n_embd}"
    scale = np.broadcast_to(np.asarray(channel_scale, np.float32), (r,)) * float(dial)

    existing = getattr(model, "injection_sites", None)
    assert existing is None or name not in existing, \
        f"attach_site: model already has a {name!r} site (attach_site is for the baseline control)"

    cfg = InjectionCfg(name=name, r=r, after_block=int(after_block),
                       channel_scale=scale.astype(np.float32).tolist(),
                       threshold=np.asarray(threshold, np.float32).tolist(),
                       trainable_direction=False, direction_init="zeros")
    sites = model.setup_injection_sites([cfg])
    with torch.no_grad():
        site = sites[name]
        site.direction.copy_(torch.from_numpy(direction).to(site.direction.device,
                                                             site.direction.dtype))
    return sites[name]


# --------------------------------------------------------------------------- #
# 2. loudness_scale context manager (the on/off knob) + acts dict wrapping
# INVARIANT: on-vs-off is ALWAYS loudness_scale 1.0 vs 0.0 through the SAME acts code
# path — never acts=None vs acts=... . At scale 0 every channel_scale is 0, so the site
# adds rms(x)*(0 @ D_hat) == exactly 0: a bit-identical no-op. Scales are applied in
# place and restored; sites are never optimized during eval.
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def scaled_loudness(model, scale):
    """Temporarily multiply every injection site's channel_scale by ``scale`` IN PLACE,
    restoring on exit. scale=1.0 is a no-op; scale=0.0 makes every site an EXACT forward
    no-op (framework invariant). The threshold is untouched — this is a volume knob, not
    a gating change. Mutations run under no_grad (sites are never optimized during eval:
    channel_scale carries _never_optimize and sits in no optimizer group). Safe to nest
    inside torch.inference_mode forwards (the mutation happens on entry, before the
    inference-mode region)."""
    sites = getattr(model, "injection_sites", None)
    if sites is None or scale == 1.0:
        yield
        return
    saved = {}
    with torch.no_grad():
        for nm, site in sites.items():
            saved[nm] = site.channel_scale.detach().clone()
            site.channel_scale.mul_(float(scale))
    try:
        yield
    finally:
        with torch.no_grad():
            for nm, site in sites.items():
                site.channel_scale.copy_(saved[nm])


def _acts_dict(model, acts_tensor):
    """Map a single [B,T,r] acts tensor onto the model's injection sites (every
    geometry arm has exactly one site). Every site whose r matches gets the tensor;
    asserts at least one match."""
    sites = getattr(model, "injection_sites", None)
    if sites is None:
        return None
    r = acts_tensor.shape[-1]
    out = {nm: acts_tensor for nm, s in sites.items() if s.cfg.r == r}
    assert out, f"no injection site with r={r} to receive acts (sites: " \
                f"{[(nm, s.cfg.r) for nm, s in sites.items()]})"
    return out


# --------------------------------------------------------------------------- #
# 3. GemmaScorer — gemma-2-2b residuals -> the family's frozen ridge probes -> z
# --------------------------------------------------------------------------- #
def _find_attribution_out():
    """Locate the probe-constant dir (probe_set_arrays.npz + probe_set.json).
    Precedence: $ORACLE_ATTR_OUT, the superproject attribution/out, then the
    VENDORED copy in runs/lib/eval/attr_out (committed with the suite:
    probe_set_arrays.npz is gitignored in the superproject and lives on NO HF
    repo, so a fresh pod clone has no other source)."""
    env = os.environ.get("ORACLE_ATTR_OUT")
    cands = ([env] if env else []) + [
        os.path.join(_REPO_ROOT, "..", "attribution", "out"),   # superproject sibling of nanochat
        os.path.join(os.getcwd(), "..", "attribution", "out"),
        os.path.join(os.getcwd(), "attribution", "out"),
        os.path.join(_HERE, "attr_out"),                        # vendored (self-contained)
    ]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "probe_set_arrays.npz")):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "attribution/out not found (need probe_set_arrays.npz + probe_set.json). "
        "Set $ORACLE_ATTR_OUT, or use the vendored runs/lib/eval/attr_out.")


class GemmaScorer:
    """Score fresh texts with the family's frozen probe pipeline, reproducing the
    z-scores training injected (up to the store's int8 quantization). Pipeline
    (frozen; from attribution/measure_loudness.py):
        step-1: x_std = (x - nat_mean_L) / nat_std_L
        probe : raw   = x_std @ W_c^T + b_c
        step-2: z     = (raw - mu2_c) / std2_c
    x = gemma-2-2b raw residual at layer L for a non-BOS token (eager attention
    MANDATORY — sdpa drops gemma-2 softcapping; BOS prepended then its row
    dropped; fp32 norms). ``concepts`` is the family's STORE order; the store
    columns.json order and probe_set main_block order are asserted equal below, so
    W/b/nat_* index the same positions as mu2/std2.

    GPU-only: constructs gemma-2-2b. Never built in the CPU test suite.
    """

    def __init__(self, device, concepts, *, store=DEFAULT_SCORE_STORE, attr_out=None,
                 layer=GEMMA_LAYER, gemma_model=GEMMA_MODEL, dtype=None):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.layer = int(layer)
        self.concepts = list(concepts)
        self.gemma_model = gemma_model
        attr_out = attr_out or _find_attribution_out()

        # Probe geometry (step-1 + ridge), main_block order (probe_set_arrays.npz).
        ps = json.load(open(os.path.join(attr_out, "probe_set.json")))
        main_block = list(ps["main_block_concepts"])
        layers = list(ps["layers"])
        li_p = layers.index(self.layer)
        cidx_p = np.array([main_block.index(c) for c in self.concepts], np.int64)
        arr = np.load(os.path.join(attr_out, "probe_set_arrays.npz"))
        self._W = np.asarray(arr["W"], np.float64)[li_p][cidx_p]           # (r, D)
        self._b = np.asarray(arr["b"], np.float64)[li_p][cidx_p]           # (r,)
        self._nat_mean = np.asarray(arr["nat_mean"], np.float64)[li_p]     # (D,)
        self._nat_std = np.asarray(arr["nat_std"], np.float64)[li_p]       # (D,)

        # step-2 corpus standardization (store corpus_stats.json), store column
        # order (climbmix-scored) — the SAME constants training standardized with.
        from nanochat.injection.sources import _read_store_json
        columns = _read_store_json(store, "columns.json")
        corpus_stats = _read_store_json(store, "corpus_stats.json")
        col_names = list(columns["concepts"])
        li_s = list(columns["layers"]).index(self.layer)
        cidx_s = np.array([col_names.index(c) for c in self.concepts], np.int64)
        self._mu2 = np.asarray(corpus_stats["mean"], np.float64)[li_s][cidx_s]  # (r,)
        self._std2 = np.asarray(corpus_stats["std"], np.float64)[li_s][cidx_s]  # (r,)
        assert main_block == col_names, \
            "probe main_block order != store columns.json order (permutation!) — refusing"

        self._tok = None
        self._model = None
        self._dtype = dtype

    # -- lazy gemma load (eager attention MANDATORY) --
    def _ensure_model(self):
        if self._model is not None:
            return
        import transformers.utils.import_utils as _iu  # torchvision ABI guard (see measure_loudness)
        _iu.is_torchvision_available = lambda: False
        if hasattr(_iu, "_torchvision_available"):
            _iu._torchvision_available = False
        from transformers import AutoModel, AutoTokenizer
        dev = self.device.type
        dtype = self._dtype or (torch.float16 if dev == "mps"
                                else torch.bfloat16 if dev == "cuda" else torch.float32)
        self._tok = AutoTokenizer.from_pretrained(self.gemma_model)
        # eager is MANDATORY: sdpa silently drops gemma-2's attention logit softcapping.
        m = AutoModel.from_pretrained(self.gemma_model, dtype=dtype, attn_implementation="eager")
        self._model = m.eval().to(self.device)

    def gemma_encode(self, text):
        """(ids, char-offsets) for ONE text via the gemma FAST tokenizer only
        (add_special_tokens=False, BOS-free). Same convention as
        RuntimeProbeScoreSource.gemma_encode; the ids/offsets match score()'s
        tokenization so build_acts can align without re-running gemma."""
        self._ensure_model()
        from nanochat.injection.align import get_offsets
        return get_offsets(self._tok, text, add_special_tokens=False)

    @property
    def tokenizer(self):
        """The gemma fast tokenizer (loaded on demand) — a further offsets fallback."""
        self._ensure_model()
        return self._tok

    def gemma_offsets(self, texts):
        """[ (n_gemma, 2) int char offsets, ... ] via the gemma FAST tokenizer only
        (add_special_tokens=False, BOS-free — the tokenization the scores index).
        Cheap (no model); reuses align.get_offsets. Matches the ids that score()
        produces so build_acts can align without re-running gemma."""
        self._ensure_model()
        from nanochat.injection.align import get_offsets
        out = []
        for t in texts:
            _, off = get_offsets(self._tok, t, add_special_tokens=False)
            out.append(np.asarray(off, np.int64).reshape(-1, 2))
        return out

    @torch.inference_mode()
    def _residuals(self, ids):
        """Raw residual hidden states at self.layer for a BOS-free id list -> (n, D)
        fp32. BOS prepended, its row dropped; windowed at 2048 like the scorer."""
        WIN = 2048
        chunks = [ids[i:i + WIN] for i in range(0, len(ids), WIN)] or [[]]
        bos = self._tok.bos_token_id
        parts = []
        for w in chunks:
            inp = torch.tensor([[bos] + list(w)], dtype=torch.long, device=self.device)
            out = self._model(input_ids=inp, output_hidden_states=True)
            h = out.hidden_states[self.layer + 1][0, 1:1 + len(w), :].float().cpu().numpy()
            parts.append(h)
        return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]

    def score(self, texts):
        """list[str] -> list[np.ndarray [n_gemma, r] float32 z]. One row per gemma
        token (BOS-free), standardized concept z-scores. Pair with gemma_offsets()
        (same tokenization) to feed build_acts."""
        self._ensure_model()
        from nanochat.injection.align import get_offsets
        out = []
        for t in texts:
            ids, _ = get_offsets(self._tok, t, add_special_tokens=False)
            if len(ids) == 0:
                out.append(np.zeros((0, len(self.concepts)), np.float32))
                continue
            x = self._residuals(ids).astype(np.float64)              # (n, D)
            xstd = (x - self._nat_mean) / self._nat_std
            raw = xstd @ self._W.T + self._b                         # (n, r)
            z = (raw - self._mu2) / self._std2
            out.append(z.astype(np.float32))
        return out

    def score_with_offsets(self, texts):
        """Convenience: [ (z [n,7], offsets [n,2]), ... ] in one gemma pass worth
        of tokenization. z and offsets share the exact BOS-free tokenization."""
        self._ensure_model()
        from nanochat.injection.align import get_offsets
        out = []
        for t in texts:
            ids, off = get_offsets(self._tok, t, add_special_tokens=False)
            if len(ids) == 0:
                out.append((np.zeros((0, len(self.concepts)), np.float32),
                            np.zeros((0, 2), np.int64)))
                continue
            x = self._residuals(ids).astype(np.float64)
            xstd = (x - self._nat_mean) / self._nat_std
            z = ((xstd @ self._W.T + self._b) - self._mu2) / self._std2
            out.append((z.astype(np.float32), np.asarray(off, np.int64).reshape(-1, 2)))
        return out


# --------------------------------------------------------------------------- #
# 4. build_acts — align gemma z onto nanochat tokens, exactly like training
# --------------------------------------------------------------------------- #
def build_acts(text, nano_ids, gemma_z, offsets, threshold=DEFAULT_THRESHOLD,
               policy="mean", nano_enc=None):
    """Align standardized gemma z onto the nanochat token grid EXACTLY like training:
    char-span OVERLAP pooling (default 'mean' over covering gemma tokens; 'last' =
    rightmost), then the realism threshold (a nanochat-token row is EXACT zero unless
    its max channel >= threshold). Returns (T, r) float32, T = len(nano_ids).

    Reuses the shared machinery verbatim — it calls
    ConceptProbeScoreSource._align_and_gather (overlap pool + threshold, the same code
    training ran) on a store-free shim instance, feeding it the precomputed gemma
    tokenization via ``offsets`` (so no gemma model/tokenizer re-runs here).
    ``gemma_z`` is already standardized (GemmaScorer output); alignment does not
    re-standardize.

    nano_enc (tiktoken Encoding: encode_ordinary + decode_single_token_bytes)
    defaults to the module tokenizer set via set_nano_tokenizer.
    """
    if ConceptProbeScoreSource is None:  # pragma: no cover
        raise RuntimeError("runs/lib/probe_source.py unavailable; cannot reuse the align machinery")
    enc = _resolve_nano_enc(nano_enc)

    gemma_z = np.ascontiguousarray(gemma_z, dtype=np.float32)
    offsets = np.asarray(offsets, np.int64).reshape(-1, 2)
    n_gemma = gemma_z.shape[0]
    assert offsets.shape[0] == n_gemma, \
        f"offsets ({offsets.shape[0]}) must match gemma_z rows ({n_gemma})"

    # Store-free shim: an object of the real class with only the attributes
    # _align_and_gather (parent overlap) + its threshold override touch. Its
    # gemma_encode returns the PRECOMPUTED (ids, offsets) so no gemma runs.
    src = object.__new__(ConceptProbeScoreSource)
    src.nano_enc = enc
    src.gemma_encode = lambda _t: (list(range(n_gemma)), offsets)
    src.align_policy = policy
    src.r = int(gemma_z.shape[1])
    src.present_z = float(threshold)

    out = src._align_and_gather(text, len(nano_ids), gemma_z)
    if out is None:  # nano retokenization drift (shouldn't happen for matching ids)
        return np.zeros((len(nano_ids), src.r), np.float32)
    return out


# --------------------------------------------------------------------------- #
# 5. forward_metrics — per-token CE + logits, on/off via loudness_scale
# --------------------------------------------------------------------------- #
def forward_metrics(model, ids, acts=None, loudness_scale=1.0, transform=None,
                    return_logits=True):
    """Forward ``ids`` [B,T] once and return per-token CE + logits. On-vs-off is
    loudness_scale=1.0 vs 0.0 through the SAME acts path (0 = exact no-op).

    Args:
      ids: LongTensor [B,T] (input tokens; CE targets are next-token, ids[:,1:]).
      acts: [B,T,r] float array/tensor of injected activations aligned to ``ids``
            input positions (row t = input token t; BOS row is a zero row).
            None => vanilla forward (no site fires).
      loudness_scale: multiplies every site's channel_scale in place for this
            forward only.
      transform: optional acts -> acts hook (applied to the [B,T,r] tensor before
            injection; used by causal tests, e.g. permute/scale directions).
      return_logits: False skips the [B,T,V] fp32 device->cpu logits copy
            (~268 MB at T=2048, V=32768) — per-token CE is unaffected. Use for
            CE-only consumers (val-bpb); the CE math is identical either way
            (asserted in test_consolidation.py).

    Returns dict:
      per_token_ce: np.ndarray [B,T] fp32 — CE of predicting token t
                    (-log p(ids[t] | ids[:t])); position 0 is NaN (no context).
      logits:       torch.Tensor [B,T,V] fp32 (cpu) — inspect any position.
                    None when return_logits=False.
      ce_mean:      float, mean CE over valid (non-NaN) positions.
      acts:         the [B,T,r] np array actually injected (post-transform) or None.
      loudness_scale: echoed.
    """
    device = model.get_device()
    ids = ids.to(device) if isinstance(ids, torch.Tensor) else torch.as_tensor(ids, device=device)
    if ids.ndim == 1:
        ids = ids[None, :]
    B, T = ids.shape

    acts_tensor = None
    acts_np = None
    has_sites = getattr(model, "injection_sites", None) is not None
    if acts is not None and has_sites:
        acts_tensor = (acts if isinstance(acts, torch.Tensor)
                       else torch.as_tensor(np.asarray(acts, np.float32)))
        acts_tensor = acts_tensor.to(device=device, dtype=torch.float32)
        if transform is not None:
            acts_tensor = transform(acts_tensor)
        acts_np = acts_tensor.detach().cpu().numpy()
        acts_dict = _acts_dict(model, acts_tensor)
    else:
        acts_dict = None

    with scaled_loudness(model, loudness_scale):
        with torch.inference_mode():
            logits = model(ids, acts=acts_dict)            # [B,T,V] fp32 (forward casts)
    logits = logits.float()

    # per-token CE, fp32 accumulation. logits[:, t] predicts token t+1.
    V = logits.shape[-1]
    lg = logits[:, :-1, :].reshape(-1, V)
    tgt = ids[:, 1:].reshape(-1)
    ce = F.cross_entropy(lg, tgt, reduction="none").reshape(B, T - 1)  # CE for targets 1..T-1
    per_token_ce = np.full((B, T), np.nan, np.float32)
    per_token_ce[:, 1:] = ce.detach().cpu().numpy()

    valid = ~np.isnan(per_token_ce)
    return {
        "per_token_ce": per_token_ce,
        "logits": logits.detach().cpu() if return_logits else None,
        "ce_mean": float(per_token_ce[valid].mean()) if valid.any() else float("nan"),
        "acts": acts_np,
        "loudness_scale": float(loudness_scale),
    }


# --------------------------------------------------------------------------- #
# 6. ce_report — dilution-aware CE decomposition
# --------------------------------------------------------------------------- #
def ce_report(per_token_ce, acts):
    """Decompose per-token CE by injection status. per_token_ce [B,T] indexes the
    PREDICTED token t; acts [B,T,r] row t marks whether input token t was injected
    (nonzero). Buckets partition the valid (non-NaN) positions:
      * injected        — predicting an injected token (acts row t nonzero)
      * after_injected  — predicting the token right after an injected one, and
                          not itself injected
      * other           — everything else
    Injected tokens are ~2-3% of the corpus, so the conditional means are the
    signal (the overall mean is dominated by 'other')."""
    ptc = np.asarray(per_token_ce, np.float64)
    a = np.asarray(acts)
    if ptc.ndim == 1:                                      # tolerate a single [T] sequence
        ptc = ptc[None]
    if a.ndim == 2:                                        # tolerate a single [T,r] sequence
        a = a[None]
    inj_raw = np.any(a != 0.0, axis=-1)                    # [B,T] any concept channel fired
    valid = ~np.isnan(ptc)
    after_raw = np.zeros_like(inj_raw)
    after_raw[:, 1:] = inj_raw[:, :-1]

    injected = valid & inj_raw
    after = valid & after_raw & ~inj_raw
    other = valid & ~inj_raw & ~after_raw

    def _mean(mask):
        return float(ptc[mask].mean()) if mask.any() else float("nan")

    return {
        "ce_overall": _mean(valid), "n_overall": int(valid.sum()),
        "ce_injected": _mean(injected), "n_injected": int(injected.sum()),
        "ce_after_injected": _mean(after), "n_after_injected": int(after.sum()),
        "ce_other": _mean(other), "n_other": int(other.sum()),
    }
