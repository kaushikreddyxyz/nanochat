"""Activation sources: where the per-token content of an injection site comes
from. Defines the ActivationSource interface, the on-disk activation-store
format ("activation-store-v2", written by scripts/precompute_activations.py),
its two readers (QwenEncoderSource, ProbeScoreSource), the arbitrary-callable
FnSource, and the structured-activation construction (ring phases / PCA) used
by the Qwen-encoder precompute. Format + rationale: nanochat/injection/README.md.
"""
import abc
import hashlib
import json
import math
import os

import numpy as np
import torch

STORE_FORMAT = "activation-store-v2"


class ActivationSource(abc.ABC):
    """Per-document activation provider for the ride-along dataloader.

    Attributes: ``name`` (site name the source feeds), ``r`` (channels).

    Contract:
      * ``lookup(doc_text, n_tokens) -> (z, key)`` where ``z`` is a float32
        ``(n_tokens, r)`` array or ``None``, and ``key`` is a per-doc int used
        to seed noise. ``None`` means "unknown doc" (missing from a store, or
        stored token count != n_tokens, i.e. tokenizer drift). Callers MUST map
        None to EXACT zeros with NO noise: the injection site renormalizes any
        nonzero row to full gate amplitude, so noised zeros would inject pure
        noise at full strength on exactly the docs we know nothing about.
        Exact zeros keep the injection a strict no-op there.
      * ``add_noise(z, key)`` returns z plus any train-time noise; must be
        deterministic in (source config, key) and never applied to the
        None-fallback zeros.
      * Both run loader-side without grad; the site detaches again, so
        activations are unoptimizable by construction.
    """

    name: str
    r: int

    @abc.abstractmethod
    def lookup(self, doc_text: str, n_tokens: int):
        ...

    @abc.abstractmethod
    def add_noise(self, z: np.ndarray, key: int) -> np.ndarray:
        ...

    def sample_activation_rows(self, k: int, seed: int):
        """Pooled ((n, r) rows, n_docs, n_tokens) over a seeded sample of up to ``k``
        docs — the raw rows dose calibration needs. Sources that cannot be sampled
        (e.g. FnSource) leave this unimplemented.

        Implementations MUST also set ``last_sample_doc_index``: an (n,) int array
        giving each pooled row's SOURCE DOCUMENT ordinal. calibrate_dose_gate uses it
        to count distinct documents per channel — a bursty concept can rack up
        thousands of firing tokens inside 3 documents, and a per-token count calls
        that well-measured when its scale rests on 3 samples. Sources that sample
        across shards should also set ``last_sample_shards``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support gate='auto' / amplitude='dose' "
            f"(no sampleable statistics)")

    last_sample_doc_index = None   # (n,) source-doc ordinal per pooled row; see above
    last_sample_shards = None      # shard ids the sample was drawn from, when sharded

    def sample_activation_stats(self, k: int, seed: int):
        """(per-channel rms, per-channel nonzero-rate, n_docs, n_tokens) over the SAME
        seeded sample as sample_activation_rows — the statistics gate="auto" needs."""
        rows, n_docs, n_tokens = self.sample_activation_rows(k, seed)
        if n_tokens == 0:
            return np.zeros(self.r, np.float32), np.zeros(self.r, np.float32), 0, 0
        rms, nz = _channel_stats(rows)
        return rms, nz, n_docs, n_tokens


def doc_hash(text: str) -> np.uint64:
    """Stable 64-bit content hash of a document string (order-independent key)."""
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return np.frombuffer(h, dtype="<u8")[0]


def hash_seeded_noise(z: np.ndarray, key: int, sigma: float, seed: int) -> np.ndarray:
    """Gaussian noise seeded by (train seed, doc content hash): DDP-rank- and
    resume-independent, not memorizable as a per-position identity. Never applied
    to the None-fallback zeros (the loader adds noise only to nonzero rows)."""
    if sigma <= 0:
        return z
    rng = np.random.default_rng([seed & 0x7FFFFFFF, key & 0xFFFFFFFFFFFFFFFF])
    return z + rng.normal(0.0, sigma, size=z.shape).astype(np.float32)


def _pool_doc_rows(per_doc, r):
    """[(n_i, r) per-doc arrays] -> (pooled (n, r), doc_index (n,), n_docs, n_tokens).
    doc_index labels every pooled row with the ordinal of the document it came from,
    which is what lets calibration count DOCUMENTS rather than tokens per channel."""
    if not per_doc:
        return np.zeros((0, r), np.float32), np.zeros(0, np.int64), 0, 0
    pooled = np.concatenate(per_doc, axis=0)
    doc_index = np.repeat(np.arange(len(per_doc), dtype=np.int64),
                          [z.shape[0] for z in per_doc])
    return pooled, doc_index, len(per_doc), pooled.shape[0]


def _channel_stats(rows: np.ndarray):
    """Per-channel (rms, nonzero-rate) over pooled (n, r) activation rows."""
    rms = np.sqrt((rows.astype(np.float64) ** 2).mean(0))
    nz = (rows != 0.0).mean(0)
    return rms.astype(np.float32), nz.astype(np.float32)


def make_orthonormal_P(n_embd: int, r: int, seed: int = 1337) -> np.ndarray:
    """Seeded orthonormal (n_embd, r) projection; sites.orthonormal_direction
    is its transpose (same generator, fp64 QR)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n_embd, r, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(a, mode="reduced")
    return q.to(torch.float32).numpy()


# --------------------------------------------------------------------------- #
# Store-backed sources.
#
# Store layout (one directory):
#   activations.int8   memmap int8 [n_doc_tokens, r]  (standardized, quantized)
#   index.npy          structured [n_docs] (hash uint64, off int64, n int32)
#   meta.json          {"format": "activation-store-v2", "source_kind": ..., r, scale, ...}
#   P.npy              optional float32 [n_embd, r] fixed orthonormal projection
#
# Activations are stored PER DOCUMENT keyed by content hash (not per corpus
# position): the training dataloader packs+crops docs in a data-dependent
# order, and hash keying keeps the store order/DDP-independent.
# --------------------------------------------------------------------------- #
class _StoreSource(ActivationSource):
    source_kind = None  # subclasses pin the expected meta["source_kind"]

    def __init__(self, store_dir, noise_sigma=0.15, seed=0, name=None):
        with open(os.path.join(store_dir, "meta.json")) as f:
            meta = json.load(f)
        fmt = meta.get("format")
        if fmt != STORE_FORMAT:
            raise ValueError(f"{store_dir}: meta.json format={fmt!r}, expected {STORE_FORMAT!r} "
                             f"(no pre-v2 stores exist; rebuild with scripts/precompute_activations.py)")
        kind = meta.get("source_kind")
        if self.source_kind is not None and kind != self.source_kind:
            raise ValueError(f"{store_dir}: source_kind={kind!r}, expected {self.source_kind!r} "
                             f"for {type(self).__name__}")
        self.meta = meta
        self.store_dir = store_dir
        self.name = name or kind
        self.r = int(meta["r"])
        self.scale = float(meta["scale"])       # int8 -> float dequant scale
        self.noise_sigma = float(noise_sigma)
        self.seed = int(seed)
        self.mm = np.memmap(os.path.join(store_dir, "activations.int8"),
                            dtype=np.int8, mode="r").reshape(-1, self.r)
        idx = np.load(os.path.join(store_dir, "index.npy"))
        self._index = {int(h): (int(o), int(n))
                       for h, o, n in zip(idx["hash"], idx["off"], idx["n"])}

    def __len__(self):
        return len(self._index)

    def lookup(self, text: str, n_tokens: int):
        """(n_tokens, r) float32 activations + hash, or (None, hash) if the doc
        is missing or its stored token count mismatches (see the ActivationSource
        contract: callers map None to exact zeros, no noise)."""
        h = int(doc_hash(text))
        rec = self._index.get(h)
        if rec is None:
            return None, h
        off, n = rec
        if n != n_tokens:  # tokenizer drift -> fail safe to zero
            return None, h
        return self.mm[off:off + n].astype(np.float32) * self.scale, h

    def add_noise(self, z: np.ndarray, key: int) -> np.ndarray:
        return hash_seeded_noise(z, key, self.noise_sigma, self.seed)

    def sample_activation_rows(self, k: int, seed: int):
        """Pool the (dequantized) rows of up to ``k`` seeded index entries."""
        keys = list(self._index)
        rng = np.random.default_rng(seed & 0x7FFFFFFF)
        pick = rng.permutation(len(keys))[:min(k, len(keys))]
        rows = []
        for i in pick:
            off, n = self._index[keys[int(i)]]
            if n > 0:
                rows.append(self.mm[off:off + n].astype(np.float32) * self.scale)
        pooled, doc_index, n_docs, n_tokens = _pool_doc_rows(rows, self.r)
        self.last_sample_doc_index = doc_index
        return pooled, n_docs, n_tokens


class QwenEncoderSource(_StoreSource):
    """Precomputed predictions of the frozen Qwen oracle-encoder (structured
    ring/PCA activations, r=14). Store built by scripts/precompute_activations.py
    modes fit/sweep/assemble."""
    source_kind = "qwen-encoder"


class ProbeScoreSource(_StoreSource):
    """Reader for a source_kind="probe-scores" v2 store (one layer's 54
    standardized scores per nanochat token). Kept for any pre-built probe-scores
    store; the live/runtime path (RuntimeProbeScoreSource) is what training uses
    now — it applies gold gemma scores at runtime with nothing stored offline."""
    source_kind = "probe-scores"


_KIND_TO_SOURCE = {cls.source_kind: cls for cls in (QwenEncoderSource, ProbeScoreSource)}


def open_store(store_dir, noise_sigma=0.15, seed=0, name=None, expect_kind=None):
    """Open an activation store, dispatching on meta.json source_kind."""
    with open(os.path.join(store_dir, "meta.json")) as f:
        kind = json.load(f).get("source_kind")
    if expect_kind is not None and kind != expect_kind:
        raise ValueError(f"{store_dir}: source_kind={kind!r}, expected {expect_kind!r}")
    cls = _KIND_TO_SOURCE.get(kind)
    if cls is None:
        raise ValueError(f"{store_dir}: unknown source_kind {kind!r} "
                         f"(known: {sorted(_KIND_TO_SOURCE)})")
    return cls(store_dir, noise_sigma=noise_sigma, seed=seed, name=name)


def load_source_class(ref):
    """Resolve an experiment-side source class named in an --activation-config
    source spec ("class": "..."). Two forms:

      * FILE PATH  — "path/to/file.py:ClassName". Relative paths resolve from
        the launch CWD (run from the nanochat repo root). Collision-proof: no
        importable module name is involved, so an installed PyPI package can
        never shadow the experiment file. PREFER THIS FORM.
      * DOTTED MODULE — "pkg.mod:ClassName". Requires the module to be
        importable; fragile when a same-named installed package shadows a local
        namespace dir (e.g. the PyPI package "runs" vs a local runs/ dir).
    """
    import importlib
    import importlib.util
    target, _, cls_name = ref.rpartition(":")
    assert target and cls_name, \
        f"source 'class' must be 'path/to/file.py:Class' or 'pkg.mod:Class', got {ref!r}"
    if target.endswith(".py") or "/" in target or os.sep in target:
        path = target if os.path.isabs(target) else os.path.join(os.getcwd(), target)
        spec = importlib.util.spec_from_file_location(f"_inj_src_{cls_name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)
    return getattr(mod, cls_name)


class FnSource(ActivationSource):
    """Arbitrary-callable source: fn(text, n_tokens) -> (n_tokens, r) float32.
    For synthetic/control injections (positional ramps, random features, ...);
    a future runtime gold-probe source plugs in the same way. Noise-free."""

    def __init__(self, fn, r: int, name: str = "fn"):
        self.fn, self.r, self.name = fn, int(r), name

    def lookup(self, text: str, n_tokens: int):
        out = self.fn(text, n_tokens)
        assert out.shape == (n_tokens, self.r)
        return out, 0

    def add_noise(self, z, key):
        return z


# --------------------------------------------------------------------------- #
# Runtime probe-score sources: apply the gold gemma probe scores per nanochat
# token AT RUNTIME, nothing stored offline. Both backends share the same
# gemma->nanochat alignment (char-span OVERLAP; default per-channel MAX over
# covering gemma tokens, "mean"/"last" opt-in. A boundary-straddling gemma token
# leaks sub-token future-char content into position t; compact-tokens mode is
# the leak-free variant)
# and the same dequant/standardize with the frozen quant/corpus_stats constants;
# they differ only in where the per-doc gemma scores come from (a pre-scored HF
# store, or a live gemma scorer). Columns default to one layer's 54 concepts in
# columns.json order; a concept subset narrows r. Format + cost: README.
# --------------------------------------------------------------------------- #
def _read_store_json(loc, name):
    """Read columns/quant/corpus_stats json from a local dir or an HF dataset repo."""
    if os.path.isdir(loc):
        with open(os.path.join(loc, name)) as f:
            return json.load(f)
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(loc, name, repo_type="dataset")) as f:
        return json.load(f)


class _RuntimeProbeBase(ActivationSource):
    """Shared alignment + standardization for the runtime probe-score backends.
    Subclasses supply per-doc gemma scores via ``_gemma_z``; this base aligns them
    onto the nanochat token grid and enforces the unknown-doc/None contract."""

    def _init_layout(self, columns, quant, corpus_stats, layer, concepts, name,
                     noise_sigma, seed, align_policy="max"):
        col_names = list(columns["concepts"])           # score axis-2 order == columns.json
        layers = list(columns["layers"])
        if layer not in layers:
            raise ValueError(f"layer {layer} not in store layers {layers}")
        self.li = layers.index(layer)
        self.layer = int(layer)
        sel = list(concepts) if concepts else col_names  # default: all 54, columns.json order
        self.col_idx = np.array([col_names.index(c) for c in sel], np.int64)
        self.r = int(self.col_idx.size)
        z = np.asarray(quant["zero"], np.float32)[self.li][self.col_idx]     # (r,)
        s = np.asarray(quant["scale"], np.float32)[self.li][self.col_idx]
        m = np.asarray(corpus_stats["mean"], np.float32)[self.li][self.col_idx]
        d = np.asarray(corpus_stats["std"], np.float32)[self.li][self.col_idx]
        self._zero, self._scale, self._mean, self._std = z, s, m, d
        self.concepts = sel
        self.name = name
        self.noise_sigma = float(noise_sigma)
        self.seed = int(seed)
        if align_policy not in ("max", "mean", "last"):
            raise ValueError(f"align_policy must be 'max', 'mean' or 'last', got {align_policy!r}")
        self.align_policy = align_policy

    def _depth_counter(self):
        """The pooling-depth histogram's lock, allocating both on first use — the eval
        harness aligns with sources it builds without running _init_layout."""
        lock = getattr(self, "_depth_lock", None)
        if lock is None:
            import threading
            lock = self._depth_lock = threading.Lock()
            self._depth_hist = np.zeros(1, np.int64)   # index = gemma tokens pooled per nanochat token
        return lock

    def _record_pool_depth(self, depths):
        """Accumulate the per-nanochat-token pooling-depth histogram. Depth is the
        MECHANISM behind any realized-vs-calibrated loudness gap (max concentrates
        co-firing, mean dilutes peaks) and behind max's raised noise floor, which both
        scale with it — so the startup report prints it beside the loudness ladder."""
        if depths.size == 0:
            return
        h = np.bincount(np.maximum(depths, 0).astype(np.int64))
        with self._depth_counter():
            if h.size > self._depth_hist.size:
                self._depth_hist = np.pad(self._depth_hist, (0, h.size - self._depth_hist.size))
            self._depth_hist[:h.size] += h

    def pool_depth_stats(self):
        """{n_tokens, p50, p90, p99, max, mean, zero_coverage_rate, hist} over every
        nanochat token aligned so far, or None if nothing has been aligned yet."""
        with self._depth_counter():
            hist = self._depth_hist.copy()
        n = int(hist.sum())
        if n == 0:
            return None
        depths = np.arange(hist.size)
        cum = np.cumsum(hist)
        q = {f"p{p}": int(depths[np.searchsorted(cum, math.ceil(p / 100 * n))])
             for p in (50, 90, 99)}
        return {"n_tokens": n, "max": int(depths[hist > 0].max()),
                "mean": float((depths * hist).sum() / n),
                "zero_coverage_rate": float(hist[0] / n),
                "hist": {int(d): int(c) for d, c in enumerate(hist) if c}, **q}

    def _standardize(self, raw):
        """raw (n, r) probe scores -> z (n, r); dequant already applied upstream."""
        return (raw - self._mean) / self._std

    def _align_and_gather(self, text, n_tokens, z_gemma):
        """z_gemma (n_gemma, r) standardized -> (n_tokens, r) on the nanochat grid,
        or None (unknown/drift). Each nanochat token's covering set is the gemma
        tokens whose CHAR SPAN OVERLAPS its own — so a big gemma token broadcasts
        its score to every nanochat token nested inside it, and a big nanochat
        token pools every gemma token it spans. ``align_policy`` picks the pool op:
        'max' (default) takes the per-channel maximum over the covering z-scores —
        the site's relu asks whether the span CONTAINS the concept, so a peak must
        survive pooling; 'mean' averages them (dilutes peaks, and the site then runs
        quieter than calibration predicted — README); 'last' keeps only the last
        (rightmost) covering gemma token (the historical prefix behavior for the
        multi-gemma->one-nano direction). None of the three re-standardizes. A
        nanochat token with NO overlapping gemma token (a char the gemma tokenizer
        dropped) stays EXACT zero."""
        from nanochat.injection.align import nanochat_char_offsets
        nano_ids = self.nano_enc.encode_ordinary(text)
        if len(nano_ids) != n_tokens:            # drift vs the loader's body-token count
            return None
        nano_off = np.asarray(nanochat_char_offsets(self.nano_enc, nano_ids, text), np.int64)
        g_ids, g_off = self.gemma_encode(text)
        if len(g_ids) != z_gemma.shape[0]:       # gemma retokenization disagrees with the scores
            return None
        out = np.zeros((n_tokens, self.r), np.float32)
        g_off = np.asarray(g_off, np.int64).reshape(-1, 2)
        keep = np.where(g_off[:, 1] > g_off[:, 0])[0]   # non-empty gemma spans only cover chars
        if keep.size == 0:
            self._record_pool_depth(np.zeros(n_tokens, np.int64))
            return out
        gs, ge = g_off[keep, 0], g_off[keep, 1]         # char-monotonic (left-to-right tokenization)
        z_keep = z_gemma[keep]
        ns, ne = nano_off[:, 0], nano_off[:, 1]
        lo = np.searchsorted(ge, ns, side="right")      # first gemma whose end char > nano start
        hi = np.searchsorted(gs, ne, side="left") - 1   # last gemma whose start char < nano end
        has = (lo <= hi) & (ne > ns)                    # >=1 overlapping gemma and non-empty nano span
        self._record_pool_depth(np.where(has, hi - lo + 1, 0))
        if not has.any():
            return out
        if self.align_policy == "last":
            out[has] = z_keep[hi[has]]
        elif self.align_policy == "mean":               # mean over the overlapping gemma tokens
            csum = np.concatenate([np.zeros((1, self.r), np.float32),
                                   np.cumsum(z_keep, axis=0, dtype=np.float32)], axis=0)
            s, e = lo[has], hi[has] + 1
            out[has] = (csum[e] - csum[s]) / (e - s)[:, None]
        else:                                           # per-channel max over [lo, hi]
            # reduceat over interleaved starts/ends: even slots are the real ranges
            # (always non-empty, since has => lo <= hi), odd slots are the gaps between
            # them and are discarded. One pad row so an end of exactly n stays in range.
            s, e = lo[has], hi[has] + 1
            idx = np.empty(2 * s.size, np.int64)
            idx[0::2], idx[1::2] = s, e
            z_pad = np.concatenate([z_keep, np.zeros((1, self.r), np.float32)], axis=0)
            out[has] = np.maximum.reduceat(z_pad, idx, axis=0)[0::2]
        return out

    def add_noise(self, z, key):
        return hash_seeded_noise(z, key, self.noise_sigma, self.seed)


class RuntimeProbeScoreSource(_RuntimeProbeBase):
    """Stored-scores backend (primary). Joins each packed doc to its scored rows
    POSITIONALLY: the ride-along loader re-runs the real corpus enumeration, so it
    knows every doc's (shard, row); docs_<sid>.jsonl (row order, full coverage) maps
    row -> (offset, n_gemma) into scores_<sid>.npy. No hashing, no startup walk —
    docs_<sid>.jsonl and the score memmap load lazily per shard. Per doc it
    dequantizes+standardizes one layer's columns, retokenizes gemma for offsets, and
    prefix-aligns onto the nanochat tokens. Shard not configured / row out of range /
    tokenizer drift -> None (loader maps to exact zeros, no noise; counted in stats).

    Optional text-keyed fallback (``build_hash_index``/``index_path``, needs
    ``climbmix_dir``) walks docs_<sid>.jsonl + parquet text once to hash-key docs,
    for callers without the (shard, row) position — a real startup cost (README)."""

    positional = True  # loader joins by (shard, row), not by hashing doc text

    def __init__(self, score_loc, shards, layer=8, *, nano_enc, gemma_encode=None,
                 gemma_model="google/gemma-2-2b", concepts=None, noise_sigma=0.15,
                 seed=0, name="probe-scores-runtime", climbmix_dir=None,
                 build_hash_index=False, index_path=None, text_column="text",
                 align_policy="max", n_calib_shards=4):
        columns = _read_store_json(score_loc, "columns.json")
        quant = _read_store_json(score_loc, "quant.json")
        corpus_stats = _read_store_json(score_loc, "corpus_stats.json")
        self._init_layout(columns, quant, corpus_stats, layer, concepts, name, noise_sigma, seed,
                          align_policy)
        self.score_loc = score_loc
        self.shards = set(int(s) for s in shards)
        # Calibration reads whole shards (~GBs each) and runs BEFORE the prefetcher is
        # attached, so it must touch a BOUNDED, LOW set: unbounded striping across the
        # configured shards would both download the corpus and 404 on the first shard
        # living in an overflow repo. >=2 so the sample is never one shard's idiosyncrasy.
        self.n_calib_shards = max(2, int(n_calib_shards))
        self.climbmix_dir = climbmix_dir
        self.text_column = text_column
        self.nano_enc = nano_enc
        self.gemma_encode = gemma_encode or _default_gemma_encode(gemma_model)
        self._score_mm, self._docs_cache = {}, {}
        self._counts = {"ok": 0, "miss_shard": 0, "miss_row": 0, "drift": 0}
        self._index = None
        self.prefetcher = None   # optional ShardPrefetcher (Amendment 3); set by injection_train
        self._mm_lock = None     # set alongside prefetcher (guards memmap-cache eviction)
        if index_path and os.path.exists(index_path):
            self._index = _load_index(index_path)
        elif build_hash_index or index_path:
            self._index = self._build_hash_index()
            if index_path:
                _save_index(index_path, self._index)

    def __len__(self):
        return len(self._index) if self._index is not None else sum(len(self._docs(s)) for s in self.shards)

    def stats(self):
        return dict(self._counts)

    # -- positional (primary): (shard, row) from the ride-along enumeration --
    def lookup_by_row(self, sid, row, text, n_tokens):
        h = int(doc_hash(text))  # noise key: per-content, position-independent
        if sid not in self.shards:
            self._counts["miss_shard"] += 1
            return None, h
        docs = self._docs(sid)
        if not 0 <= row < len(docs):
            self._counts["miss_row"] += 1
            return None, h
        start, n = docs[row]
        out = self._align_and_gather(text, n_tokens, self._gemma_z(sid, start, n))
        self._counts["ok" if out is not None else "drift"] += 1
        return out, h

    # -- text-keyed fallback (needs a hash index) --
    def lookup(self, text, n_tokens):
        if self._index is None:
            raise RuntimeError(f"{self.name!r}: text-keyed lookup needs a hash index (pass "
                               f"build_hash_index=True/index_path, or use the positional "
                               f"ride-along path via lookup_by_row)")
        h = int(doc_hash(text))
        rec = self._index.get(h)
        if rec is None:
            return None, h
        sid, start, n = rec
        return self._align_and_gather(text, n_tokens, self._gemma_z(sid, start, n)), h

    def sample_activation_rows(self, k, seed, n_calib_shards=None):
        """Standardized GEMMA-grid rows (pre-alignment: this samples the scores, not
        the nanochat-aligned activations the loader serves), over the LOWEST
        ``n_calib_shards`` configured shards only.

        The bound is load-bearing, not a nicety. Calibration runs before
        _attach_prefetchers, so every shard opened here is a full ~GB
        hf_hub_download against the PRIMARY repo, on every rank's node — and a shard
        that lives in an overflow repo raises EntryNotFoundError. Striping ceil(k/n)
        docs over a handful of shards keeps the k-doc budget while touching a fixed,
        low, primary-repo-resident set. Same hazard __len__ is documented to avoid."""
        n_sh = self.n_calib_shards if n_calib_shards is None else max(2, int(n_calib_shards))
        sids = sorted(self.shards)[:n_sh]
        rng = np.random.default_rng(seed & 0x7FFFFFFF)
        rows, used = [], []
        per = -(-int(k) // max(len(sids), 1))    # ceil, so the k-doc budget is met
        for sid in sids:
            docs = self._docs(sid)
            for row in rng.permutation(len(docs))[:min(per, len(docs))]:
                start, n = docs[int(row)]
                if n > 0:
                    rows.append(self._gemma_z(sid, start, n))
                    used.append(sid)
                if len(rows) >= k:
                    break
            if len(rows) >= k:
                break
        n_shards_used = len(set(used))
        if len(self.shards) >= 2 and n_shards_used < 2:
            raise RuntimeError(
                f"{self.name!r}: calibration sample came from {n_shards_used} shard(s) of "
                f"{len(sids)} tried ({sorted(set(used))}) — one shard's idiosyncrasy would set "
                f"the whole run's channel_scale. Raise --loudness-k or n_calib_shards.")
        pooled, doc_index, n_docs, n_tokens = _pool_doc_rows(rows, self.r)
        self.last_sample_doc_index = doc_index
        self.last_sample_shards = sorted(set(used))
        return pooled, n_docs, n_tokens

    # -- internals --
    def _docs(self, sid):
        d = self._docs_cache.get(sid)
        if d is None:
            path = self._shard_file(sid, f"docs_{sid:05d}.jsonl")
            with open(path) as f:
                d = [(int(x["start"]), int(x["n"])) for x in map(json.loads, f)]
            if self._mm_lock is not None:
                with self._mm_lock:
                    self._docs_cache[sid] = d
            else:
                self._docs_cache[sid] = d
        return d

    def _scores(self, sid):
        mm = self._score_mm.get(sid)
        if mm is None:
            mm = np.load(self._shard_file(sid, f"scores_{sid:05d}.npy"), mmap_mode="r")  # int8 [N,L,54]
            if self._mm_lock is not None:
                with self._mm_lock:
                    self._score_mm[sid] = mm
            else:
                self._score_mm[sid] = mm
        return mm

    def _gemma_z(self, sid, start, n):
        q = self._scores(sid)[start:start + n, self.li][:, self.col_idx].astype(np.float32)  # (n, r)
        return self._standardize(q * self._scale + self._zero)  # int8 -> raw -> z

    def _shard_file(self, sid, name):
        if self.prefetcher is not None:      # rolling prefetch stages into a local dir
            self.prefetcher.ensure(sid)      # blocks until sid staged; fires starvation hook if behind
            return os.path.join(self.score_loc, name)
        if os.path.isdir(self.score_loc):
            return os.path.join(self.score_loc, name)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.score_loc, name, repo_type="dataset")

    def evict_shard(self, sid):
        """Drop cached memmaps for ``sid`` so the prefetcher can delete its files
        (Amendment 3 on_delete hook). Consumption is monotonic and keep_behind>=1,
        so an evicted shard is well behind the read frontier."""
        lock = self._mm_lock
        if lock is not None:
            lock.acquire()
        try:
            self._score_mm.pop(sid, None)
            self._docs_cache.pop(sid, None)
        finally:
            if lock is not None:
                lock.release()

    def _build_hash_index(self):
        import pyarrow.parquet as pq
        if not self.climbmix_dir:
            raise ValueError("build_hash_index needs climbmix_dir (parquet text to hash)")
        index = {}
        for sid in sorted(self.shards):
            docs = self._docs(sid)
            pf = pq.ParquetFile(os.path.join(self.climbmix_dir, f"shard_{sid:05d}.parquet"))
            i = 0
            for rg in range(pf.num_row_groups):
                for v in pf.read_row_group(rg, columns=[self.text_column]).column(0):
                    if i >= len(docs):
                        raise ValueError(f"shard {sid}: parquet has more rows than docs_{sid:05d}.jsonl")
                    start, n = docs[i]
                    index[int(doc_hash(v.as_py()))] = (sid, start, n)
                    i += 1
            if i != len(docs):
                raise ValueError(f"shard {sid}: parquet rows {i} != docs {len(docs)} (full-coverage broken)")
        return index


class LiveProbeScoreSource(_RuntimeProbeBase):
    """Live-gemma backend (bounded stub): ``score_fn(texts) -> [ (n_gemma, L, 54)
    raw float, ... ]`` scores each doc's gemma tokens on the fly; this class only
    slices+standardizes+aligns. Real gemma wiring for score_fn is a documented
    follow-up (cost: README). Tests inject a stub score_fn; gate='auto' is
    unsupported (no corpus to sample)."""

    def __init__(self, score_fn, score_loc, layer=8, *, nano_enc, gemma_encode=None,
                 gemma_model="google/gemma-2-2b", concepts=None, noise_sigma=0.0,
                 seed=0, name="probe-scores-live", align_policy="max", sample_texts=None):
        columns = _read_store_json(score_loc, "columns.json")
        quant = _read_store_json(score_loc, "quant.json")
        corpus_stats = _read_store_json(score_loc, "corpus_stats.json")
        self._init_layout(columns, quant, corpus_stats, layer, concepts, name, noise_sigma, seed,
                          align_policy)
        self.score_fn = score_fn
        self.score_loc = score_loc
        self.nano_enc = nano_enc
        self.gemma_encode = gemma_encode or _default_gemma_encode(gemma_model)
        # Startup-only calibration corpus: a donor/auto gate must resolve BEFORE
        # training, so the live scorer runs over these at startup (never lazily).
        self.sample_texts = list(sample_texts) if sample_texts else None

    def lookup(self, text, n_tokens):
        raw = np.asarray(self.score_fn([text])[0], np.float32)         # (n_gemma, L, 54) raw
        z = self._standardize(raw[:, self.li][:, self.col_idx])        # (n_gemma, r)
        return self._align_and_gather(text, n_tokens, z), 0

    def sample_activation_rows(self, k, seed):
        """Run the live scorer over a seeded sample of ``sample_texts`` at startup
        for donor/auto/dose calibration — a one-time cost, strictly before any
        training batch. Fails loudly if no calibration corpus was supplied."""
        if not self.sample_texts:
            raise NotImplementedError(
                f"{self.name!r}: donor/auto/dose calibration needs a startup sample corpus — construct "
                f"LiveProbeScoreSource(..., sample_texts=[...]) so the scorer runs over a sample "
                f"BEFORE training (it never scores lazily).")
        rng = np.random.default_rng(seed & 0x7FFFFFFF)
        idx = rng.permutation(len(self.sample_texts))[:min(k, len(self.sample_texts))]
        rows = []
        for i in idx:
            raw = np.asarray(self.score_fn([self.sample_texts[int(i)]])[0], np.float32)  # (n_gemma, L, 54)
            z = self._standardize(raw[:, self.li][:, self.col_idx])
            if z.shape[0] > 0:
                rows.append(z)
        pooled, doc_index, n_docs, n_tokens = _pool_doc_rows(rows, self.r)
        self.last_sample_doc_index = doc_index
        return pooled, n_docs, n_tokens


def _default_gemma_encode(gemma_model):
    """(text) -> (ids, char-offsets) via the gemma fast tokenizer, add_special_tokens=False
    (BOS-free, matching the scored tokenization). Loaded lazily; gemma-2-2b is gated."""
    from transformers import AutoTokenizer
    from nanochat.injection.align import get_offsets
    tok = AutoTokenizer.from_pretrained(gemma_model)
    return lambda text: get_offsets(tok, text, add_special_tokens=False)


_RUNTIME_INDEX_DTYPE = np.dtype([("hash", "<u8"), ("sid", "<i4"), ("off", "<i8"), ("n", "<i4")])


def _save_index(path, index):
    rec = np.fromiter(((h, sid, off, n) for h, (sid, off, n) in index.items()),
                      dtype=_RUNTIME_INDEX_DTYPE, count=len(index))
    np.save(path if path.endswith(".npy") else path + ".npy", rec)


def _load_index(path):
    rec = np.load(path if os.path.exists(path) else path + ".npy")
    return {int(h): (int(sid), int(off), int(n))
            for h, sid, off, n in zip(rec["hash"], rec["sid"], rec["off"], rec["n"])}


# --------------------------------------------------------------------------- #
# Structured activations from encoder probe-score predictions (Qwen-encoder
# precompute). Canonical cyclic class orderings are calendar/wheel order, NOT
# alphabetical; names match probe_set.json exactly.
# --------------------------------------------------------------------------- #
CYCLIC_ORDER = {
    "months": ["january", "february", "march", "april", "may", "june", "july",
               "august", "september", "october", "november", "december"],
    "weekdays": ["monday", "tuesday", "wednesday", "thursday", "friday",
                 "saturday", "sunday"],
    "seasons": ["spring", "summer", "autumn", "winter"],
    "directions": ["north", "northeast", "east", "southeast", "south",
                   "southwest", "west", "northwest"],
    "moon_phases": ["new_moon", "waxing_crescent", "first_quarter",
                    "waxing_gibbous", "full_moon", "waning_gibbous",
                    "last_quarter", "waning_crescent"],
    "color_wheel": ["red", "red-orange", "orange", "yellow", "yellow-green",
                    "green", "blue-green", "blue", "violet"],
}
NONCYCLIC_PCA = {"continents"}  # coords via saved family-PCA-2D


def build_structured_activations(preds, concepts, families, pca=None, pred_order=None):
    """preds (..., K) predicted probe scores for ONE gemma layer block ->
    (activations (..., r), legend), family-by-family: cyclic families collapse
    to a 2-D ring point (cos, sin), PCA families to 2 components.

    pred_order is the concept name of each preds COLUMN — this MUST be the
    encoder head's output order == the score store's main-block order ==
    probe_set.json "main_block_concepts" (family-sorted), NOT the name-sorted
    "concepts" key. Mixing them up attaches phase angles to the wrong concepts
    (the permutation bug; see the superproject's attribution/README.md).
    """
    if pred_order is None:
        pred_order = concepts
    idx = {c: i for i, c in enumerate(pred_order)}
    fam_to_concepts = {}
    for c in concepts:
        fam_to_concepts.setdefault(families[c], []).append(c)

    cols, legend = [], []
    for fam in sorted(fam_to_concepts):        # deterministic family order
        cs = fam_to_concepts[fam]
        if fam in CYCLIC_ORDER:
            order = CYCLIC_ORDER[fam]
            present = [c for c in order if c in idx]
            n = len(order)
            ang = np.array([2 * math.pi * order.index(c) / n for c in present])
            sub = preds[..., [idx[c] for c in present]]
            cx = (sub * np.cos(ang)).sum(-1)
            cy = (sub * np.sin(ang)).sum(-1)
            cols += [cx, cy]
            legend += [f"{fam}.cos", f"{fam}.sin"]
        elif fam in NONCYCLIC_PCA:
            sub = preds[..., [idx[c] for c in cs]]
            proj = sub @ pca[fam]                       # (m, 2) saved PCA
            cols += [proj[..., 0], proj[..., 1]]
            legend += [f"{fam}.pc1", f"{fam}.pc2"]
        else:  # 1-D fallback for any surviving scalar family
            for c in cs:
                cols.append(preds[..., idx[c]]); legend.append(c)
    acts = np.stack(cols, axis=-1).astype(np.float32)
    return acts, legend
