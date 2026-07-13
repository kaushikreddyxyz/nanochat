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

    def sample_activation_stats(self, k: int, seed: int):
        """(per-channel rms, per-channel nonzero-rate, n_docs, n_tokens) over a
        seeded sample of up to ``k`` docs — the statistics ``gate: "auto"`` needs.
        Sources that cannot be sampled (e.g. FnSource) leave this unimplemented."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support gate='auto' (no sampleable statistics)")


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

    def sample_activation_stats(self, k: int, seed: int):
        """Pool the (dequantized) rows of up to ``k`` seeded index entries and
        return per-channel (rms, nonzero-rate, n_docs, n_tokens) for auto-gate."""
        keys = list(self._index)
        rng = np.random.default_rng(seed & 0x7FFFFFFF)
        pick = rng.permutation(len(keys))[:min(k, len(keys))]
        rows = []
        for i in pick:
            off, n = self._index[keys[int(i)]]
            if n > 0:
                rows.append(self.mm[off:off + n].astype(np.float32) * self.scale)
        if not rows:
            return np.zeros(self.r, np.float32), np.zeros(self.r, np.float32), 0, 0
        pooled = np.concatenate(rows, axis=0)
        rms, nz = _channel_stats(pooled)
        return rms, nz, len(rows), pooled.shape[0]


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
# gemma->nanochat alignment (prefix mode: each nanochat token takes the LAST
# gemma token whose char span ends at or before it — causal, no future leakage)
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
                     noise_sigma, seed):
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

    def _standardize(self, raw):
        """raw (n, r) probe scores -> z (n, r); dequant already applied upstream."""
        return (raw - self._mean) / self._std

    def _align_and_gather(self, text, n_tokens, z_gemma):
        """z_gemma (n_gemma, r) standardized -> (n_tokens, r) on the nanochat grid,
        or None (unknown/drift). Unmapped nanochat tokens stay EXACT zero."""
        from nanochat.injection.align import gemma_to_qwen_map, nanochat_char_offsets
        nano_ids = self.nano_enc.encode_ordinary(text)
        if len(nano_ids) != n_tokens:            # drift vs the loader's body-token count
            return None
        nano_off = nanochat_char_offsets(self.nano_enc, nano_ids, text)
        g_ids, g_off = self.gemma_encode(text)
        if len(g_ids) != z_gemma.shape[0]:       # gemma retokenization disagrees with the scores
            return None
        amap = gemma_to_qwen_map(text, nano_off, g_off, mode="prefix")  # nano -> last gemma <= it
        out = np.zeros((n_tokens, self.r), np.float32)
        valid = amap >= 0
        if valid.any():
            out[valid] = z_gemma[amap[valid]]
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
                 build_hash_index=False, index_path=None, text_column="text"):
        columns = _read_store_json(score_loc, "columns.json")
        quant = _read_store_json(score_loc, "quant.json")
        corpus_stats = _read_store_json(score_loc, "corpus_stats.json")
        self._init_layout(columns, quant, corpus_stats, layer, concepts, name, noise_sigma, seed)
        self.score_loc = score_loc
        self.shards = set(int(s) for s in shards)
        self.climbmix_dir = climbmix_dir
        self.text_column = text_column
        self.nano_enc = nano_enc
        self.gemma_encode = gemma_encode or _default_gemma_encode(gemma_model)
        self._score_mm, self._docs_cache = {}, {}
        self._counts = {"ok": 0, "miss_shard": 0, "miss_row": 0, "drift": 0}
        self._index = None
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

    def sample_activation_stats(self, k, seed):
        rng = np.random.default_rng(seed & 0x7FFFFFFF)
        rows, n_docs = [], 0
        sids = sorted(self.shards)
        per = max(1, k // max(len(sids), 1))
        for sid in sids:
            docs = self._docs(sid)
            for row in rng.permutation(len(docs))[:min(per, len(docs))]:
                start, n = docs[int(row)]
                if n > 0:
                    rows.append(self._gemma_z(sid, start, n)); n_docs += 1
                if n_docs >= k:
                    break
            if n_docs >= k:
                break
        if not rows:
            return np.zeros(self.r, np.float32), np.zeros(self.r, np.float32), 0, 0
        pooled = np.concatenate(rows, axis=0)
        rms, nz = _channel_stats(pooled)
        return rms, nz, n_docs, pooled.shape[0]

    # -- internals --
    def _docs(self, sid):
        d = self._docs_cache.get(sid)
        if d is None:
            with open(self._shard_file(sid, f"docs_{sid:05d}.jsonl")) as f:
                d = [(int(x["start"]), int(x["n"])) for x in map(json.loads, f)]
            self._docs_cache[sid] = d
        return d

    def _scores(self, sid):
        mm = self._score_mm.get(sid)
        if mm is None:
            mm = np.load(self._shard_file(sid, f"scores_{sid:05d}.npy"), mmap_mode="r")  # int8 [N,L,54]
            self._score_mm[sid] = mm
        return mm

    def _gemma_z(self, sid, start, n):
        q = self._scores(sid)[start:start + n, self.li][:, self.col_idx].astype(np.float32)  # (n, r)
        return self._standardize(q * self._scale + self._zero)  # int8 -> raw -> z

    def _shard_file(self, sid, name):
        if os.path.isdir(self.score_loc):
            return os.path.join(self.score_loc, name)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.score_loc, name, repo_type="dataset")

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
                 seed=0, name="probe-scores-live"):
        columns = _read_store_json(score_loc, "columns.json")
        quant = _read_store_json(score_loc, "quant.json")
        corpus_stats = _read_store_json(score_loc, "corpus_stats.json")
        self._init_layout(columns, quant, corpus_stats, layer, concepts, name, noise_sigma, seed)
        self.score_fn = score_fn
        self.nano_enc = nano_enc
        self.gemma_encode = gemma_encode or _default_gemma_encode(gemma_model)

    def lookup(self, text, n_tokens):
        raw = np.asarray(self.score_fn([text])[0], np.float32)         # (n_gemma, L, 54) raw
        z = self._standardize(raw[:, self.li][:, self.col_idx])        # (n_gemma, r)
        return self._align_and_gather(text, n_tokens, z), 0


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
