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


def doc_hash(text: str) -> np.uint64:
    """Stable 64-bit content hash of a document string (order-independent key)."""
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return np.frombuffer(h, dtype="<u8")[0]


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
        """Gaussian noise seeded by (train seed, doc content hash): DDP-rank- and
        resume-independent, and not memorizable as a per-position identity."""
        if self.noise_sigma <= 0:
            return z
        rng = np.random.default_rng([self.seed & 0x7FFFFFFF, key & 0xFFFFFFFFFFFFFFFF])
        return z + rng.normal(0.0, self.noise_sigma, size=z.shape).astype(np.float32)


class QwenEncoderSource(_StoreSource):
    """Precomputed predictions of the frozen Qwen oracle-encoder (structured
    ring/PCA activations, r=14). Store built by scripts/precompute_activations.py
    modes fit/sweep/assemble."""
    source_kind = "qwen-encoder"


class ProbeScoreSource(_StoreSource):
    """Gold gemma probe scores (one layer's 54 standardized scores, or a
    configured concept subset), repackaged per nanochat token from the
    climbmix-scored HF stores by scripts/precompute_activations.py
    --mode repackage-probe-scores."""
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
