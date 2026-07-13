"""Oracle coords: fixed projection P, structured-coord construction, and the
document-keyed coord store consumed by the ride-along dataloader
(``nanochat.oracle.coord_dataloader``) during pretraining.

``scripts/base_train.py`` imports this only when injection is enabled
(``--inject-coords`` / ``--inject-config``); absent, nanochat trains vanilla.

Design:
  * Injection is CONTEXTUAL and per-token-occurrence, not per-token-id: each
    training position gets the r-dim coord vector produced by the frozen Qwen
    encoder over that token's document context.
  * The nanochat dataloader packs+crops documents (best-fit, ~35% cropped) in a
    data-dependent order, so a flat per-position memmap cannot be pre-aligned.
    Instead we store coords PER DOCUMENT keyed by a 64-bit hash of the doc text,
    and ride the coords through the exact same best-fit packing as the tokens.
    Same crop, same placement -> coords stay in lockstep with tokens, and the
    scheme is DDP-/order-independent (hash keying, not iteration position).
  * P is a FIXED random orthonormal (n_embd x r) matrix (seeded, saved once).
    Orthonormal columns => ||P z|| = ||z||, so the injected direction is an
    isometric embedding of the coord manifold.

Storage format (produced by ``scripts/precompute_coords.py``):
  <coords_dir>/coords.int8         memmap int8  [n_doc_tokens, r]   (standardized coords, quantized)
  <coords_dir>/index.npy           structured   [n_docs] fields (hash uint64, off int64, n int32)
  <coords_dir>/meta.json           {r, scale, families, class_order, P_path, encoder_ckpt, ...}
  <coords_dir>/P.npy               float32 [n_embd, r]  fixed orthonormal projection
"""
import hashlib
import json
import math
import os

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Canonical cyclic class orderings (NOT alphabetical -- calendar/wheel order).
# The phase of class k in a family of n is theta_k = 2*pi*k/n.
# Concept names match the probe set's probe_set.json exactly.
# --------------------------------------------------------------------------- #
CYCLIC_ORDER = {
    "months": ["january", "february", "march", "april", "may", "june", "july",
               "august", "september", "october", "november", "december"],
    "weekdays": ["monday", "tuesday", "wednesday", "thursday", "friday",
                 "saturday", "sunday"],
    "seasons": ["spring", "summer", "autumn", "winter"],
    "directions": ["north", "northeast", "east", "southeast", "south",
                   "southwest", "west", "northwest"],
    # new_moon .. waning_crescent, one full synodic cycle
    "moon_phases": ["new_moon", "waxing_crescent", "first_quarter",
                    "waxing_gibbous", "full_moon", "waning_gibbous",
                    "last_quarter", "waning_crescent"],
    # standard 9-survivor color wheel (3 blends dropped in Phase 0)
    "color_wheel": ["red", "red-orange", "orange", "yellow", "yellow-green",
                    "green", "blue-green", "blue", "violet"],
}
# non-cyclic families: coords via saved family-PCA-2D (continents, 6 classes)
NONCYCLIC_PCA = {"continents"}


def doc_hash(text: str) -> np.uint64:
    """Stable 64-bit content hash of a document string (order-independent key)."""
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return np.frombuffer(h, dtype="<u8")[0]


# --------------------------------------------------------------------------- #
# Fixed orthonormal projection P (n_embd x r), seeded + saved once.
# --------------------------------------------------------------------------- #
def make_orthonormal_P(n_embd: int, r: int, seed: int = 1337) -> np.ndarray:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n_embd, r, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(a, mode="reduced")   # (n_embd, r), orthonormal columns
    return q.to(torch.float32).numpy()


# --------------------------------------------------------------------------- #
# Structured coords from encoder probe-score predictions.
#   preds: (..., K) predicted probe scores for ONE gemma layer block (the
#          "layer-8" block, i.e. the K columns of the encoder head for
#          gemma layer 8), already corpus-standardized.
#   pred_order: the concept name of each of the K `preds` columns, IN COLUMN
#          ORDER. This MUST be the encoder head's output order, which equals
#          the score store's MAIN-block order == probe_set.json
#          "main_block_concepts" (family-sorted in the current store) -- NOT
#          "concepts" (name-sorted). Attaching phase angles to the wrong
#          concept was the coords half of the permutation bug (see the
#          superproject's PERMUTATION_FIX.md). If None, defaults to `concepts`
#          for backward compatibility (only correct for a fixed/rerun probe_set
#          where main_block_concepts == concepts).
#   families: {concept: family}
# Returns coords (..., r) ordered family-by-family, and a column legend.
# --------------------------------------------------------------------------- #
def build_coords(preds: np.ndarray, concepts, families, pca=None, pred_order=None):
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
            sub = preds[..., [idx[c] for c in present]]         # (..., m)
            cx = (sub * np.cos(ang)).sum(-1)
            cy = (sub * np.sin(ang)).sum(-1)
            cols += [cx, cy]
            legend += [f"{fam}.cos", f"{fam}.sin"]
        elif fam in NONCYCLIC_PCA:
            sub = preds[..., [idx[c] for c in cs]]              # (..., m)
            comp = pca[fam]                                     # (m, 2) saved PCA
            proj = sub @ comp
            cols += [proj[..., 0], proj[..., 1]]
            legend += [f"{fam}.pc1", f"{fam}.pc2"]
        else:  # 1-D fallback for any surviving scalar family
            for c in cs:
                cols.append(preds[..., idx[c]]); legend.append(c)
    coords = np.stack(cols, axis=-1).astype(np.float32)
    return coords, legend


# --------------------------------------------------------------------------- #
# Coord source: hash -> (offset, n) index over the int8 coord memmap.
# --------------------------------------------------------------------------- #
class CoordSource:
    def __init__(self, coords_dir, noise_sigma=0.15, seed=0):
        meta = json.load(open(os.path.join(coords_dir, "meta.json")))
        self.r = int(meta["r"])
        self.scale = float(meta["scale"])       # int8 -> float dequant scale
        self.noise_sigma = float(noise_sigma)
        self.seed = int(seed)
        self.mm = np.memmap(os.path.join(coords_dir, "coords.int8"),
                            dtype=np.int8, mode="r").reshape(-1, self.r)
        idx = np.load(os.path.join(coords_dir, "index.npy"))
        self._index = {int(h): (int(o), int(n))
                       for h, o, n in zip(idx["hash"], idx["off"], idx["n"])}

    def __len__(self):
        return len(self._index)

    def lookup(self, text: str, n_tokens: int):
        """Return ((n_tokens, r) float32 coords, hash) for a doc, or (None, hash)
        if the doc is not in the precompute or its stored token count mismatches
        (tokenizer drift). Callers MUST map None to EXACT zeros with NO noise, so
        the fallback keeps the injection an exact no-op (the injection site
        renormalizes any nonzero zc to full gate amplitude, so noised zeros would
        inject pure noise at full strength -- never do that)."""
        h = int(doc_hash(text))
        rec = self._index.get(h)
        if rec is None:
            return None, h
        off, n = rec
        if n != n_tokens:                       # tokenizer drift -> fail safe to zero
            return None, h
        return self.mm[off:off + n].astype(np.float32) * self.scale, h

    def add_noise(self, z: np.ndarray, key: int) -> np.ndarray:
        """Gaussian noise on standardized coords, deterministic per doc CONTENT
        (seeded by (train seed, doc hash)): rank-order-independent under DDP,
        reproducible across resumes, and never memorizable as a per-position
        identity (it varies with the doc, not the training position)."""
        if self.noise_sigma <= 0:
            return z
        rng = np.random.default_rng([self.seed & 0x7FFFFFFF, key & 0xFFFFFFFFFFFFFFFF])
        return z + rng.normal(0.0, self.noise_sigma, size=z.shape).astype(np.float32)
