"""Baseline-nanochat Difference-of-Means (DoM) concept probes: one reading
direction per concept per block, fit on 1_dataset span-labeled text forwarded
through a frozen nanochat checkpoint. Mirrors concept_probes/2_probes DoM
(standardized-space mean(pos)-mean(neg), pos = token span-score >= 0.5), but in
nanochat residual space instead of gemma.

The pure-numpy core (assign_targets, DomAccum, auroc, batch_docs) has no torch /
model / network dependency and is unit-tested on CPU. capture_layers is the only
torch-touching helper.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import concepts as concept_registry  # noqa: E402

BINARIZE_AT = 0.5
VAR_EPS = 1e-8


def assign_targets(offsets, spans):
    """Per-token concept score [n]: max data-span score overlapping each token's
    (non-empty) char span, else 0. offsets: [n,2] char spans; spans: [[c0,c1,score]]."""
    offs = np.asarray(offsets, np.int64).reshape(-1, 2)
    y = np.zeros(len(offs), np.float64)
    if not spans:
        return y
    s0 = np.array([s[0] for s in spans], np.int64)
    s1 = np.array([s[1] for s in spans], np.int64)
    sc = np.array([s[2] for s in spans], np.float64)
    for t, (c0, c1) in enumerate(offs):
        if c1 <= c0:
            continue
        ov = (s0 < c1) & (s1 > c0)
        if ov.any():
            y[t] = sc[ov].max()
    return y


def tokenize_and_target(enc, text, spans):
    """(ids [n], y [n]). ids from encode_ordinary (no BOS); y per assign_targets."""
    from nanochat.injection.align import nanochat_char_offsets
    ids = enc.encode_ordinary(text)
    if not ids:
        return [], np.zeros(0, np.float64)
    offs = nanochat_char_offsets(enc, ids, text)
    return ids, assign_targets(offs, spans)


def batch_docs(docs, bos_id, max_tokens=16384, max_rows=64):
    """docs: list of (ids:list, y:np[n]). Yields (ids_bt [B,T] int64, mask [B,T] bool,
    y_cat [M] float) where M=mask.sum() and y_cat is row-major (b,t) aligned to the
    True positions of mask (BOS at col 0 is excluded). Sorted by length desc."""
    docs = [d for d in docs if len(d[0]) > 0]
    order = sorted(range(len(docs)), key=lambda i: -len(docs[i][0]))

    def emit(idxs):
        T = 1 + max(len(docs[i][0]) for i in idxs)
        B = len(idxs)
        ids = np.full((B, T), bos_id, np.int64)
        mask = np.zeros((B, T), bool)
        ys = []
        for b, i in enumerate(idxs):
            di, dy = docs[i]
            n = len(di)
            ids[b, 1:1 + n] = di
            mask[b, 1:1 + n] = True
            ys.append(dy)
        return ids, mask, np.concatenate(ys) if ys else np.zeros(0, np.float64)

    batch = []
    for i in order:
        cand = batch + [i]
        T = 1 + max(len(docs[j][0]) for j in cand)
        if batch and (len(cand) > max_rows or len(cand) * T > max_tokens):
            yield emit(batch)
            batch = [i]
        else:
            batch = cand
    if batch:
        yield emit(batch)


class _StopForward(Exception):
    pass


def capture_layers(model, ids_bt, device):
    """Forward [B,T] ids once, return list (len n_layer) of [B,T,d] fp32 block outputs.
    Hooks transformer.h[i] output == the same residual point 2_probes/probe_lib capture.
    Short-circuits after the last block (all captures are in hand) so the expensive
    lm_head over the full vocab never runs. Baseline is uncompiled, so the hook-raised
    stop is safe; a compiled model would need the plain (no-raise) path."""
    import torch
    n = len(model.transformer.h)
    outs = [None] * n
    hooks = []
    for i, blk in enumerate(model.transformer.h):
        def mk(j):
            def hook(_m, _inp, out):
                outs[j] = out.detach().float()
                if j == n - 1:
                    raise _StopForward
            return hook
        hooks.append(blk.register_forward_hook(mk(i)))
    try:
        with torch.no_grad():
            model(torch.from_numpy(ids_bt).to(device))
    except _StopForward:
        pass
    finally:
        for h in hooks:
            h.remove()
    return outs


class DomAccum:
    """Raw per-class pos/neg sums + global per-layer moments over TRAIN tokens.
    DoM_std = (pos_mean - neg_mean)/sd; the global mean cancels in the difference, so
    one pass and one standardization (global per-channel sd) suffice."""

    def __init__(self, n_layers, d, classes):
        self.L, self.d, self.classes = n_layers, d, list(classes)
        C = len(classes)
        self.pos_sum = np.zeros((C, n_layers, d), np.float64)
        self.neg_sum = np.zeros((C, n_layers, d), np.float64)
        self.pos_n = np.zeros((C, n_layers), np.int64)
        self.neg_n = np.zeros((C, n_layers), np.int64)
        self.g_sum = np.zeros((n_layers, d), np.float64)
        self.g_sq = np.zeros((n_layers, d), np.float64)
        self.g_n = 0

    def add(self, ci, layer_rows, y):
        """layer_rows: list (len L) of [M,d] fp32; y: [M] scores (shared across layers)."""
        pos = y >= BINARIZE_AT
        neg = ~pos
        for l, X in enumerate(layer_rows):
            Xd = X.astype(np.float64)
            self.pos_sum[ci, l] += Xd[pos].sum(0)
            self.neg_sum[ci, l] += Xd[neg].sum(0)
            self.g_sum[l] += Xd.sum(0)
            self.g_sq[l] += (Xd * Xd).sum(0)
            self.pos_n[ci, l] += int(pos.sum())
            self.neg_n[ci, l] += int(neg.sum())
        self.g_n += len(y)

    def finalize(self):
        mu = self.g_sum / max(self.g_n, 1)
        sd = np.sqrt(np.maximum(self.g_sq / max(self.g_n, 1) - mu ** 2, VAR_EPS))
        C = len(self.classes)
        W = np.zeros((C, self.L, self.d), np.float64)
        for ci in range(C):
            for l in range(self.L):
                pm = self.pos_sum[ci, l] / max(self.pos_n[ci, l], 1)
                nm = self.neg_sum[ci, l] / max(self.neg_n[ci, l], 1)
                W[ci, l] = (pm - nm) / sd[l]
        return {"W_dom": W, "mu": mu, "sd": sd,
                "pos_n": self.pos_n, "neg_n": self.neg_n, "g_n": int(self.g_n)}


def auroc(pos, neg):
    """Mann-Whitney AUROC of pos>neg. Ties get midranks. 0.5 if either side empty."""
    pos = np.asarray(pos, np.float64)
    neg = np.asarray(neg, np.float64)
    if pos.size == 0 or neg.size == 0:
        return 0.5
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(allv.size, np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    # midranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    start = csum - cnt
    mid = (start + csum + 1) / 2.0
    ranks = mid[inv]
    r_pos = ranks[:pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def dprime(pos, neg):
    pos = np.asarray(pos, np.float64)
    neg = np.asarray(neg, np.float64)
    if pos.size < 2 or neg.size < 2:
        return float("nan")
    denom = np.sqrt(0.5 * (pos.var(ddof=1) + neg.var(ddof=1)))
    return float((pos.mean() - neg.mean()) / denom) if denom > 0 else float("nan")


def project(X, mu_l, sd_l, w_cl):
    """Standardized-space projection of raw rows X [M,d] onto direction w_cl [d]."""
    return ((X.astype(np.float64) - mu_l) / sd_l) @ w_cl


def family_classes(family):
    return list(concept_registry.get_family(family).store_order)
