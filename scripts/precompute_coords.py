"""Precompute per-document oracle coords over the nanochat pretraining corpus.

For every document in the nanochat ClimbMix shards (karpathy/climbmix-400b-shuffle,
the SAME parquet 'text' the dataloader reads), produce an (n_nanochat_tokens, r)
coord array and append it to an int8 memmap keyed by doc-content hash. The
nanochat model later rides these coords through best-fit packing in lockstep
with the tokens (see nanochat.oracle.coord_dataloader + nanochat.oracle.coords_store
= the CONSUMER contract this producer must match exactly). The full pipeline --
ending in the MANDATORY ``--mode preflight`` gate -- is documented in
nanochat/oracle/README.md.

Pipeline per doc (nanochat.oracle.align prefix mode -- the tokenizer-agnostic
module already validated at 7.08% crossing gemma->qwen; it serves
qwen->nanochat directly):

    nanochat-tokenize doc  -> BOS-less ids via tiktoken encode_ordinary
                              (token count == what coord_dataloader sees:
                               loader does len(encode(...,prepend=BOS))-1)
    char offsets           -> reconstructed from tiktoken token bytes
                              (RustBPE/tiktoken has no return_offsets_mapping;
                               accumulate decode_single_token_bytes lengths ->
                               byte spans -> char spans; audit-fixed partition
                               assert kept)
    qwen-tokenize doc      -> char offsets (HF fast tokenizer, add_special so
                              the encoder sees the SAME context it trained on)
    nano_tok t             -> last qwen token whose char span ends <= end(t)
    gather Qwen hidden     -> frozen Exp-A encoder head -> preds[3K]
    slice the LAYER-8 block -> preds[:, block*K:(block+1)*K]  (block=layers.index(8)=1
                              for layers=[6,8,14] => columns [54:108]; VERIFIED)
    build_coords            -> per-family structured coords, r=14
                              (6 cyclic families -> 2-D ring; continents -> PCA-2D)

Coord standardization + int8 (see `quantize` / the extended note there): the
per-column mean/std ARE computed and recorded in meta (required artifact), but
quantization is ZERO-PRESERVING (raw coord 0 -> int8 0 -> consumer 0 -> exact
injection no-op). Mean-centering is deliberately NOT applied: the self-
normalizing injection renormalizes ANY nonzero coord to full beta amplitude, so
a centered "no-concept" token (raw 0) would inject a full-strength constant
direction on ~every token -- exactly what the design's "no-concept -> zc=0 ->
term=0" invariant forbids. Noise (sigma=0.15) is added by the LOADER at train
time, never here.

Modes (--mode):
  fit              pod-0 one-time: fit continents PCA-2D + coord mean/std/scale
                   on a prefix sample; write coord_fit.npz (shared by all pods).
  sweep (default)  per pod: single pass over its round-robin shard slice, write
                   per-shard int8 store files (resumable, atomic); each shard's
                   Welford stats partial is embedded in its meta_<sid>.json.
  merge-stats      merge the per-shard Welford partials -> corpus stats.
  assemble         concatenate per-shard store files -> final coords.int8 /
                   index.npy / meta.json / P.npy (the consolidated store the
                   training node loads).
  verify           sample K docs from a finished store shard, recompute coords
                   live, assert int8 round-trip within scale, report zero frac.
  preflight        MANDATORY before training (CPU, no encoder): tokenize real
                   docs through the CONSUMER path (RustBPETokenizer.encode
                   batch + prepend BOS, exactly as coord_dataloader does) and
                   assert CoordSource.lookup hits on the assembled store.
                   Catches tokenizer-contract drift that would otherwise
                   silently zero every coord (run trains as baseline).
  measure-crossing align.crossing_rate for the qwen->nanochat pair over ~2k docs
                   (open audit item: prefix-mode assumption for tiktoken).

Run from the repo root (per pod; shard range split across the fleet by
--pod-index/--n-pods):
    python -m scripts.precompute_coords --mode fit   --encoder-ckpt <expA.pt> \
        --probe-set <dir>/probe_set.json --shards 0-3 --out /workspace/coords
    python -m scripts.precompute_coords --mode sweep --encoder-ckpt <expA.pt> \
        --probe-set <dir>/probe_set.json --shards 0-190 --out /workspace/coords \
        --pod-index 0 --n-pods 4
    python -m scripts.precompute_coords --mode merge-stats --out /workspace/coords
    python -m scripts.precompute_coords --mode assemble   --out /workspace/coords \
        --encoder-ckpt <expA.pt> --probe-set <dir>/probe_set.json --shards 0-190
    python -m scripts.precompute_coords --mode preflight --shards 0-190 \
        --out /workspace/coords --preflight-docs 1024      # MANDATORY gate
"""
import argparse
import glob
import json
import math
import os
import sys
import threading

import numpy as np

from nanochat.oracle.align import get_offsets, gemma_to_qwen_map, crossing_rate
from nanochat.oracle.coords_store import (build_coords, make_orthonormal_P, doc_hash,
                                          CYCLIC_ORDER, NONCYCLIC_PCA)

INDEX_DTYPE = np.dtype([("hash", "<u8"), ("off", "<i8"), ("n", "<i4")])


# --------------------------------------------------------------------------- #
# char-offset reconstruction for tiktoken/RustBPE (audit-fixed; partition assert)
# --------------------------------------------------------------------------- #
def nanochat_char_offsets(enc, ids, text):
    """Reconstruct (start,end) CHAR spans for tiktoken/RustBPE ids by accumulating
    per-token decoded byte lengths and converting byte spans -> char spans.

    ``enc`` is a tiktoken.Encoding (or any object with decode_single_token_bytes).
    ``ids`` must come from encode_ordinary(text) (NO BOS/special ids): byte-level
    BPE partitions text.encode('utf-8'), so the per-token byte lengths must sum to
    the doc's byte length (asserted). We use text's own bytes rather than
    enc.decode(ids) so tiktoken's errors='replace' decoding can never desync byte
    offsets. A char is attributed to the token holding its UTF-8 LEAD byte; a
    token that ends mid-character gets its span end just past that char and the
    next token starts there (empty spans possible for pure-continuation-byte
    tokens -- align.py treats empty source spans as -1, which build_coords then
    zero-fills)."""
    byte_lens = [len(enc.decode_single_token_bytes(int(i))) for i in ids]
    b = np.concatenate([[0], np.cumsum(byte_lens)]).astype(np.int64)  # byte boundary per token
    fb = text.encode("utf-8")
    assert int(b[-1]) == len(fb), (
        f"token byte lengths do not partition the document bytes "
        f"({int(b[-1])} != {len(fb)}); ids must come from encode_ordinary(text)")
    # byte offset -> char offset: a utf-8 continuation byte (0b10xxxxxx) does not
    # start a new char, so the char index increments only on lead bytes.
    # (vectorized -- this runs once per doc over ~13.5B tokens corpus-wide, a
    # per-byte python loop here measurably stalls the GPU feed)
    char_at = np.zeros(len(fb) + 1, dtype=np.int64)
    if len(fb):
        fb_arr = np.frombuffer(fb, dtype=np.uint8)
        char_at[1:] = np.cumsum((fb_arr & 0xC0) != 0x80)
    return [(int(char_at[b[i]]), int(char_at[b[i + 1]])) for i in range(len(ids))]


# --------------------------------------------------------------------------- #
# shard range parsing + pod round-robin
# --------------------------------------------------------------------------- #
def parse_shard_range(spec):
    """'0-190' or '0-3,10,20-22' -> sorted unique list of shard ids."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, c = part.split("-")
            out.update(range(int(a), int(c) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def assign_shards(all_shards, pod_index, n_pods):
    """Shard-level round-robin: pod p handles all_shards[p], all_shards[p+n_pods], ...
    Disjoint across pods; union == all_shards (coverage)."""
    assert 0 <= pod_index < n_pods, f"pod_index {pod_index} out of range [0,{n_pods})"
    return [s for i, s in enumerate(all_shards) if i % n_pods == pod_index]


# --------------------------------------------------------------------------- #
# probe-set + block layout resolution
# --------------------------------------------------------------------------- #
def load_probe_meta(probe_set_arg):
    """Accept either a probe_set.json file or a dir containing it."""
    path = probe_set_arg
    if os.path.isdir(path):
        path = os.path.join(path, "probe_set.json")
    with open(path) as f:
        ps = json.load(f)
    return ps


def resolve_layout(ps, layer8, r_check):
    """Return (concepts, families, pred_order, block, K, legend_len sanity)."""
    concepts = list(ps["concepts"])
    families = ps["families"]
    pred_order = ps.get("main_block_concepts")
    if pred_order is None:
        print("WARNING: probe_set.json has no 'main_block_concepts'; using name-sorted "
              "'concepts' as encoder pred column order. If the encoder was trained on the "
              "pre-fix (family-sorted) score store this attaches phase angles to the WRONG "
              "concepts (see out/PERMUTATION_FIX.md).", file=sys.stderr)
        pred_order = concepts
    assert set(concepts) == set(pred_order), \
        "main_block_concepts and concepts must be the same names (order differs)"
    layers = list(ps["layers"])
    if layer8 not in layers:
        raise ValueError(f"--layer8-block {layer8} not in probe layers {layers}")
    block = layers.index(layer8)  # which of the 3 K-wide blocks in preds[3K]
    K = len(concepts)
    # legend-length sanity: build_coords on a zero row must yield r_check cols.
    z0, legend = build_coords(np.zeros((1, K), np.float32), concepts, families,
                              pca=_zero_pca(concepts, families), pred_order=pred_order)
    if len(legend) != r_check:
        raise ValueError(f"build_coords legend has {len(legend)} cols (legend={legend}), "
                         f"expected r_check={r_check}")
    return concepts, families, pred_order, block, K, legend


def _zero_pca(concepts, families):
    """A placeholder PCA (identity-shaped zeros) so build_coords' legend can be
    probed before the real PCA is fit. continents -> (m,2) zeros."""
    pca = {}
    for fam in NONCYCLIC_PCA:
        m = sum(1 for c in concepts if families[c] == fam)
        if m:
            pca[fam] = np.zeros((m, 2), np.float32)
    return pca


# --------------------------------------------------------------------------- #
# tokenizers + encoder
# --------------------------------------------------------------------------- #
def load_nanochat_enc(tokenizer_dir):
    """Return the tiktoken Encoding backing the baseline RustBPE tokenizer.

    MUST be the baseline run's tokenizer.pkl -- coord/token alignment is keyed to
    its exact merges (pull tokenizer/ from HF oracle_baseline_noVE_d24_fp8)."""
    from nanochat.tokenizer import RustBPETokenizer
    tok = RustBPETokenizer.from_directory(tokenizer_dir)
    return tok.enc  # tiktoken.Encoding: encode_ordinary + decode_single_token_bytes


def make_qwen_encode(qwen_tok, add_special=True):
    """(substr) -> (ids, offsets) using align.get_offsets (fast HF tokenizer).

    add_special=True mirrors train_encoder.process_doc, which qwen-tokenized with
    add_special_tokens=True so the encoder sees its trained-on BOS/context. The
    qwen BOS (empty span) is never an alignment anchor, so it only affects the
    hidden states (as intended), not the map indices."""
    def _enc(substr):
        ids, offs = get_offsets(qwen_tok, substr, add_special_tokens=add_special)
        return ids, offs
    return _enc


def _encoder_head_class():
    """State-dict-compatible mirror of the superproject's
    ``train_encoder.EncoderHead`` (the checkpoint's ``head_state`` loads into it
    unchanged: keys ``up.weight``/``up.bias`` and, for expB-learn only,
    ``down.weight``). Defined lazily so this module stays importable without
    torch.nn at the top level."""
    import torch.nn as nn

    D_MODEL_GEMMA = 2304  # gemma-2-2b residual width (Exp B v* dimension)

    class EncoderHead(nn.Module):
        def __init__(self, hidden_size, K, mode):
            super().__init__()
            self.mode = mode
            if mode == "expA":
                self.up = nn.Linear(hidden_size, 3 * K)
                self.down = None
            elif mode == "expB-fixed":
                self.up = nn.Linear(hidden_size, K)
                self.down = None
            elif mode == "expB-learn":
                self.up = nn.Linear(hidden_size, K)
                self.down = nn.Linear(K, D_MODEL_GEMMA, bias=False)
            else:
                raise ValueError(f"unknown mode {mode!r}")

        def forward(self, h):
            y = self.up(h)
            v = self.down(y) if self.down is not None else None
            return y, v

    return EncoderHead


def load_encoder(ckpt_path, device, dtype):
    """Load the frozen Exp-A encoder (Qwen full-FT weights) + linear head from
    best.pt (the superproject's train_encoder.save_checkpoint structure)."""
    import torch
    from transformers import AutoModel
    EncoderHead = _encoder_head_class()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if state.get("mode") != "expA":
        print(f"WARNING: checkpoint mode={state.get('mode')!r} (expected 'expA'); "
              f"the coord head slice assumes a 3K expA up-projection.", file=sys.stderr)
    model_name = state["model_name"]
    hidden = state["hidden_size"]
    K = state["K"]
    model = AutoModel.from_pretrained(model_name, dtype=dtype)
    if "encoder_state" in state:
        model.load_state_dict(state["encoder_state"])  # full-FT weights
    head = EncoderHead(hidden, K, "expA")
    head.load_state_dict(state["head_state"])
    model.to(device).eval()
    head.to(device).eval()
    if dtype == torch.bfloat16:
        head.to(dtype)
    return model, head, hidden, model_name, K


# --------------------------------------------------------------------------- #
# per-doc segmentation (windowed to keep qwen forwards in-distribution)
# --------------------------------------------------------------------------- #
def iter_doc_segments(text, enc, qwen_encode, max_nano, max_qwen):
    """Split a doc into <=max_nano-token windows; per window build the qwen
    tokenization + prefix-align map. Returns (hash, n_body, [segment,...]).

    A segment: {win_start, win_len, q_ids (list[int]), amap (int64 [win_len],
    indices into q_ids or -1)}. Concatenating segment coords in win_start order
    reconstructs the full (n_body, r) coord array the loader expects."""
    nano_ids = enc.encode_ordinary(text)
    n = len(nano_ids)
    if n == 0:
        return doc_hash(text), 0, []
    nano_off = nanochat_char_offsets(enc, nano_ids, text)  # full-doc char spans
    segments = []
    for start in range(0, n, max_nano):
        end = min(start + max_nano, n)
        win_off = nano_off[start:end]
        char_lo = win_off[0][0]
        char_hi = max((e for (s, e) in win_off), default=char_lo)
        win_len = end - start
        if char_hi <= char_lo:
            # window is all empty spans -> no real chars -> zero coords
            segments.append({"win_start": start, "win_len": win_len,
                             "q_ids": [], "amap": np.full(win_len, -1, np.int64)})
            continue
        substring = text[char_lo:char_hi]
        rebased = [(max(0, s - char_lo), max(0, e - char_lo)) for (s, e) in win_off]
        q_ids, q_off = qwen_encode(substring)
        if len(q_ids) > max_qwen:
            # rare for same-language text; clip qwen -- tail nano tokens then map
            # to the last surviving qwen anchor (prefix mode), a graceful degrade.
            q_ids = list(q_ids[:max_qwen])
            q_off = list(q_off[:max_qwen])
        amap = gemma_to_qwen_map(substring, rebased, q_off, mode="prefix")
        segments.append({"win_start": start, "win_len": win_len,
                         "q_ids": list(q_ids), "amap": np.asarray(amap, np.int64)})
    return doc_hash(text), n, segments


# --------------------------------------------------------------------------- #
# CPU feeder pool: parallelize the per-doc segmentation (tiktoken encode +
# byte->char offsets + qwen tokenize + prefix align) across worker PROCESSES,
# while the GPU forward + int8 store writes stay in the main process.
#
# WHY: the sweep is CPU-bound in iter_doc_segments (single-core tokenization/
# alignment), leaving the GPU 40-65% idle. Workers load ONLY the two tokenizers
# (never the qwen MODEL / encoder head), so they are cheap and there is exactly
# one model on the GPU. `spawn` is used (never fork) so the CUDA context the main
# process created when loading the encoder is NOT inherited by the children.
#
# BYTE-IDENTICAL guarantee: iter_doc_segments is a pure function of
# (text, tokenizers, window sizes) producing only integer ids / int64 index
# arrays -- no floats, no RNG. imap() preserves INPUT ORDER, so the main process
# calls engine.add_doc / emit in the exact same doc order as the serial path.
# Same order => same coords bytes, same index, same Welford accumulation order,
# same hashes. The change is a pure throughput refactor (opt-in via
# --feeder-workers N; N=0 keeps the original serial path untouched).
# --------------------------------------------------------------------------- #
_FEEDER = {}  # per-worker-process globals (tokenizers + window sizes)


def _feeder_init(nano_tokenizer_dir, qwen_model_name, qwen_add_special,
                 max_doc_tokens, max_qwen_tokens):
    """Worker-process initializer (runs once per spawned worker). Loads ONLY the
    nanochat tiktoken Encoding and the qwen fast tokenizer -- no torch, no CUDA,
    no encoder weights."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # 1 rust thread/worker
    enc = load_nanochat_enc(nano_tokenizer_dir)
    from transformers import AutoTokenizer
    qtok = AutoTokenizer.from_pretrained(qwen_model_name)
    if qtok.pad_token_id is None:
        qtok.pad_token = qtok.eos_token
    _FEEDER["enc"] = enc
    _FEEDER["qwen_encode"] = make_qwen_encode(qtok, add_special=qwen_add_special)
    _FEEDER["max_doc_tokens"] = max_doc_tokens
    _FEEDER["max_qwen_tokens"] = max_qwen_tokens


def _feeder_worker(text):
    """Map one doc text -> (hash, n_body, segments), identical to the serial call
    in iter_doc_segments. Returned segments carry int lists + int64 arrays only."""
    return iter_doc_segments(text, _FEEDER["enc"], _FEEDER["qwen_encode"],
                             _FEEDER["max_doc_tokens"], _FEEDER["max_qwen_tokens"])


class _FeederPool:
    """Bounded, ORDER-PRESERVING spawn pool over _feeder_worker.

    imap_docs(texts) yields (hash, n, segments) in input order. A semaphore caps
    the number of in-flight docs (backpressure: the pool's feeder thread blocks
    on next() once `prefetch` docs are outstanding), so a huge shard never pulls
    all its text into memory and the in-order reorder buffer stays bounded.
    stats() exposes queue depth + worker liveness for the heartbeat."""

    def __init__(self, n_workers, nano_tokenizer_dir, qwen_model_name,
                 qwen_add_special, max_doc_tokens, max_qwen_tokens, prefetch,
                 worker=None, initializer=None, initargs=None):
        # worker/initializer/initargs are injection points for tests ONLY; in
        # production they default to the real tokenizer-loading feeder functions.
        import multiprocessing as mp
        self.n_workers = int(n_workers)
        self.prefetch = int(prefetch)
        self._worker = worker or _feeder_worker
        self._outstanding = 0
        self._lock = threading.Lock()
        ctx = mp.get_context("spawn")
        self._pool = ctx.Pool(
            processes=self.n_workers,
            initializer=initializer or _feeder_init,
            initargs=(initargs if initargs is not None else
                      (nano_tokenizer_dir, qwen_model_name, qwen_add_special,
                       max_doc_tokens, max_qwen_tokens)))

    def imap_docs(self, texts_iter):
        sem = threading.Semaphore(self.prefetch)

        def bounded():
            for text in texts_iter:
                sem.acquire()
                with self._lock:
                    self._outstanding += 1
                yield text

        for res in self._pool.imap(self._worker, bounded(), chunksize=1):
            sem.release()
            with self._lock:
                self._outstanding -= 1
            yield res

    def stats(self):
        with self._lock:
            depth = self._outstanding
        try:
            alive = sum(int(p.is_alive()) for p in self._pool._pool)
        except Exception:
            alive = -1
        return {"queue_depth": depth, "workers_alive": alive,
                "feeder_workers": self.n_workers}

    def close(self):
        try:
            self._pool.terminate()
            self._pool.join()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# batched encoder forward -> per-segment L8 preds -> per-doc float coords
# --------------------------------------------------------------------------- #
class CoordEngine:
    """Batches qwen segments across docs into padded forwards, gathers L8-block
    preds at the alignment indices, and builds per-family float coords.

    Pure w.r.t. numpy/torch: the caller supplies model+head (real Qwen or a tiny
    test stand-in). Emits completed docs (hash, n, float32 coords [n,r]) in
    completion order (order is irrelevant -- the store is hash-keyed)."""

    def __init__(self, model, head, block, K, concepts, families, pca, pred_order,
                 r, device, dtype, pad_id, batch_seqs=32,
                 fast_forward=False, max_batch_tokens=32768, seg_buffer=4096):
        self.model, self.head = model, head
        self.block, self.K = block, K
        self.concepts, self.families, self.pca, self.pred_order = concepts, families, pca, pred_order
        self.r = r
        self.device, self.dtype = device, dtype
        self.pad_id = pad_id
        self.batch_seqs = batch_seqs
        # --fast-forward: buffer many segments across docs, sort by qwen length,
        # and pack each padded forward to a MAX-TOKEN budget (B*maxlen) instead of
        # a fixed sequence count. Length bucketing kills the ragged-padding waste
        # of the serial FIFO path and lets each forward saturate the H100. Doc
        # completion order changes (irrelevant: the store is hash-keyed and each
        # doc's rows are still written contiguously via win_start scatter); coords
        # equal the serial path within one int8 step (fp-noise from batch shape),
        # and the zero-fallback (unmapped -> exact 0) is bit-identical by
        # construction (set in _gather, independent of batching).
        self.fast_forward = bool(fast_forward)
        self.max_batch_tokens = int(max_batch_tokens)
        self.seg_buffer = int(seg_buffer)
        self._flush_threshold = self.seg_buffer if self.fast_forward else batch_seqs
        self._pending = []                # list of (doc_key, segment)
        self._docs = {}                   # doc_key -> {"coords","remaining","n","hash"}
        self._order = []                  # doc_keys in add order (stable tie-break)
        self._ready = []                  # completed doc_keys

    def add_doc(self, doc_key, hsh, n, segments):
        if n == 0:
            return  # empty doc: nothing to store
        self._docs[doc_key] = {"coords": np.zeros((n, self.r), np.float32),
                               "remaining": len(segments), "n": n, "hash": hsh}
        self._order.append(doc_key)
        for seg in segments:
            if len(seg["q_ids"]) == 0:
                # all-unmapped window -> zero coords already in the buffer
                self._docs[doc_key]["remaining"] -= 1
            else:
                self._pending.append((doc_key, seg))
        if self._docs[doc_key]["remaining"] == 0:
            self._complete(doc_key)
        if len(self._pending) >= self._flush_threshold:
            self.flush()

    def _complete(self, doc_key):
        self._ready.append(doc_key)

    def _forward_l8(self, batch):
        """One padded forward over the batch's segments -> L8-block preds
        [B, L, K] (float32 numpy). Padding is masked by attention_mask, so real
        positions are unaffected by batch composition (up to bf16 fp-noise)."""
        import torch
        maxlen = max(len(seg["q_ids"]) for _, seg in batch)
        B = len(batch)
        # Build on the host as one numpy block (no per-row torch.tensor alloc),
        # then a single pinned-ish H2D copy per tensor.
        ids_np = np.full((B, maxlen), self.pad_id, dtype=np.int64)
        attn_np = np.zeros((B, maxlen), dtype=np.int64)
        for i, (_, seg) in enumerate(batch):
            q = seg["q_ids"]
            L = len(q)
            ids_np[i, :L] = q
            attn_np[i, :L] = 1
        input_ids = torch.from_numpy(ids_np).to(self.device, non_blocking=True)
        attn = torch.from_numpy(attn_np).to(self.device, non_blocking=True)
        with torch.inference_mode():
            out = self.model(input_ids=input_ids, attention_mask=attn)
            hidden = out.last_hidden_state                       # [B, L, H]
            preds, _ = self.head(hidden)                         # [B, L, 3K]
            l8 = preds[..., self.block * self.K:(self.block + 1) * self.K]  # [B, L, K]
            return l8.float().cpu().numpy()

    def _gather(self, seg, l8_row):
        """Scatter the segment's aligned qwen preds onto its nano-token grid;
        unmapped (-1) tokens stay zero."""
        amap = seg["amap"]
        win_len = seg["win_len"]
        gathered = np.zeros((win_len, self.K), np.float32)
        valid = amap >= 0
        if valid.any():
            gathered[valid] = l8_row[amap[valid], :]
        return gathered

    def flush(self):
        if self.fast_forward:
            self._flush_bucketed()
            return
        # Serial path: chunk into batch_seqs-sized forwards in FIFO order. A
        # single long doc can enqueue far more than batch_seqs windows at once,
        # and a monolithic forward over all of them would OOM on rare giant docs.
        while self._pending:
            batch = self._pending[:self.batch_seqs]
            self._pending = self._pending[self.batch_seqs:]
            l8 = self._forward_l8(batch)
            for i, (doc_key, seg) in enumerate(batch):
                self._consume(doc_key, seg, self._gather(seg, l8[i]))

    def _flush_bucketed(self):
        """Length-bucketed cross-doc batching. Sort all buffered segments by
        qwen length, then greedily pack forwards to a max padded-token budget
        (B*maxlen <= max_batch_tokens). Similar lengths batch together, so the
        pad waste that dominates the ragged FIFO path collapses and each forward
        is sized to saturate the GPU rather than latency-bound at batch_seqs."""
        pend = self._pending
        self._pending = []
        if not pend:
            return
        order = sorted(range(len(pend)), key=lambda i: len(pend[i][1]["q_ids"]))
        i, N = 0, len(order)
        while i < N:
            # pend[order[i]] is the shortest remaining; as we grow the batch the
            # running max length is the newest (sorted-asc) element. Always keep
            # >=1 segment so a lone segment longer than the budget still forwards.
            j = i + 1
            while j < N:
                cand_max = len(pend[order[j]][1]["q_ids"])
                if (j - i + 1) * cand_max > self.max_batch_tokens:
                    break
                j += 1
            batch = [pend[order[k]] for k in range(i, j)]
            l8 = self._forward_l8(batch)
            for k, (doc_key, seg) in enumerate(batch):
                self._consume(doc_key, seg, self._gather(seg, l8[k]))
            i = j

    def _consume(self, doc_key, seg, gathered):
        z, _ = build_coords(gathered, self.concepts, self.families,
                            pca=self.pca, pred_order=self.pred_order)  # [win_len, r]
        d = self._docs[doc_key]
        d["coords"][seg["win_start"]:seg["win_start"] + seg["win_len"]] = z
        d["remaining"] -= 1
        if d["remaining"] == 0:
            self._complete(doc_key)

    def drain(self, final=True):
        """Yield (hash, n, coords) for completed docs.

        Serial path (and the FINAL drain of any shard) flush the pending
        segments first. In --fast-forward mode the sweep's per-doc, in-loop
        drain passes final=False so pending segments keep ACCUMULATING into big
        length-bucketed batches (add_doc triggers the real flush once seg_buffer
        segments are queued). Forcing a flush after every doc -- as the original
        unconditional flush did -- collapses each forward back to just that doc's
        1-2 segments and completely defeats the cross-doc bucketing (the GPU then
        runs hundreds of tiny latency-bound forwards instead of a few saturating
        ones). The final drain (final=True, the default) flushes the tail so no
        doc is left buffered."""
        if final or not self.fast_forward:
            self.flush()
        for doc_key in self._ready:
            d = self._docs.pop(doc_key)
            yield d["hash"], d["n"], d["coords"]
        self._ready = []
        # `_order` entries for popped docs are stale but harmless.


# --------------------------------------------------------------------------- #
# quantization (zero-preserving; see module docstring / note here)
# --------------------------------------------------------------------------- #
def compute_scale(coord_std, clip_sigma):
    """Single global int8 scale from the per-column stats. Coords are already in
    (standardized-score) units ~O(1), comparable across columns, so a single
    scale covering +-clip_sigma*max(std) resolves active concepts while keeping
    resolution fine. NOTE: NO mean-centering -- raw coord 0 must stay 0 so the
    self-normalizing injection is an exact no-op on concept-free tokens."""
    s = clip_sigma * float(np.max(coord_std)) / 127.0
    return max(s, 1e-8)


def quantize(coords_f, scale):
    """coords_f [n,r] float -> int8 [n,r]. Zero-preserving: 0 -> 0."""
    q = np.round(coords_f / scale)
    q = np.clip(q, -127, 127)
    return q.astype(np.int8)


# --------------------------------------------------------------------------- #
# Welford running stats (per column), mergeable like score_corpus --merge-stats
# --------------------------------------------------------------------------- #
class Welford:
    def __init__(self, r):
        self.n = 0
        self.mean = np.zeros(r, np.float64)
        self.M2 = np.zeros(r, np.float64)

    def update(self, x):  # x: [m, r]
        x = np.asarray(x, np.float64)
        m = x.shape[0]
        if m == 0:
            return
        nb = self.n + m
        delta = x.mean(0) - self.mean
        self.mean += delta * (m / nb)
        self.M2 += x.var(0) * m + (delta ** 2) * (self.n * m / nb)
        self.n = nb

    def to_dict(self):
        return {"n": int(self.n), "mean": self.mean.tolist(), "M2": self.M2.tolist()}

    @staticmethod
    def merge_dicts(parts):
        n = 0
        mean = None
        M2 = None
        for p in parts:
            pn = int(p["n"])
            if pn == 0:
                continue
            pm = np.asarray(p["mean"], np.float64)
            pM2 = np.asarray(p["M2"], np.float64)
            if mean is None:
                n, mean, M2 = pn, pm.copy(), pM2.copy()
                continue
            nb = n + pn
            delta = pm - mean
            mean = mean + delta * (pn / nb)
            M2 = M2 + pM2 + (delta ** 2) * (n * pn / nb)
            n = nb
        if mean is None:
            return 0, None, None
        std = np.sqrt(M2 / max(n, 1))
        return n, mean, std


# --------------------------------------------------------------------------- #
# corpus iteration over the karpathy climbmix shards
# --------------------------------------------------------------------------- #
def default_climbmix_dir():
    base = os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat"))
    return os.path.join(base, "base_data_climbmix")


def shard_path(climbmix_dir, sid):
    return os.path.join(climbmix_dir, f"shard_{sid:05d}.parquet")


def iter_shard_texts(climbmix_dir, sid, text_column="text"):
    """Yield each doc's raw text from shard_<sid>.parquet, in stored row order
    (matches how the dataloader reads the 'text' column)."""
    import pyarrow.parquet as pq
    path = shard_path(climbmix_dir, sid)
    if not os.path.exists(path):
        raise FileNotFoundError(f"shard {sid} not found at {path}")
    pf = pq.ParquetFile(path)
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx, columns=[text_column])
        for v in rg.column(0):
            yield v.as_py()


# --------------------------------------------------------------------------- #
# FIT: continents PCA-2D + coord mean/std/scale on a prefix sample (pod 0)
# --------------------------------------------------------------------------- #
def fit_pca_2d(x):
    """Deterministic PCA-2D of x [m, d]: top-2 right singular vectors of the
    centered data, with a fixed sign convention so two runs agree bit-for-bit."""
    x = np.asarray(x, np.float64)
    xc = x - x.mean(0, keepdims=True)
    # economy SVD; components = V[:, :2]
    _, _, Vt = np.linalg.svd(xc, full_matrices=False)
    comp = Vt[:2].T.copy()                      # [d, 2]
    for j in range(comp.shape[1]):
        k = int(np.argmax(np.abs(comp[:, j])))  # fix sign: largest-|loading| positive
        if comp[k, j] < 0:
            comp[:, j] = -comp[:, j]
    return comp.astype(np.float32)


def continents_pred_columns(concepts, families, pred_order):
    """Column indices (into the K-wide L8 preds) of continents concepts, in the
    order build_coords consumes them (concepts filtered to the family, then
    indexed by pred_order)."""
    idx = {c: i for i, c in enumerate(pred_order)}
    cs = [c for c in concepts if families[c] == "continents"]
    return [idx[c] for c in cs]


def run_fit(args):
    import torch
    concepts, families, pred_order, block, K, legend = resolve_layout(
        load_probe_meta(args.probe_set), args.layer8_block, args.r_check)
    dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
    model, head, hidden, model_name, K2 = load_encoder(args.encoder_ckpt, args.device, dtype)
    assert K2 == K, f"encoder K={K2} != probe_set K={K}"
    enc = load_nanochat_enc(args.nano_tokenizer_dir)
    qwen_tok = _load_qwen_tok(args, model_name)
    qwen_encode = make_qwen_encode(qwen_tok, add_special=not args.qwen_no_special)

    # 1) collect raw L8 preds for a prefix sample (cap rows)
    cont_cols = continents_pred_columns(concepts, families, pred_order)
    pred_buf = []
    n_rows = 0
    shards = parse_shard_range(args.shards)
    engine = _RawPredEngine(model, head, block, K, args.device, dtype,
                            qwen_tok.pad_token_id, args.batch_seqs)
    print(f"[fit] collecting up to {args.fit_tokens} pred rows from shards {shards[:4]}...")
    done = False
    for sid in shards:
        for text in iter_shard_texts(args.climbmix_dir, sid, args.text_column):
            hsh, n, segs = iter_doc_segments(text, enc, qwen_encode,
                                             args.max_doc_tokens, args.max_qwen_tokens)
            engine.add_doc(hsh, n, segs)
            for preds in engine.drain():
                pred_buf.append(preds)
                n_rows += preds.shape[0]
            if n_rows >= args.fit_tokens:
                done = True
                break
        if done:
            break
    for preds in engine.drain_final():
        pred_buf.append(preds)
    preds_all = np.concatenate(pred_buf, axis=0)[:args.fit_tokens]  # [N, K]
    print(f"[fit] collected {preds_all.shape[0]} pred rows (K={K})")

    # 2) fit continents PCA on those preds
    cont = preds_all[:, cont_cols]                      # [N, m]
    pca_comp = fit_pca_2d(cont)                         # [m, 2]
    pca = {"continents": pca_comp}

    # 3) build coords for the sample, derive coord mean/std + global scale
    coords, legend = build_coords(preds_all, concepts, families, pca=pca, pred_order=pred_order)
    assert coords.shape[1] == args.r_check
    coord_mean = coords.mean(0).astype(np.float32)
    coord_std = coords.std(0).astype(np.float32)
    scale = compute_scale(np.maximum(coord_std, 1e-8), args.clip_sigma)
    # clip fraction diagnostic
    q = quantize(coords, scale)
    clip_frac = float(np.mean(np.abs(q.astype(np.int32)) >= 127))

    fit_path = os.path.join(args.out, "coord_fit.npz")
    os.makedirs(args.out, exist_ok=True)
    np.savez(fit_path + ".tmp.npz",
             pca_components=pca_comp, coord_mean=coord_mean, coord_std=coord_std,
             scale=np.float32(scale), clip_sigma=np.float32(args.clip_sigma),
             legend=np.array(legend), pred_order=np.array(pred_order),
             block=np.int64(block), n_fit=np.int64(coords.shape[0]))
    os.replace(fit_path + ".tmp.npz", fit_path)
    meta = {"scale": float(scale), "clip_sigma": float(args.clip_sigma),
            "clip_frac_fit": clip_frac, "n_fit": int(coords.shape[0]),
            "coord_mean": coord_mean.tolist(), "coord_std": coord_std.tolist(),
            "legend": list(legend), "pca_fit_hash": _hash_array(pca_comp)}
    with open(os.path.join(args.out, "coord_fit.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[fit] wrote {fit_path}: scale={scale:.5g} clip_frac={clip_frac:.4%} "
          f"pca_hash={meta['pca_fit_hash'][:12]}")
    return fit_path


class _RawPredEngine(CoordEngine):
    """Variant of CoordEngine that emits per-doc raw L8 preds [n,K] (not coords)
    -- used by --fit before PCA/stats exist. Reuses the batching machinery."""

    def __init__(self, model, head, block, K, device, dtype, pad_id, batch_seqs):
        # build_coords deps unused; pass placeholders
        super().__init__(model, head, block, K, [], {}, {}, [], K, device, dtype,
                         pad_id, batch_seqs)
        self._raw = {}

    def add_doc(self, hsh, n, segments):  # note: no doc_key arg (hash is key)
        if n == 0:
            return
        key = len(self._order)
        self._docs[key] = {"coords": None, "remaining": len(segments), "n": n, "hash": hsh}
        self._raw[key] = np.zeros((n, self.K), np.float32)
        self._order.append(key)
        for seg in segments:
            if len(seg["q_ids"]) == 0:
                self._docs[key]["remaining"] -= 1
            else:
                self._pending.append((key, seg))
        if self._docs[key]["remaining"] == 0:
            self._ready.append(key)
        if len(self._pending) >= self.batch_seqs:
            self.flush()

    def _consume(self, key, seg, gathered):
        # raw preds, no build_coords (PCA/stats don't exist yet at fit time)
        self._raw[key][seg["win_start"]:seg["win_start"] + seg["win_len"]] = gathered
        self._docs[key]["remaining"] -= 1
        if self._docs[key]["remaining"] == 0:
            self._complete(key)

    def drain(self):
        self.flush()
        for key in self._ready:
            self._docs.pop(key, None)
            yield self._raw.pop(key)
        self._ready = []

    def drain_final(self):
        yield from self.drain()


def _hash_array(a):
    import hashlib
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=16).hexdigest()


def _load_qwen_tok(args, model_name):
    from transformers import AutoTokenizer
    name = args.qwen_model or model_name
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_fit(out_dir):
    p = os.path.join(out_dir, "coord_fit.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"coord_fit.npz missing in {out_dir}: run `--mode fit` on pod 0 first "
            f"(fits continents PCA + coord scale; shared by all pods).")
    z = np.load(p, allow_pickle=True)
    return {"pca": {"continents": z["pca_components"].astype(np.float32)},
            "coord_mean": z["coord_mean"].astype(np.float32),
            "coord_std": z["coord_std"].astype(np.float32),
            "scale": float(z["scale"]), "clip_sigma": float(z["clip_sigma"]),
            "legend": list(z["legend"]), "pred_order": list(z["pred_order"]),
            "block": int(z["block"]), "pca_hash": _hash_array(z["pca_components"])}


# --------------------------------------------------------------------------- #
# SWEEP: per-shard store files (resumable, atomic) + partial Welford stats
# --------------------------------------------------------------------------- #
def shard_done_marker(out_dir, sid):
    return os.path.join(out_dir, "shards", f"shard_{sid:05d}.done")


def run_sweep(args):
    import torch
    from tqdm import tqdm
    if args.fast_forward and str(args.device).startswith("cuda"):
        # tf32 for any fp32 matmul (model+head are bf16 so effect is marginal,
        # but free); does not touch the bf16 compute path.
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    out_shards = os.path.join(args.out, "shards")
    os.makedirs(out_shards, exist_ok=True)
    ps = load_probe_meta(args.probe_set)
    concepts, families, pred_order, block, K, legend = resolve_layout(ps, args.layer8_block, args.r_check)
    fit = load_fit(args.out)
    if fit["block"] != block:
        raise ValueError(f"fit block {fit['block']} != resolved block {block}")
    if [str(c) for c in fit["pred_order"]] != [str(c) for c in pred_order]:
        raise ValueError(
            "coord_fit.npz pred_order != probe_set main_block_concepts: the fit "
            "was run against a different probe_set.json -- phase angles would "
            "attach to the WRONG concepts. Re-run --mode fit with this probe set.")
    if [str(c) for c in fit["legend"]] != [str(c) for c in legend]:
        raise ValueError(f"coord_fit.npz legend {fit['legend']} != resolved legend {legend}")
    r = args.r_check
    dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
    model, head, hidden, model_name, K2 = load_encoder(args.encoder_ckpt, args.device, dtype)
    assert K2 == K
    enc = load_nanochat_enc(args.nano_tokenizer_dir)
    qwen_tok = _load_qwen_tok(args, model_name)
    qwen_encode = make_qwen_encode(qwen_tok, add_special=not args.qwen_no_special)

    all_shards = parse_shard_range(args.shards)
    my_shards = assign_shards(all_shards, args.pod_index, args.n_pods)
    print(f"[sweep] pod {args.pod_index}/{args.n_pods}: {len(my_shards)} shards "
          f"{my_shards[:6]}{'...' if len(my_shards) > 6 else ''}")
    if args.fast_forward:
        print(f"[sweep] FAST-FORWARD: length-bucketed batching, "
              f"max_batch_tokens={args.max_batch_tokens}, seg_buffer={args.seg_buffer} "
              f"(coords equal serial within 1 int8 step; zero-fallback exact)")

    # Opt-in CPU feeder pool: parallelize per-doc segmentation across worker
    # processes (tokenizers only, spawn). N=0 keeps the original serial path.
    pool = None
    if args.feeder_workers and args.feeder_workers > 0:
        prefetch = args.feeder_prefetch or max(4 * args.feeder_workers, 64)
        pool = _FeederPool(
            args.feeder_workers, args.nano_tokenizer_dir,
            args.qwen_model or model_name, not args.qwen_no_special,
            args.max_doc_tokens, args.max_qwen_tokens, prefetch)
        print(f"[sweep] feeder pool: {args.feeder_workers} spawn workers, "
              f"prefetch={prefetch} (per-doc segmentation parallelized; GPU forward "
              f"+ writes stay in main; output byte-identical to serial path)")

    hb_path = args.heartbeat_path
    try:
        for sid in my_shards:
            if os.path.exists(shard_done_marker(args.out, sid)):
                print(f"[sweep] shard {sid}: DONE (skip)")
                # its Welford partial already lives in meta_<sid>.json (written at
                # shard publish) -- merge-stats reads per-SHARD partials, so a
                # crash+resume never loses or double-counts stats.
                continue
            _sweep_one_shard(args, sid, enc, qwen_encode, qwen_tok, model, head, block, K,
                             concepts, families, pred_order, fit, r, dtype, hb_path, pool)
    finally:
        if pool is not None:
            pool.close()
    print(f"[sweep] pod {args.pod_index} done ({len(my_shards)} shards; per-shard "
          f"Welford partials live in meta_<sid>.json)")


def _sweep_one_shard(args, sid, enc, qwen_encode, qwen_tok, model, head, block, K,
                     concepts, families, pred_order, fit, r, dtype, hb_path, pool=None):
    from tqdm import tqdm
    out_shards = os.path.join(args.out, "shards")
    tmp_int8 = os.path.join(out_shards, f"coords_{sid:05d}.int8.tmp")
    tmp_idx = os.path.join(out_shards, f"index_{sid:05d}.tmp.npy")  # .npy suffix so np.save won't re-append
    engine = CoordEngine(model, head, block, K, concepts, families, fit["pca"], pred_order,
                         r, args.device, dtype, qwen_tok.pad_token_id, args.batch_seqs,
                         fast_forward=args.fast_forward,
                         max_batch_tokens=args.max_batch_tokens,
                         seg_buffer=args.seg_buffer)
    scale = fit["scale"]
    welford = Welford(r)  # per-SHARD partial, persisted in meta_<sid>.json
    recs = []
    off = 0
    n_docs = 0
    n_tokens = 0
    n_zero_tokens = 0
    doc_keys_seen = 0
    with open(tmp_int8, "wb") as fout:
        pbar = tqdm(desc=f"shard {sid}", unit="doc")

        def emit(hsh, n, coords):
            nonlocal off, n_docs, n_tokens, n_zero_tokens
            q = quantize(coords, scale)
            fout.write(q.tobytes())
            recs.append((np.uint64(hsh), np.int64(off), np.int32(n)))
            welford.update(coords)
            off += n
            n_docs += 1
            n_tokens += n
            n_zero_tokens += int(np.all(q == 0, axis=1).sum())

        texts = iter_shard_texts(args.climbmix_dir, sid, args.text_column)
        if pool is None:
            # serial: tokenize+align inline (original path)
            seg_iter = (iter_doc_segments(text, enc, qwen_encode,
                                          args.max_doc_tokens, args.max_qwen_tokens)
                        for text in texts)
        else:
            # parallel: worker processes segment docs; imap preserves doc order,
            # so add_doc/emit run in the SAME order as serial -> byte-identical.
            seg_iter = pool.imap_docs(texts)

        last_hb_docs = -2000  # first iteration writes hb at docs=0 ("shard started")
        for hsh, n, segs in seg_iter:
            engine.add_doc(("d", doc_keys_seen), hsh, n, segs)
            doc_keys_seen += 1
            # final=False: let fast-forward accumulate a full seg_buffer before a
            # bucketed flush (serial path flushes here as before). The tail is
            # flushed by the post-loop drain() below.
            for chsh, cn, ccoords in engine.drain(final=False):
                emit(chsh, cn, ccoords)
                pbar.update(1)
            # >= threshold, not exact-multiple: fast-forward emits docs in bursts
            # of thousands per bucketed flush, so n_docs rarely lands EXACTLY on
            # a multiple of 2000 and `% 2000 == 0` would leave the heartbeat
            # stale for a whole shard.
            if hb_path and n_docs - last_hb_docs >= 2000:
                extra = pool.stats() if pool is not None else {}
                _heartbeat(hb_path, sid=sid, docs=n_docs, tokens=n_tokens, **extra)
                last_hb_docs = n_docs
        for chsh, cn, ccoords in engine.drain():
            emit(chsh, cn, ccoords)
            pbar.update(1)
        pbar.close()

    index = np.array(recs, dtype=INDEX_DTYPE) if recs else np.empty(0, dtype=INDEX_DTYPE)
    np.save(tmp_idx, index)  # tmp_idx ends in .npy -> no double suffix
    # atomic publish
    final_int8 = os.path.join(out_shards, f"coords_{sid:05d}.int8")
    final_idx = os.path.join(out_shards, f"index_{sid:05d}.npy")
    os.replace(tmp_int8, final_int8)
    os.replace(tmp_idx, final_idx)
    meta = {"sid": sid, "n_docs": n_docs, "n_tokens": n_tokens,
            "n_zero_tokens": n_zero_tokens,
            "zero_frac": (n_zero_tokens / n_tokens if n_tokens else 0.0),
            "scale": scale,
            "welford": welford.to_dict()}
    with open(os.path.join(out_shards, f"meta_{sid:05d}.json"), "w") as f:
        json.dump(meta, f)
    # done marker last (resumability gate)
    with open(shard_done_marker(args.out, sid), "w") as f:
        f.write(json.dumps(meta))
    print(f"[sweep] shard {sid}: {n_docs} docs, {n_tokens} tokens, "
          f"zero_frac={meta['zero_frac']:.3%}")


def _heartbeat(path, **fields):
    import time
    try:
        with open(path, "w") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), **fields}))
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# MERGE-STATS: fold per-pod Welford partials into corpus mean/std
# --------------------------------------------------------------------------- #
def run_merge_stats(args):
    out_shards = os.path.join(args.out, "shards")
    parts = []
    for p in sorted(glob.glob(os.path.join(out_shards, "meta_*.json"))):
        with open(p) as f:
            m = json.load(f)
        if "welford" in m:
            parts.append(m["welford"])
    if not parts:
        print("[merge-stats] no per-shard Welford partials found in meta_*.json")
        return
    n, mean, std = Welford.merge_dicts(parts)
    out = {"n": int(n), "coord_mean_observed": mean.tolist(), "coord_std_observed": std.tolist()}
    with open(os.path.join(args.out, "corpus_coord_stats.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[merge-stats] merged {len(parts)} shard partials over n={n} tokens -> corpus_coord_stats.json")


# --------------------------------------------------------------------------- #
# ASSEMBLE: per-shard files -> consolidated coords.int8 / index.npy / meta / P
# --------------------------------------------------------------------------- #
def run_assemble(args):
    out_shards = os.path.join(args.out, "shards")
    all_shards = parse_shard_range(args.shards)
    ps = load_probe_meta(args.probe_set)
    concepts, families, pred_order, block, K, legend = resolve_layout(ps, args.layer8_block, args.r_check)
    fit = load_fit(args.out)
    if [str(c) for c in fit["pred_order"]] != [str(c) for c in pred_order]:
        raise ValueError(
            "coord_fit.npz pred_order != probe_set main_block_concepts: the store "
            "was swept against a different probe_set.json than this assemble.")
    r = args.r_check
    P = make_orthonormal_P(args.n_embd, r=r, seed=args.p_seed)
    np.save(os.path.join(args.out, "P.npy"), P.astype(np.float32))

    coords_out = os.path.join(args.out, "coords.int8")
    idx_recs = []
    global_off = 0
    per_shard = {}
    n_docs_total = 0
    n_tokens_total = 0
    n_zero_total = 0
    missing = []
    with open(coords_out + ".tmp", "wb") as fout:
        for sid in all_shards:
            fi = os.path.join(out_shards, f"coords_{sid:05d}.int8")
            xi = os.path.join(out_shards, f"index_{sid:05d}.npy")
            if not (os.path.exists(fi) and os.path.exists(xi)):
                missing.append(sid)
                continue
            arr = np.fromfile(fi, dtype=np.int8).reshape(-1, r)
            index = np.load(xi)
            fout.write(arr.tobytes())
            for h, o, n in zip(index["hash"], index["off"], index["n"]):
                idx_recs.append((np.uint64(h), np.int64(global_off + int(o)), np.int32(n)))
            global_off += arr.shape[0]
            mp = os.path.join(out_shards, f"meta_{sid:05d}.json")
            if os.path.exists(mp):
                with open(mp) as f:
                    m = json.load(f)
                per_shard[str(sid)] = {"n_docs": m["n_docs"], "n_tokens": m["n_tokens"],
                                        "zero_frac": m["zero_frac"]}
                n_docs_total += m["n_docs"]
                n_tokens_total += m["n_tokens"]
                n_zero_total += m.get("n_zero_tokens", 0)
    if missing and not getattr(args, "allow_missing_shards", False):
        os.remove(coords_out + ".tmp")
        raise SystemExit(
            f"[assemble] REFUSING to publish a partial store: {len(missing)} of "
            f"{len(all_shards)} shards missing ({missing[:12]}{'...' if len(missing) > 12 else ''}). "
            f"Docs from missing shards would silently train with zero coords "
            f"(injection no-op). Finish the sweep, or pass --allow-missing-shards "
            f"to publish anyway.")
    os.replace(coords_out + ".tmp", coords_out)
    index = np.array(idx_recs, dtype=INDEX_DTYPE)
    np.save(os.path.join(args.out, "index.npy"), index)

    # duplicate-hash check (content collisions across shards are real duplicate docs)
    _, counts = np.unique(index["hash"], return_counts=True)
    n_dup = int((counts > 1).sum())

    observed = None
    csp = os.path.join(args.out, "corpus_coord_stats.json")
    if os.path.exists(csp):
        with open(csp) as f:
            observed = json.load(f)

    meta = {
        "r": r,
        "scale": fit["scale"],
        "clip_sigma": fit["clip_sigma"],
        "n_embd": args.n_embd,
        "layer8_block": args.layer8_block,
        "block_index": block,
        "block_columns": [block * K, (block + 1) * K],
        "K": K,
        "legend": list(fit["legend"]),
        "families": families,
        "class_order": CYCLIC_ORDER,
        "noncyclic_pca": sorted(NONCYCLIC_PCA),
        "pred_order": list(pred_order),
        "coord_mean": fit["coord_mean"].tolist(),
        "coord_std": fit["coord_std"].tolist(),
        "coord_stats_observed": observed,
        "pca_fit_hash": fit["pca_hash"],
        "encoder_ckpt": os.path.abspath(args.encoder_ckpt) if args.encoder_ckpt else None,
        "P_path": "P.npy",
        "p_seed": args.p_seed,
        "n_docs": n_docs_total,
        "n_tokens": n_tokens_total,
        "n_docs_dup_hash": n_dup,
        "zero_frac": (n_zero_total / n_tokens_total if n_tokens_total else 0.0),
        "per_shard": per_shard,
        "missing_shards": missing,
        "noise_baked": False,
        "noise_note": "sigma=0.15 added by coord_dataloader at train time, keyed by doc hash",
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[assemble] {n_docs_total} docs / {n_tokens_total} tokens over "
          f"{len(all_shards) - len(missing)} shards -> coords.int8 ({global_off} rows), "
          f"index.npy, meta.json, P.npy. zero_frac={meta['zero_frac']:.3%}, dup_hash={n_dup}")
    if missing:
        print(f"[assemble] WARNING: {len(missing)} shards missing from store: {missing}")


# --------------------------------------------------------------------------- #
# VERIFY: recompute K docs live from a finished shard, check int8 round-trip
# --------------------------------------------------------------------------- #
def run_verify(args):
    import torch
    from nanochat.oracle.coords_store import CoordSource
    cs = CoordSource(args.out, noise_sigma=0.0)  # dequant only, no noise
    ps = load_probe_meta(args.probe_set)
    concepts, families, pred_order, block, K, legend = resolve_layout(ps, args.layer8_block, args.r_check)
    fit = load_fit(args.out)
    r = args.r_check
    dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
    model, head, hidden, model_name, K2 = load_encoder(args.encoder_ckpt, args.device, dtype)
    enc = load_nanochat_enc(args.nano_tokenizer_dir)
    qwen_tok = _load_qwen_tok(args, model_name)
    qwen_encode = make_qwen_encode(qwen_tok, add_special=not args.qwen_no_special)
    scale = fit["scale"]

    sid = parse_shard_range(args.shards)[0]
    texts = []
    for t in iter_shard_texts(args.climbmix_dir, sid, args.text_column):
        texts.append(t)
        if len(texts) >= args.verify_docs * 4:
            break
    rng = np.random.default_rng(0)
    picks = rng.choice(len(texts), size=min(args.verify_docs, len(texts)), replace=False)

    engine = CoordEngine(model, head, block, K, concepts, families, fit["pca"], pred_order,
                         r, args.device, dtype, qwen_tok.pad_token_id, args.batch_seqs)
    live = {}
    for j in picks:
        hsh, n, segs = iter_doc_segments(texts[j], enc, qwen_encode,
                                         args.max_doc_tokens, args.max_qwen_tokens)
        engine.add_doc(("v", int(j)), hsh, n, segs)
        for chsh, cn, cc in engine.drain():
            live[int(chsh)] = cc
    for chsh, cn, cc in engine.drain():
        live[int(chsh)] = cc

    max_err = 0.0
    checked = 0
    missing = 0
    zero_tokens = 0
    tot_tokens = 0
    for j in picks:
        text = texts[j]
        hsh = int(doc_hash(text))
        stored, _ = cs.lookup(text, len(enc.encode_ordinary(text)))
        if stored is None:
            missing += 1
            continue
        live_c = live.get(hsh)
        if live_c is None:
            continue
        want = quantize(live_c, scale).astype(np.float32) * scale
        err = float(np.max(np.abs(stored - want))) if stored.size else 0.0
        max_err = max(max_err, err)
        zero_tokens += int(np.all(stored == 0, axis=1).sum())
        tot_tokens += stored.shape[0]
        checked += 1
    print(f"[verify] shard {sid}: checked {checked} docs, missing {missing}, "
          f"int8 round-trip max_abs_err={max_err:.3e} (scale={scale:.3e}), "
          f"zero_frac={zero_tokens / max(tot_tokens, 1):.3%}")
    assert max_err <= scale * 1.0 + 1e-6, \
        f"round-trip error {max_err} exceeds one quant step {scale}"
    print("[verify] OK: stored coords reproduce live recompute within one int8 step.")


# --------------------------------------------------------------------------- #
# PREFLIGHT: consumer-path cross-validation (CPU, no encoder, run BEFORE train)
# --------------------------------------------------------------------------- #
def preflight_check(tok, texts, cs, batch_size=128, num_threads=4):
    """Cross-validate the CONSUMER contract on real docs: tokenize with the
    exact call coord_dataloader makes (`tok.encode(batch, prepend=bos,
    num_threads=...)`, i.e. tiktoken encode_ordinary_batch + BOS insert), take
    n_body = len(t)-1, and require CoordSource.lookup(text, n_body) to hit.

    This is the check that catches a broken producer<->consumer token-count
    contract BEFORE training: a systematic drift (e.g. encode vs
    encode_ordinary special handling, wrong tokenizer.pkl) makes every lookup
    return None -> all-zero coords -> the injected run silently trains as a
    baseline. `tok` is the full nanochat tokenizer (RustBPETokenizer);
    `cs` a CoordSource over the ASSEMBLED store. Returns a stats dict."""
    bos = tok.get_bos_token_id()
    enc = getattr(tok, "enc", None)  # producer-side path, for the direct contract check
    n_docs = n_cov = n_miss = n_empty = n_contract = n_bos_bad = 0
    tok_total = tok_cov = zero_rows = 0
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        toks = tok.encode(batch, prepend=bos, num_threads=num_threads)
        for text, t in zip(batch, toks):
            if len(t) == 0 or t[0] != bos:
                n_bos_bad += 1
                continue
            n_body = len(t) - 1                       # what coord_dataloader computes
            if enc is not None and len(enc.encode_ordinary(text)) != n_body:
                n_contract += 1                       # producer path disagrees with consumer path
            if n_body == 0:
                n_empty += 1
                continue
            n_docs += 1
            tok_total += n_body
            z, _ = cs.lookup(text, n_body)
            if z is None:
                n_miss += 1
            else:
                n_cov += 1
                tok_cov += n_body
                zero_rows += int(np.all(z == 0.0, axis=1).sum())
    return {"n_docs": n_docs, "n_covered": n_cov, "n_missing": n_miss,
            "n_empty": n_empty, "n_contract_mismatch": n_contract,
            "n_bos_bad": n_bos_bad,
            "doc_coverage": (n_cov / n_docs if n_docs else 0.0),
            "token_coverage": (tok_cov / tok_total if tok_total else 0.0),
            "zero_row_frac_covered": (zero_rows / tok_cov if tok_cov else 0.0)}


def run_preflight(args):
    from nanochat.oracle.coords_store import CoordSource
    from nanochat.tokenizer import RustBPETokenizer
    tok = RustBPETokenizer.from_directory(args.nano_tokenizer_dir)
    cs = CoordSource(args.out, noise_sigma=0.0)
    with open(os.path.join(args.out, "meta.json")) as f:
        meta = json.load(f)
    problems = []
    if meta.get("missing_shards"):
        problems.append(f"meta.json missing_shards={meta['missing_shards']}")
    shards = parse_shard_range(args.shards)
    per_shard = max(1, args.preflight_docs // len(shards))
    texts = []
    for sid in shards:
        got = 0
        for t in iter_shard_texts(args.climbmix_dir, sid, args.text_column):
            texts.append(t)
            got += 1
            if got >= per_shard:
                break
    res = preflight_check(tok, texts, cs)
    print(f"[preflight] {len(texts)} docs from {len(shards)} shards: "
          f"doc_coverage={res['doc_coverage']:.4%} token_coverage={res['token_coverage']:.4%} "
          f"missing={res['n_missing']} empty={res['n_empty']} "
          f"contract_mismatch={res['n_contract_mismatch']} bos_bad={res['n_bos_bad']} "
          f"zero_row_frac(covered)={res['zero_row_frac_covered']:.3%}")
    if res["n_contract_mismatch"] or res["n_bos_bad"]:
        problems.append(
            f"{res['n_contract_mismatch']} producer/consumer token-count mismatches, "
            f"{res['n_bos_bad']} bad-BOS docs: the tokenizer contract is BROKEN "
            f"(wrong tokenizer.pkl or encode semantics drift)")
    if res["token_coverage"] < args.preflight_min_coverage:
        problems.append(
            f"token_coverage {res['token_coverage']:.4%} < required "
            f"{args.preflight_min_coverage:.4%}: docs would fall back to zero "
            f"coords (silent baseline)")
    if problems:
        raise SystemExit("[preflight] FAIL:\n  - " + "\n  - ".join(problems))
    print("[preflight] OK: store covers the consumer token path -- safe to launch.")
    return res


# --------------------------------------------------------------------------- #
# MEASURE-CROSSING: prefix-mode crossing rate for the qwen<->nanochat pair
# --------------------------------------------------------------------------- #
def run_measure_crossing(args):
    enc = load_nanochat_enc(args.nano_tokenizer_dir)
    from transformers import AutoModel  # noqa: F401  (ensure transformers importable path)
    qwen_tok = _load_qwen_tok(args, args.qwen_model or "Qwen/Qwen3-0.6B-Base")
    qwen_encode = make_qwen_encode(qwen_tok, add_special=not args.qwen_no_special)
    shards = parse_shard_range(args.shards)
    rates = []
    n = 0
    for sid in shards:
        for text in iter_shard_texts(args.climbmix_dir, sid, args.text_column):
            nano_ids = enc.encode_ordinary(text)
            if not nano_ids:
                continue
            nano_off = nanochat_char_offsets(enc, nano_ids, text)
            q_ids, q_off = qwen_encode(text)
            # NOTE: source = nanochat (we align nano tokens onto qwen anchors).
            rates.append(crossing_rate(text, nano_off, q_off))
            n += 1
            if n >= args.crossing_docs:
                break
        if n >= args.crossing_docs:
            break
    if not rates:
        print("[measure-crossing] no docs processed")
        return
    print(f"[measure-crossing] nanochat->qwen prefix-mode crossing over {n} docs: "
          f"mean={np.mean(rates):.4%} median={np.median(rates):.4%} "
          f"p90={np.percentile(rates, 90):.4%}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="sweep",
                    choices=["fit", "sweep", "merge-stats", "assemble", "verify",
                             "preflight", "measure-crossing"])
    ap.add_argument("--encoder-ckpt", help="expA best.pt (Qwen full-FT + 1024->162 head)")
    ap.add_argument("--probe-set", help="probe_set.json file or its dir")
    ap.add_argument("--shards", default="0-190", help="e.g. 0-190 or 0-3,10")
    ap.add_argument("--out", required=True, help="coord store dir")
    ap.add_argument("--climbmix-dir", default=None,
                    help="dir with shard_<sid>.parquet from karpathy/climbmix-400b-shuffle "
                         "(default $NANOCHAT_BASE_DIR/base_data_climbmix)")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--nano-tokenizer-dir", default=None,
                    help="dir with tokenizer.pkl (baseline noVE tokenizer); default "
                         "$NANOCHAT_BASE_DIR/tokenizer")
    ap.add_argument("--qwen-model", default=None,
                    help="override encoder tokenizer name (default = ckpt model_name)")
    ap.add_argument("--qwen-no-special", action="store_true",
                    help="tokenize qwen WITHOUT special tokens (default adds them, "
                         "matching train_encoder.process_doc)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--pod-index", type=int, default=0)
    ap.add_argument("--n-pods", type=int, default=1)
    ap.add_argument("--layer8-block", type=int, default=8,
                    help="gemma layer whose K predicted cols build the coords (block=layers.index)")
    ap.add_argument("--n-embd", type=int, default=1536)
    ap.add_argument("--r-check", type=int, default=14)
    ap.add_argument("--p-seed", type=int, default=1337)
    ap.add_argument("--max-doc-tokens", type=int, default=2048,
                    help="nano-token window size (docs longer than this are windowed)")
    ap.add_argument("--max-qwen-tokens", type=int, default=4096,
                    help="hard cap on qwen tokens per window (clip beyond)")
    ap.add_argument("--batch-seqs", type=int, default=32,
                    help="qwen segments per padded forward (serial path)")
    ap.add_argument("--fast-forward", action="store_true",
                    help="GPU-forward throughput: length-bucketed cross-doc batching. "
                         "Buffer --seg-buffer segments across docs, sort by qwen length, "
                         "and pack each forward to a --max-batch-tokens budget (kills "
                         "ragged-padding waste + latency-bound small forwards). Output "
                         "equals the serial path within one int8 step (bf16 fp-noise from "
                         "batch shape); zero-fallback (unmapped -> 0) is bit-identical. "
                         "Default off = original --batch-seqs FIFO path.")
    ap.add_argument("--max-batch-tokens", type=int, default=32768,
                    help="--fast-forward: max padded tokens (B*maxlen) per forward.")
    ap.add_argument("--seg-buffer", type=int, default=4096,
                    help="--fast-forward: segments buffered across docs before a "
                         "length-bucketed flush (bounds in-flight doc memory).")
    ap.add_argument("--clip-sigma", type=float, default=6.0,
                    help="int8 clip at +-clip_sigma*max(coord_std)")
    ap.add_argument("--fit-tokens", type=int, default=2_000_000,
                    help="pred rows collected for the PCA + coord-scale fit")
    ap.add_argument("--noise-none", action="store_true",
                    help="no-op (noise is added by the loader, never baked here)")
    ap.add_argument("--verify-docs", type=int, default=64)
    ap.add_argument("--crossing-docs", type=int, default=2000)
    ap.add_argument("--preflight-docs", type=int, default=512,
                    help="docs sampled across --shards for the consumer-path check")
    ap.add_argument("--preflight-min-coverage", type=float, default=0.999,
                    help="minimum token coverage required by --mode preflight")
    ap.add_argument("--allow-missing-shards", action="store_true",
                    help="let assemble publish a partial store (missing shards' "
                         "docs fall back to zero coords at train time)")
    ap.add_argument("--heartbeat-path", default=None)
    ap.add_argument("--feeder-workers", type=int, default=0,
                    help="CPU feeder pool size for --mode sweep: N worker PROCESSES "
                         "run per-doc tokenize+offsets+align in parallel, feeding a "
                         "bounded queue the main process drains for GPU forwards + "
                         "writes. 0 (default) = original inline serial path. Output "
                         "is byte-identical regardless of N (imap preserves doc order).")
    ap.add_argument("--feeder-prefetch", type=int, default=0,
                    help="max in-flight docs across the feeder pool (backpressure/"
                         "reorder-buffer bound); 0 = auto (max(4*workers, 64)).")
    return ap


def main():
    args = build_argparser().parse_args()
    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"
    if args.climbmix_dir is None:
        args.climbmix_dir = default_climbmix_dir()
    if args.nano_tokenizer_dir is None:
        base = os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat"))
        args.nano_tokenizer_dir = os.path.join(base, "tokenizer")
    os.makedirs(args.out, exist_ok=True)

    if args.mode == "fit":
        _require(args, ["encoder_ckpt", "probe_set"])
        run_fit(args)
    elif args.mode == "sweep":
        _require(args, ["encoder_ckpt", "probe_set"])
        run_sweep(args)
    elif args.mode == "merge-stats":
        run_merge_stats(args)
    elif args.mode == "assemble":
        _require(args, ["probe_set"])
        run_assemble(args)
    elif args.mode == "verify":
        _require(args, ["encoder_ckpt", "probe_set"])
        run_verify(args)
    elif args.mode == "preflight":
        run_preflight(args)
    elif args.mode == "measure-crossing":
        run_measure_crossing(args)


def _require(args, names):
    missing = [n for n in names if getattr(args, n) is None]
    if missing:
        raise SystemExit(f"--mode {args.mode} requires: {', '.join('--' + m.replace('_','-') for m in missing)}")


if __name__ == "__main__":
    main()
