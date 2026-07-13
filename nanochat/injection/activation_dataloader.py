"""Ride-along activation dataloader: nanochat's BOS-aligned best-fit packing
with parallel (B, T, r) activation tensors carried in lockstep with the tokens.
Mirrors ``nanochat.dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit``
1:1 on the token path (same best-fit pick, same crop, same DDP sharding), and
places each doc's activation rows wherever that doc's tokens go.

``sources`` is a dict name -> ActivationSource. Yields
``(inputs, targets, acts, state_dict)`` with ``acts`` a dict
name -> (B, T, r_name) float32 on ``device`` — what ``GPT.forward(acts=...)``
consumes.

Positional sources (``.positional`` + ``.lookup_by_row``) join by (shard, row):
this loader re-runs the real corpus enumeration, so it knows each doc's shard and
absolute row and joins the score store positionally (no hashing, no startup walk).
Other sources join by doc text (``.lookup``). ``lookup_workers>0`` runs the
per-doc lookups (gemma tokenize + align) in an ordered thread pool so the CPU
scoring overlaps training; the row cursor is assigned serially first, so the
parallel work stays order-independent and deterministic.

Production (tokenize + per-doc lookup/align) is factored into ``_ChunkProducer``
and the best-fit packing into ``_pack_batches`` so both the synchronous default
loader and the buffered loader (``BufferControl``, video-player prefill/rebuffer)
run the identical packing over the identical chunk stream.
"""
import collections
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from nanochat.dataloader import _document_batches

_Chunk = collections.namedtuple("_Chunk", ["docs", "state", "body_tokens"])


def _shard_row_resolver(split):
    """(pq_idx, rg_idx) -> (climbmix shard id, row index of that row group's first
    doc). Reads only parquet row-group metadata (cheap), cached per file."""
    from nanochat.dataset import list_parquet_files
    import pyarrow.parquet as pq
    paths = list_parquet_files()
    paths = paths[:-1] if split == "train" else paths[-1:]  # same train/val split as _document_batches
    cache = {}

    def resolve(pq_idx, rg_idx):
        p = paths[pq_idx]
        sid = int(os.path.basename(p).split("_")[1].split(".")[0])
        off = cache.get(pq_idx)
        if off is None:
            md = pq.ParquetFile(p).metadata
            off = np.concatenate([[0], np.cumsum([md.row_group(i).num_rows
                                                  for i in range(md.num_row_groups)])]).astype(np.int64)
            cache[pq_idx] = off
        return sid, int(off[rg_idx])

    return resolve


class _ChunkProducer:
    """Turns the corpus enumeration into activation chunks: one ``next_chunk()``
    == one ``_document_batches`` chunk tokenized + scored (BOS row prepended = 0),
    the unit both loaders buffer/pack. Stateful (corpus cursor, positional
    (shard,row) cursor, lookup pool); ``next_chunk`` is called serially (the
    default loader inline, the buffered loader from its single producer thread)."""

    def __init__(self, tokenizer, sources, split, resume_state_dict,
                 tokenizer_threads, tokenizer_batch_size, lookup_workers, stats):
        self.tokenizer = tokenizer
        self.sources = sources
        self.names = list(sources.keys())
        self.rs = {name: int(sources[name].r) for name in self.names}
        self.pos_names = [n for n in self.names if getattr(sources[n], "positional", False)
                          and hasattr(sources[n], "lookup_by_row")]
        self.resolve_pos = _shard_row_resolver(split) if self.pos_names else None
        self.tokenizer_threads = tokenizer_threads
        self.stats = stats
        self.bos = tokenizer.get_bos_token_id()
        self.batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
        self.cursor = {"key": None, "sid": -1, "row": 0}   # within-shard row cursor for the positional join
        self.pool = ThreadPoolExecutor(max_workers=lookup_workers) if lookup_workers > 0 else None

    def _one_doc(self, text, t, sid, abs_row):
        """Per-doc activation rows for every source (BOS row prepended = 0).
        Unknown doc (None) -> EXACT zeros with NO noise, and exact-zero rows
        inside a known doc (unmapped/concept-free tokens) stay exactly zero
        through the noise (ActivationSource contract — the site renormalizes
        any nonzero row to full gate amplitude, so a noised zero row would
        inject pure noise at full strength; exact zeros keep the site a
        strict no-op there)."""
        n_body = len(t) - 1
        out = {}
        for name in self.names:
            src = self.sources[name]
            z, key = (src.lookup_by_row(sid, abs_row, text, n_body) if name in self.pos_names
                      else src.lookup(text, n_body))
            if z is None:
                z = np.zeros((n_body, self.rs[name]), np.float32)
            else:
                zero_rows = ~np.any(z != 0.0, axis=1)
                z = src.add_noise(z, key)
                if zero_rows.any():
                    z[zero_rows] = 0.0
            out[name] = np.concatenate([np.zeros((1, self.rs[name]), np.float32), z], axis=0)
        return t, out

    def next_chunk(self):
        t_ref = time.time() if self.stats is not None else 0.0
        doc_batch, (pq_idx, rg_idx, epoch) = next(self.batches)
        toks = self.tokenizer.encode(doc_batch, prepend=self.bos, num_threads=self.tokenizer_threads)
        tasks = []
        for text, t in zip(doc_batch, toks):
            if self.pos_names:
                # Assign (shard, row) serially (all chunks of one row group carry
                # the same (pq,rg,epoch) consecutively, so a per-run cursor is
                # exact; epoch resets it when a row group is re-read next pass).
                # The heavy per-doc lookup is order-independent and may run pooled.
                key = (pq_idx, rg_idx, epoch)
                if key != self.cursor["key"]:
                    self.cursor["key"] = key
                    self.cursor["sid"], self.cursor["row"] = self.resolve_pos(pq_idx, rg_idx)
                sid, abs_row = self.cursor["sid"], self.cursor["row"]
                self.cursor["row"] += 1
            else:
                sid, abs_row = -1, -1
            tasks.append((text, t, sid, abs_row))
        results = (self.pool.map(lambda a: self._one_doc(*a), tasks) if self.pool is not None
                   else (self._one_doc(*a) for a in tasks))
        docs = list(results)                       # ordered: preserves doc order
        body_tokens = sum(len(t) - 1 for t in toks)
        if self.stats is not None:
            self.stats["produce_seconds"] = self.stats.get("produce_seconds", 0.0) + (time.time() - t_ref)
            self.stats["produced_docs"] = self.stats.get("produced_docs", 0) + len(doc_batch)
            self.stats["produced_tokens"] = self.stats.get("produced_tokens", 0) + body_tokens
        return _Chunk(docs, (pq_idx, rg_idx, epoch), body_tokens)


def _pack_batches(pull, names, rs, B, T, device, stats, buffer_size):
    """The best-fit packer, byte-identical to the stock loader: pulls chunks
    (``pull() -> _Chunk``) until ``buffer_size`` docs are staged, then packs
    B rows of T+1 and yields ``(inputs, targets, acts, state)``. ``pull`` is
    the only thing that differs between the two loaders (synchronous next_chunk
    vs a blocking draw from the buffer)."""
    row_capacity = T + 1
    tok_buffer = []                            # token ids per doc (incl. BOS)
    act_buffer = {name: [] for name in names}  # per source: (n_doc_tokens+1, r) arrays, BOS row = 0
    cur = (0, 0, 1)                            # (pq_idx, rg_idx, epoch) of the last pulled chunk

    use_cuda = device == "cuda"
    row_tok = torch.empty((B, row_capacity), dtype=torch.long)
    row_act = {name: torch.empty((B, row_capacity, rs[name]), dtype=torch.float32)
               for name in names}

    cpu_tok = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    cpu_act = {name: torch.empty(B * T * rs[name], dtype=torch.float32, pin_memory=use_cuda)
               for name in names}
    gpu_tok = torch.empty(2 * B * T, dtype=torch.long, device=device)
    gpu_act = {name: torch.empty(B * T * rs[name], dtype=torch.float32, device=device)
               for name in names}
    inputs = gpu_tok[:B * T].view(B, T)
    targets = gpu_tok[B * T:].view(B, T)
    acts = {name: gpu_act[name].view(B, T, rs[name]) for name in names}

    while True:
        for row in range(B):
            pos = 0
            while pos < row_capacity:
                while len(tok_buffer) < buffer_size:
                    chunk = pull()
                    cur = chunk.state
                    for t, per_src in chunk.docs:
                        tok_buffer.append(t)
                        for name in names:
                            act_buffer[name].append(per_src[name])
                remaining = row_capacity - pos
                # best fit: largest doc that fits entirely (same rule as stock loader)
                best_i, best_len = -1, 0
                for i, d in enumerate(tok_buffer):
                    dl = len(d)
                    if best_len < dl <= remaining:
                        best_i, best_len = i, dl
                if best_i >= 0:
                    d = tok_buffer.pop(best_i)
                    dl = len(d)
                    row_tok[row, pos:pos + dl] = torch.from_numpy(np.asarray(d))
                    for name in names:
                        z = act_buffer[name].pop(best_i)
                        row_act[name][row, pos:pos + dl] = torch.from_numpy(z)
                    pos += dl
                else:
                    # crop shortest to fill exactly (identical crop for tokens+acts)
                    si = min(range(len(tok_buffer)), key=lambda i: len(tok_buffer[i]))
                    d = tok_buffer.pop(si)
                    row_tok[row, pos:pos + remaining] = torch.from_numpy(np.asarray(d[:remaining]))
                    for name in names:
                        z = act_buffer[name].pop(si)
                        row_act[name][row, pos:pos + remaining] = torch.from_numpy(z[:remaining])
                    pos += remaining

        cpu_tok[:B * T].view(B, T).copy_(row_tok[:, :-1])
        cpu_tok[B * T:].view(B, T).copy_(row_tok[:, 1:])
        gpu_tok.copy_(cpu_tok, non_blocking=use_cuda)
        for name in names:
            cpu_act[name].view(B, T, rs[name]).copy_(row_act[name][:, :-1])  # acts align to INPUTS
            gpu_act[name].copy_(cpu_act[name], non_blocking=use_cuda)
        if stats is not None:
            stats["queue_depth"] = len(tok_buffer)   # docs staged ahead of the packer
            stats["batches"] = stats.get("batches", 0) + 1
        yield inputs, targets, acts, {"pq_idx": cur[0], "rg_idx": cur[1], "epoch": cur[2]}


def acts_data_loader_with_state(
    tokenizer, sources, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None, buffer_size=1000, lookup_workers=0,
    stats=None,
):
    """``stats`` (optional dict) is updated in place each yield with cheap
    staying-ahead counters (Amendment 2): ``produce_seconds`` (cumulative wall
    time spent producing = tokenize+lookup+align, the source's own cost),
    ``produced_docs``/``produced_tokens`` (throughput numerator), ``queue_depth``
    (docs staged ahead of the packer at yield time = prefetch depth proxy), and
    ``batches``. injection_train reads these for the duty-cycle forecast and the
    live starvation monitor. No overhead when ``stats is None``. Synchronous:
    each ``next()`` produces exactly what it packs (see BufferControl for the
    video-player buffered variant)."""
    assert split in ["train", "val"]
    assert len(sources) > 0, "need at least one activation source"
    names = list(sources.keys())
    rs = {name: int(sources[name].r) for name in names}
    producer = _ChunkProducer(tokenizer, sources, split, resume_state_dict,
                              tokenizer_threads, tokenizer_batch_size, lookup_workers, stats)
    yield from _pack_batches(producer.next_chunk, names, rs, B, T, device, stats, buffer_size)


class BufferControl:
    """Video-player buffer over ``_ChunkProducer``: a single background thread
    runs ``next_chunk`` into a token-denominated deque; the packer draws chunks
    from it via ``pull``. Prefill fills to ``prefill_tokens`` before the first
    batch; a dry buffer triggers a rebuffer to ``rebuffer_tokens`` (the DDP-
    coordinated pause is driven externally by injection_train, which calls
    ``do_rebuffer``). Backpressure: production blocks once the buffer reaches
    ``max_tokens`` (never buffer the whole "video"). ``stats`` gains a live
    ``buffer_tokens`` reading alongside the staying-ahead counters."""

    def __init__(self, producer, prefill_tokens, rebuffer_tokens, dry_tokens,
                 max_tokens, stats=None):
        self.producer = producer
        self.prefill_tokens = int(prefill_tokens)
        self.rebuffer_tokens = int(rebuffer_tokens)
        self.dry_tokens = int(dry_tokens)
        self.max_tokens = int(max_tokens)
        self.stats = stats
        self._chunks = collections.deque()
        self._buffer_tokens = 0
        self._cv = threading.Condition()
        self._thread = None
        self._stop = False
        self._error = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name="acts-producer", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    def _set_depth(self, delta):
        self._buffer_tokens += delta
        if self.stats is not None:
            self.stats["buffer_tokens"] = self._buffer_tokens

    def _run(self):
        try:
            while True:
                with self._cv:
                    while not self._stop and self._buffer_tokens >= self.max_tokens:
                        self._cv.wait()          # backpressure: buffer is full
                    if self._stop:
                        return
                chunk = self.producer.next_chunk()   # heavy work OUTSIDE the lock
                with self._cv:
                    self._chunks.append(chunk)
                    self._set_depth(chunk.body_tokens)
                    self._cv.notify_all()
        except BaseException as e:                # surface producer failure to the consumer
            with self._cv:
                self._error = e
                self._cv.notify_all()

    def pull(self):
        with self._cv:
            while not self._chunks and self._error is None:
                self._cv.wait()
            if not self._chunks and self._error is not None:
                raise self._error
            chunk = self._chunks.popleft()
            self._set_depth(-chunk.body_tokens)
            self._cv.notify_all()
        return chunk

    def buffer_tokens(self):
        with self._cv:
            return self._buffer_tokens

    def buffer_low(self):
        """Below the dry mark: the buffer ran dry and a rebuffer is warranted."""
        with self._cv:
            return self._buffer_tokens < self.dry_tokens

    def produce_rate(self):
        """Measured tokens/s of production so far (0 until the first chunk)."""
        if self.stats is None:
            return 0.0
        ps = self.stats.get("produce_seconds", 0.0)
        return (self.stats.get("produced_tokens", 0) / ps) if ps > 0 else 0.0

    def _wait_target(self, target, progress_cb, interval):
        """Block until the buffer reaches ``target`` tokens (or the producer dies
        / is stopped), calling ``progress_cb(depth, target, rate)`` at most every
        ``interval`` seconds. Returns immediately if already at target."""
        last = 0.0
        while True:
            with self._cv:
                if self._error is not None:
                    raise self._error
                depth = self._buffer_tokens
                if depth >= target or self._stop:
                    return
                self._cv.wait(timeout=interval)
                depth = self._buffer_tokens
            now = time.time()
            if progress_cb is not None and now - last >= interval:
                progress_cb(depth, target, self.produce_rate())
                last = now

    def wait_prefill(self, progress_cb=None, interval=2.0):
        self._wait_target(self.prefill_tokens, progress_cb, interval)

    def do_rebuffer(self, progress_cb=None, interval=3.0):
        self._wait_target(self.rebuffer_tokens, progress_cb, interval)


def acts_data_loader_buffered(
    tokenizer, sources, B, T, split, *,
    prefill_tokens, rebuffer_tokens, dry_tokens, max_tokens,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None, buffer_size=1000, lookup_workers=0,
    stats=None,
):
    """Buffered variant: returns ``(control, generator)``. The producer thread
    starts immediately (production begins during setup / prefill). The caller
    ``control.wait_prefill(...)`` before the first ``next(generator)``, then
    coordinates rebuffers via ``control.buffer_low()`` / ``control.do_rebuffer``.
    Packing is byte-identical to ``acts_data_loader_with_state`` (same chunks,
    same best-fit)."""
    assert split in ["train", "val"]
    assert len(sources) > 0, "need at least one activation source"
    names = list(sources.keys())
    rs = {name: int(sources[name].r) for name in names}
    producer = _ChunkProducer(tokenizer, sources, split, resume_state_dict,
                              tokenizer_threads, tokenizer_batch_size, lookup_workers, stats)
    control = BufferControl(producer, prefill_tokens, rebuffer_tokens, dry_tokens,
                            max_tokens, stats).start()
    gen = _pack_batches(control.pull, names, rs, B, T, device, stats, buffer_size)
    return control, gen
