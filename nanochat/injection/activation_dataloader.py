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
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from nanochat.dataloader import _document_batches


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
    (docs buffered ahead of the packer at yield time = prefetch depth proxy), and
    ``batches``. injection_train reads these for the startup throughput verdict and
    the live starvation monitor. No overhead when ``stats is None``."""
    assert split in ["train", "val"]
    assert len(sources) > 0, "need at least one activation source"
    names = list(sources.keys())
    rs = {name: int(sources[name].r) for name in names}
    pos_names = [n for n in names if getattr(sources[n], "positional", False)
                 and hasattr(sources[n], "lookup_by_row")]
    resolve_pos = _shard_row_resolver(split) if pos_names else None
    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos = tokenizer.get_bos_token_id()

    tok_buffer = []                            # token ids per doc (incl. BOS)
    act_buffer = {name: [] for name in names}  # per source: (n_doc_tokens+1, r) arrays, BOS row = 0
    pq_idx = rg_idx = 0
    epoch = 1
    cursor = {"key": None, "sid": -1, "row": 0}   # within-shard row cursor for the positional join
    pool = ThreadPoolExecutor(max_workers=lookup_workers) if lookup_workers > 0 else None

    def one_doc(text, t, sid, abs_row):
        """Per-doc activation rows for every source (BOS row prepended = 0).
        Unknown doc (None) -> EXACT zeros with NO noise (ActivationSource
        contract — noised zeros would inject full-gate noise on docs we know
        nothing about; exact zeros keep the site a strict no-op)."""
        n_body = len(t) - 1
        out = {}
        for name in names:
            src = sources[name]
            z, key = (src.lookup_by_row(sid, abs_row, text, n_body) if name in pos_names
                      else src.lookup(text, n_body))
            z = np.zeros((n_body, rs[name]), np.float32) if z is None else src.add_noise(z, key)
            out[name] = np.concatenate([np.zeros((1, rs[name]), np.float32), z], axis=0)
        return t, out

    def refill():
        nonlocal pq_idx, rg_idx, epoch
        t_ref = time.time() if stats is not None else 0.0
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        toks = tokenizer.encode(doc_batch, prepend=bos, num_threads=tokenizer_threads)
        tasks = []
        for text, t in zip(doc_batch, toks):
            if pos_names:
                # Assign (shard, row) serially (all chunks of one row group carry
                # the same (pq,rg,epoch) consecutively, so a per-run cursor is
                # exact; epoch resets it when a row group is re-read next pass).
                # The heavy per-doc lookup is order-independent and may run pooled.
                key = (pq_idx, rg_idx, epoch)
                if key != cursor["key"]:
                    cursor["key"] = key
                    cursor["sid"], cursor["row"] = resolve_pos(pq_idx, rg_idx)
                sid, abs_row = cursor["sid"], cursor["row"]
                cursor["row"] += 1
            else:
                sid, abs_row = -1, -1
            tasks.append((text, t, sid, abs_row))
        results = (pool.map(lambda a: one_doc(*a), tasks) if pool is not None
                   else (one_doc(*a) for a in tasks))
        for t, per_src in results:             # ordered: preserves doc order
            tok_buffer.append(t)
            for name in names:
                act_buffer[name].append(per_src[name])
        if stats is not None:
            stats["produce_seconds"] = stats.get("produce_seconds", 0.0) + (time.time() - t_ref)
            stats["produced_docs"] = stats.get("produced_docs", 0) + len(doc_batch)
            stats["produced_tokens"] = stats.get("produced_tokens", 0) + sum(len(t) - 1 for t in toks)

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
                    refill()
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
            stats["queue_depth"] = len(tok_buffer)   # docs buffered ahead of the packer
            stats["batches"] = stats.get("batches", 0) + 1
        yield inputs, targets, acts, {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
