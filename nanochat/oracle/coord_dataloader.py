"""Ride-along coord dataloader: nanochat's BOS-aligned best-fit packing, with
parallel (B, T, r) activation tensors carried in lockstep with the tokens.

Mirrors ``nanochat.dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit``
1:1 for the token path (so token order / crop / DDP sharding are byte-identical
to the baseline given the same shard set + seed policy), and places each doc's
precomputed activation rows wherever that doc's tokens go -- same best-fit pick,
same crop. Activations for the BOS token (and any doc missing from the
precompute) are zero, so the injection is a no-op there.

Two entry points:
  * ``acts_data_loader_with_state(tokenizer, sources, ...)`` -- the general
    multi-site form. ``sources`` is a dict name -> activation source (anything
    with ``.r``, ``.lookup(text, n_tokens) -> (arr|None, key)`` and
    ``.add_noise(arr, key)``: ``coords_store.CoordSource``,
    ``injections.FnActivation``, ...). Yields
    ``(inputs, targets, acts, state_dict)`` with ``acts`` a dict
    name -> (B, T, r_name) float32 on ``device`` -- exactly what
    ``GPT.forward(acts=...)`` consumes.
  * ``coord_data_loader_with_state(tokenizer, coord_source, ...)`` -- the
    original single-source form, a thin wrapper yielding the bare
    (B, T, r) tensor.

Yields (inputs, targets, acts/coords, state_dict):
  inputs/targets : (B, T) long   -- identical to the stock loader
  acts           : per-source (B, T, r) float32 on `device` -- standardized
                   activations + per-source noise
"""
import numpy as np
import torch

from nanochat.dataloader import _document_batches


def acts_data_loader_with_state(
    tokenizer, sources, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None, buffer_size=1000,
):
    assert split in ["train", "val"]
    assert len(sources) > 0, "need at least one activation source"
    names = list(sources.keys())
    rs = {name: int(sources[name].r) for name in names}
    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos = tokenizer.get_bos_token_id()

    tok_buffer = []                        # list[list[int]]  token ids per doc (incl. BOS)
    act_buffer = {name: [] for name in names}  # per source: list[(n_doc_tokens+1, r)] arrays, BOS row = 0
    pq_idx = rg_idx = 0
    epoch = 1

    def refill():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        toks = tokenizer.encode(doc_batch, prepend=bos, num_threads=tokenizer_threads)
        for text, t in zip(doc_batch, toks):
            n_body = len(t) - 1                        # minus prepended BOS
            for name in names:
                src = sources[name]
                r = rs[name]
                z, key = src.lookup(text, n_body)      # (n_body, r) or None
                if z is None:
                    # doc missing from precompute (or token-count drift): EXACT zeros,
                    # NO noise -- the injection site renormalizes any nonzero coord to
                    # full gate amplitude, so noised zeros would inject pure noise.
                    # Exact zeros make the injection a strict no-op for this doc.
                    z = np.zeros((n_body, r), np.float32)
                else:
                    z = src.add_noise(z, key)          # deterministic per doc content
                z = np.concatenate([np.zeros((1, r), np.float32), z], axis=0)  # BOS row = 0
                act_buffer[name].append(z)
            tok_buffer.append(t)

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
        yield inputs, targets, acts, {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}


def coord_data_loader_with_state(tokenizer, coord_source, B, T, split, **kwargs):
    """Single-source form: yields (inputs, targets, (B,T,r) coords, state_dict)."""
    it = acts_data_loader_with_state(tokenizer, {"coords": coord_source}, B, T, split, **kwargs)
    for inp, tgt, acts, st in it:
        yield inp, tgt, acts["coords"], st


def coord_data_loader(*args, **kwargs):
    for inp, tgt, crd, _ in coord_data_loader_with_state(*args, **kwargs):
        yield inp, tgt, crd
