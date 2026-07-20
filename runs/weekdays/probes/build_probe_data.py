#!/usr/bin/env python3
"""Phase 1 (pod, once): cache (nanochat body ids, aligned+thresholded weekday acts)
per doc from a HELD-OUT scored shard — the exact pairs training injected (prescored
path, no gemma model) — into probes/probe_data.pt, shared by every arm's trainer.
Run from repo root: python runs/weekdays/probes/build_probe_data.py --max-docs 2500
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "runs", "lib"))

import concepts as concept_registry  # noqa: E402


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=100,
                    help="held-out scored climbmix shard (>=45 is untouched by training)")
    ap.add_argument("--max-docs", type=int, default=2500)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="per-doc cap incl. BOS (matches val-bpb)")
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--gemma-model", default="google/gemma-2-2b")
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "climbmix_cache"))
    ap.add_argument("--out", default=os.path.join(HERE, "probe_data.pt"))
    args = ap.parse_args()

    RE = _load_by_path("_run_evals_probe", os.path.join(REPO, "runs/lib/eval/run_evals.py"))

    from nanochat.tokenizer import get_tokenizer
    from nanochat.injection.sources import _default_gemma_encode
    tok = get_tokenizer()
    nano_enc = tok.enc
    bos_id = tok.get_bos_token_id()
    gemma_encode = _default_gemma_encode(args.gemma_model)

    sid = args.shard
    shard_path = RE._download_climbmix_shard(sid, args.cache_dir)
    src = RE._make_prescored_source(sid, nano_enc, gemma_encode, args.layer, args.threshold,
                                    os.path.join(args.cache_dir, f"scored_{sid:05d}"))

    docs, n_tok, n_act, n_miss = [], 0, 0, 0
    for row, text in RE._iter_shard_texts(shard_path, args.max_docs):
        if not text:
            continue
        body = nano_enc.encode_ordinary(text)
        if not body:
            continue
        acts, _ = src.lookup_by_row(sid, row, text, len(body))
        if acts is None:                      # store miss / tokenizer drift
            n_miss += 1
            continue                          # probes need real targets; skip
        cap = args.max_tokens - 1             # body cap (BOS added at batch time)
        body = np.asarray(body[:cap], np.int32)
        acts = np.asarray(acts, np.float32)[:cap]
        docs.append({"ids": body, "acts": acts, "doc_idx": len(docs),
                     "shard": sid, "row": row})
        n_tok += len(body)
        n_act += int((acts != 0).any(1).sum())

    meta = {"shard": sid, "n_docs": len(docs), "n_tokens": n_tok,
            "n_active_tokens": n_act, "active_frac": n_act / max(n_tok, 1),
            "n_store_miss": n_miss, "bos_id": int(bos_id),
            "threshold": args.threshold, "layer": args.layer,
            "max_tokens": args.max_tokens, "align_policy": src.align_policy,
            "concepts": list(concept_registry.get_family("weekdays").store_order)}
    torch.save({"docs": docs, "meta": meta}, args.out)
    print(f"[build] {len(docs)} docs, {n_tok} tokens ({n_act} active, "
          f"{meta['active_frac']:.3f}), {n_miss} store-miss -> {args.out}")


if __name__ == "__main__":
    main()
