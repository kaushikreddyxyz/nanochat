"""Compute a family's empirical concept z-vectors (median r-channel z over tokens where
concept Y fires: max channel z >= present_z AND Y is argmax) from a climbmix-scored shard
via HTTP-Range .npy reads; dequant/standardize identical to the training source. Writes
the family's empirical_patterns.json (causal.py/open_gen.py "empirical" conditions).
Run: python runs/lib/eval/empirical_patterns.py --family weekdays --out <family>/empirical_patterns.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os

import sys

import numpy as np
import requests
from huggingface_hub import hf_hub_download, hf_hub_url

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402  (family_concepts: the one channel-order source)

DEFAULT_REPO = "kaushikreddyxyz/climbmix-scored"


# -- ranged .npy reads (from attribution/examples/read_corpus_scores.py) -------
def http_range(url: str, start: int, end: int) -> bytes:
    r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(10, 300))
    r.raise_for_status()
    data = r.content
    assert len(data) == end - start + 1, f"server ignored Range ({len(data)} B) {url}"
    return data


def npy_header_ranged(url: str):
    head = http_range(url, 0, 11)
    assert head[:6] == b"\x93NUMPY", f"not an npy file: {url}"
    if head[6] == 1:
        hlen, off = int.from_bytes(head[8:10], "little"), 10
    else:
        hlen, off = int.from_bytes(head[8:12], "little"), 12
    header = ast.literal_eval(http_range(url, off, off + hlen - 1).decode("latin1"))
    assert not header["fortran_order"], "ranged row reads need C order"
    return np.dtype(header["descr"]), header["shape"], off + hlen


def read_npy_rows_ranged(url: str, n_rows: int):
    dtype, shape, data_off = npy_header_ranged(url)
    n_rows = min(n_rows, shape[0])
    row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
    buf = http_range(url, data_off, data_off + n_rows * row_bytes - 1)
    return np.frombuffer(buf, dtype=dtype).reshape((n_rows,) + shape[1:]), shape[0]


def _read_json(repo, name):
    with open(hf_hub_download(repo, name, repo_type="dataset")) as f:
        return json.load(f)


def compute_vectors(repo, shard, layer, max_rows, present_z, store_order):
    columns = _read_json(repo, "columns.json")
    quant = _read_json(repo, "quant.json")
    stats = _read_json(repo, "corpus_stats.json")
    li = columns["layers"].index(layer)                       # store axis-1 index
    cols = [columns["concepts"].index(c) for c in store_order]  # by NAME, store order
    print(f"[emp] layer {layer} -> axis-1 {li}; concept cols {cols}")

    scores, total = read_npy_rows_ranged(
        hf_hub_url(repo, f"scores_{shard:05d}.npy", repo_type="dataset"), max_rows)
    print(f"[emp] shard {shard:05d}: {total:,} tokens; read {scores.shape[0]:,} rows")

    q8 = scores[:, li][:, cols].astype(np.float32)            # (n, r) int8
    scale = np.asarray(quant["scale"], np.float32)[li][cols]
    zero = np.asarray(quant["zero"], np.float32)[li][cols]
    mean = np.asarray(stats["mean"], np.float32)[li][cols]
    std = np.asarray(stats["std"], np.float32)[li][cols]
    z = ((q8 * scale + zero) - mean) / std                   # (n, r) standardized, STORE order

    argmax = z.argmax(axis=1)
    mx = z.max(axis=1)
    vectors, n_active = {}, {}
    for c, concept in enumerate(store_order):
        sel = (argmax == c) & (mx >= present_z)              # concept-c tokens past the threshold
        n_active[concept] = int(sel.sum())
        if sel.any():
            vectors[concept] = np.median(z[sel], axis=0).astype(np.float32).tolist()
        else:
            vectors[concept] = [0.0] * len(store_order)
            print(f"[emp] WARNING: 0 active tokens for {concept!r} in this sample "
                  f"(raise --max-rows or --shard); emitting zeros")
    return vectors, n_active


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True,
                    help="concept family in runs/lib/concepts.py (supplies the STORE order)")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--layer", type=int, default=8, choices=[6, 8, 14])
    ap.add_argument("--max-rows", type=int, default=3_000_000,
                    help="rows to fetch via HTTP Range (~162 B/row); 0 = full shard")
    ap.add_argument("--present-z", type=float, default=2.0)
    ap.add_argument("--out", required=True, help="the family's empirical_patterns.json")
    args = ap.parse_args()

    store_order = harness.family_concepts(args.family)
    vectors, n_active = compute_vectors(args.repo, args.shard, args.layer,
                                        args.max_rows, args.present_z, store_order)
    out = {
        "store_order": store_order, "present_z": args.present_z, "layer": args.layer,
        "repo": args.repo, "shard": args.shard, "max_rows": args.max_rows,
        "n_active": n_active, "vectors": vectors,
        "provenance": "median r-channel z over argmax==concept & max>=present_z tokens; "
                      "int8->raw->z with quant.json + corpus_stats.json at the given layer; "
                      "ranged reads per attribution/examples/read_corpus_scores.py",
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[emp] wrote {args.out}")
    for concept in store_order:
        print(f"  {concept:9s} n={n_active[concept]:>7,}  "
              f"vec={[round(v, 2) for v in vectors[concept]]}")


if __name__ == "__main__":
    main()
