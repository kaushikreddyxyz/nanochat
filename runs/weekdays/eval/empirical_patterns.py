"""Offline helper: the 7 EMPIRICAL weekday z-vectors used by the causal eval's
``cf_swap(Y, empirical)`` / ``implant(Y, empirical)`` conditions.

For each weekday Y we want the activation pattern that day naturally produces —
the median 7-channel z-vector over the tokens where Y genuinely fires. This is
the realistic counterpart of the ``onehot`` pattern (z=3.0 on Y's channel, 0
elsewhere): it carries the real cross-channel structure (e.g. Friday
co-activating a little with Thursday/Saturday) that the gemma probes emit.

Provenance / method (mirrors runs/weekdays/weekday_source.py + exp2_config.json):
  * Store: kaushikreddyxyz/climbmix-scored (the training corpus), gemma LAYER 8
    (store axis-1 index = columns['layers'].index(8)).
  * Weekday columns: the 7 STORE-order names (friday..wednesday, cols ~47..53),
    looked up by NAME in columns.json (never hard-code the indices).
  * int8 -> raw -> z: raw = int8*scale + zero (quant.json), z = (raw-mean)/std
    (corpus_stats.json), both indexed [layer][concept] — identical to the
    training source's _standardize path.
  * "Y fires" == present_z realism threshold semantics: max over the 7 weekday
    channels >= present_z (default 2.0) AND Y is the argmax channel (i.e. this is
    a day-Y token, not a token where some other day dominates). Median of the
    full 7-vector over those rows -> vectors[Y].

Ranged .npy reads (header + first --max-rows rows over HTTP Range, no 7.5 GB
download) are adapted verbatim from
  ../attribution/examples/read_corpus_scores.py
Needs the score store only — NO gemma tokenizer (nothing is decoded), so this
runs without HF_TOKEN / the gated tokenizer.

Output: runs/weekdays/eval/empirical_patterns.json
  {"store_order": [...7...], "present_z": 2.0, "layer": 8, "repo": ..., "shard":
   ..., "max_rows": ..., "n_active": {day: count}, "vectors": {day: [7 floats]}}

Usage:
  python runs/weekdays/eval/empirical_patterns.py            # defaults
  python runs/weekdays/eval/empirical_patterns.py --shard 0 --max-rows 3000000
"""
from __future__ import annotations

import argparse
import ast
import json
import os

import numpy as np
import requests
from huggingface_hub import hf_hub_download, hf_hub_url

from causal_items import STORE_ORDER  # channel order these vectors MUST match

DEFAULT_REPO = "kaushikreddyxyz/climbmix-scored"
OUT_PATH = os.path.join(os.path.dirname(__file__), "empirical_patterns.json")


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


def compute_vectors(repo, shard, layer, max_rows, present_z):
    columns = _read_json(repo, "columns.json")
    quant = _read_json(repo, "quant.json")
    stats = _read_json(repo, "corpus_stats.json")
    li = columns["layers"].index(layer)                       # store axis-1 index
    cols = [columns["concepts"].index(c) for c in STORE_ORDER]  # by NAME, store order
    print(f"[emp] layer {layer} -> axis-1 {li}; weekday cols {cols}")

    scores, total = read_npy_rows_ranged(
        hf_hub_url(repo, f"scores_{shard:05d}.npy", repo_type="dataset"), max_rows)
    print(f"[emp] shard {shard:05d}: {total:,} tokens; read {scores.shape[0]:,} rows")

    q8 = scores[:, li][:, cols].astype(np.float32)            # (n, 7) int8
    scale = np.asarray(quant["scale"], np.float32)[li][cols]
    zero = np.asarray(quant["zero"], np.float32)[li][cols]
    mean = np.asarray(stats["mean"], np.float32)[li][cols]
    std = np.asarray(stats["std"], np.float32)[li][cols]
    z = ((q8 * scale + zero) - mean) / std                   # (n, 7) standardized, STORE order

    argmax = z.argmax(axis=1)
    mx = z.max(axis=1)
    vectors, n_active = {}, {}
    for c, day in enumerate(STORE_ORDER):
        sel = (argmax == c) & (mx >= present_z)              # day-c tokens past the realism threshold
        n_active[day] = int(sel.sum())
        if sel.any():
            vectors[day] = np.median(z[sel], axis=0).astype(np.float32).tolist()
        else:
            vectors[day] = [0.0] * 7
            print(f"[emp] WARNING: 0 active tokens for {day!r} in this sample "
                  f"(raise --max-rows or --shard); emitting zeros")
    return vectors, n_active


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--layer", type=int, default=8, choices=[6, 8, 14])
    ap.add_argument("--max-rows", type=int, default=3_000_000,
                    help="rows to fetch via HTTP Range (~162 B/row); 0 = full shard")
    ap.add_argument("--present-z", type=float, default=2.0)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    vectors, n_active = compute_vectors(args.repo, args.shard, args.layer,
                                        args.max_rows, args.present_z)
    out = {
        "store_order": STORE_ORDER, "present_z": args.present_z, "layer": args.layer,
        "repo": args.repo, "shard": args.shard, "max_rows": args.max_rows,
        "n_active": n_active, "vectors": vectors,
        "provenance": "median 7-channel z over argmax==day & max>=present_z tokens; "
                      "int8->raw->z with quant.json + corpus_stats.json at gemma L8; "
                      "ranged reads per attribution/examples/read_corpus_scores.py",
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[emp] wrote {args.out}")
    for day in STORE_ORDER:
        print(f"  {day:9s} n={n_active[day]:>7,}  "
              f"vec={[round(v, 2) for v in vectors[day]]}")


if __name__ == "__main__":
    main()
