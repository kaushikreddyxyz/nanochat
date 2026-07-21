"""Driver: fit baseline-nanochat DoM concept probes for one or more families and
save per-family {W_dom, mu, sd, per-layer AUROC/d', best layer, calibrated meter}.

  python -m runs.lib.build_dom --ckpt-dir <dir> --step 2520 \
      --data-root <probe-train-data> --families all --out runs/dom_baseline \
      [--push-repo kaushikreddyxyz/nanochat-d12-injections --push-subdir dom_baseline]

Fit uses mixed/<class>.train.jsonl; eval uses mixed/<class>.val.jsonl. See dom_probe.py
for the DoM recipe. Runs on cuda (default) / cpu / mps.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO_ROOT)

import dom_probe as dp  # noqa: E402

FAMILIES = ["color_wheel", "continents", "directions", "months",
            "moon_phases", "seasons", "weekdays"]


def _load_docs(enc, path, limit=None):
    docs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            r = json.loads(line)
            ids, y = dp.tokenize_and_target(enc, r["text"], r.get("target_spans") or [])
            if ids:
                docs.append((ids, y))
    return docs


def _gather(outs, mask, d):
    """outs: list of [B,T,d] torch tensors; mask [B,T] bool -> list of [M,d] numpy."""
    m = mask.reshape(-1)
    return [o.reshape(-1, d).cpu().numpy()[m] for o in outs]


def sanity_ce(model, enc, bos_id, path, device, n_docs=30):
    """Guard against a tokenizer that does not match the checkpoint: a mismatch
    indexes the wrong embeddings and CE blows up toward ln(vocab)~10 nats. A correct
    tokenizer gives a few nats on natural prose. Returns mean next-token CE (nats)."""
    import torch
    tot, ntok = 0.0, 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_docs:
                break
            ids = enc.encode_ordinary(json.loads(line)["text"])
            if len(ids) < 2:
                continue
            idx = torch.tensor([[bos_id] + ids], device=device)
            tgt = torch.tensor([ids + [-1]], device=device)
            with torch.no_grad():
                loss = model(idx, targets=tgt)
            tot += float(loss) * len(ids)
            ntok += len(ids)
    return tot / max(ntok, 1)


def fit_family(model, enc, bos_id, family, data_root, device, args):
    d = int(model.config.n_embd)
    n_layers = len(model.transformer.h)
    classes = dp.family_classes(family)
    mixed = os.path.join(data_root, "data", family, "final", "mixed")

    accum = dp.DomAccum(n_layers, d, classes)
    for ci, cls in enumerate(classes):
        docs = _load_docs(enc, os.path.join(mixed, f"{cls}.train.jsonl"), args.limit_docs)
        for ids_bt, mask, y in dp.batch_docs(docs, bos_id, args.max_tokens, args.max_rows):
            outs = dp.capture_layers(model, ids_bt, device)
            accum.add(ci, _gather(outs, mask, d), y)
    res = accum.finalize()
    W, mu, sd = res["W_dom"], res["mu"], res["sd"]

    # eval on val split: per (class, layer) AUROC / d' of the projection
    C = len(classes)
    au = np.full((C, n_layers), np.nan)
    dpr = np.full((C, n_layers), np.nan)
    meter = np.zeros((C, 4))  # pos_mu, pos_sd, neg_mu, neg_sd at best layer
    for ci, cls in enumerate(classes):
        docs = _load_docs(enc, os.path.join(mixed, f"{cls}.val.jsonl"), args.limit_docs)
        pj = [([], []) for _ in range(n_layers)]  # (pos, neg) projections per layer
        for ids_bt, mask, y in dp.batch_docs(docs, bos_id, args.max_tokens, args.max_rows):
            outs = dp.capture_layers(model, ids_bt, device)
            rows = _gather(outs, mask, d)
            pos = y >= dp.BINARIZE_AT
            for l, X in enumerate(rows):
                p = dp.project(X, mu[l], sd[l], W[ci, l])
                pj[l][0].append(p[pos])
                pj[l][1].append(p[~pos])
        for l in range(n_layers):
            P = np.concatenate(pj[l][0]) if pj[l][0] else np.zeros(0)
            N = np.concatenate(pj[l][1]) if pj[l][1] else np.zeros(0)
            au[ci, l] = dp.auroc(P, N)
            dpr[ci, l] = dp.dprime(P, N)
        bl = int(np.nanargmax(au[ci]))
        P = np.concatenate([x for x in pj[bl][0] if x.size]) if any(x.size for x in pj[bl][0]) else np.zeros(0)
        N = np.concatenate([x for x in pj[bl][1] if x.size]) if any(x.size for x in pj[bl][1]) else np.zeros(0)
        meter[ci] = [P.mean() if P.size else 0.0, P.std() if P.size > 1 else 1.0,
                     N.mean() if N.size else 0.0, N.std() if N.size > 1 else 1.0]

    best_layer = np.nanargmax(au, axis=1).astype(np.int64)
    return {
        "family": family, "concepts": classes, "n_layers": n_layers, "d": d,
        "W_dom": W, "mu": mu, "sd": sd, "auroc": au, "dprime": dpr,
        "best_layer": best_layer, "meter": meter,
        "pos_n": res["pos_n"], "neg_n": res["neg_n"],
        "binarize_at": dp.BINARIZE_AT,
    }


def save_family(out_dir, r):
    os.makedirs(out_dir, exist_ok=True)
    fam = r["family"]
    np.savez_compressed(
        os.path.join(out_dir, f"{fam}_dom.npz"),
        W_dom=r["W_dom"].astype(np.float32), mu=r["mu"].astype(np.float32),
        sd=r["sd"].astype(np.float32), auroc=r["auroc"], dprime=r["dprime"],
        best_layer=r["best_layer"], meter=r["meter"], pos_n=r["pos_n"], neg_n=r["neg_n"],
        concepts=np.array(r["concepts"]))
    summary = {
        "family": fam, "concepts": r["concepts"], "n_layers": r["n_layers"], "d": r["d"],
        "binarize_at": r["binarize_at"],
        "best_layer": {c: int(r["best_layer"][i]) for i, c in enumerate(r["concepts"])},
        "auroc_at_best": {c: float(r["auroc"][i, r["best_layer"][i]])
                          for i, c in enumerate(r["concepts"])},
        "auroc_per_layer": {c: [round(float(x), 4) for x in r["auroc"][i]]
                            for i, c in enumerate(r["concepts"])},
        "pos_tokens": {c: int(r["pos_n"][i, 0]) for i, c in enumerate(r["concepts"])},
        "neg_tokens": {c: int(r["neg_n"][i, 0]) for i, c in enumerate(r["concepts"])},
    }
    with open(os.path.join(out_dir, f"{fam}_dom.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--data-root", required=True, help="probe-train-data root (has data/<family>/...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-rows", type=int, default=64)
    ap.add_argument("--limit-docs", type=int, default=0, help="smoke: cap docs/class")
    ap.add_argument("--push-repo", default=None)
    ap.add_argument("--push-subdir", default="dom_baseline")
    args = ap.parse_args()
    args.limit_docs = args.limit_docs or None

    import torch
    from nanochat.checkpoint_manager import build_model
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model, tokenizer, _meta = build_model(args.ckpt_dir, args.step, device, "eval")
    enc = tokenizer.enc
    bos_id = tokenizer.get_bos_token_id()
    print(f"model d={model.config.n_embd} n_layer={len(model.transformer.h)} device={device}", flush=True)

    # Tokenizer/checkpoint match gate: mean CE must be a few nats, not ~ln(vocab).
    sane_path = os.path.join(args.data_root, "data", "seasons", "final", "mixed", "autumn.val.jsonl")
    ce = sanity_ce(model, enc, bos_id, sane_path, device)
    print(f"sanity CE (nats) on natural prose = {ce:.3f}  (val_bpb ref 0.859; expect <~4)", flush=True)
    if ce > 5.0:
        raise SystemExit(f"ABORT: CE {ce:.2f} nats too high -- tokenizer likely does not match checkpoint")

    fams = FAMILIES if args.families == "all" else args.families.split(",")
    summaries = {}
    for fam in fams:
        t0 = time.time()
        r = fit_family(model, enc, bos_id, fam, args.data_root, device, args)
        s = save_family(args.out, r)
        summaries[fam] = s
        best = {c: (s["best_layer"][c], round(s["auroc_at_best"][c], 3)) for c in s["concepts"]}
        print(f"[{fam}] {time.time()-t0:.0f}s  best(layer,auroc)={best}", flush=True)

    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump({"families": list(summaries), "step": args.step,
                   "ckpt_dir": args.ckpt_dir, "summaries": summaries}, f, indent=2)

    if args.push_repo:
        from huggingface_hub import HfApi
        HfApi().upload_folder(folder_path=args.out, repo_id=args.push_repo,
                              path_in_repo=args.push_subdir, repo_type="model")
        print(f"pushed -> {args.push_repo}/{args.push_subdir}/", flush=True)


if __name__ == "__main__":
    main()
