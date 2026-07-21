"""Driver: fit baseline-nanochat DoM concept probes for all registered families and
save ONE stacked .npz per layer (all 54 concepts x d), standardized by a single
global per-layer mu/sd so every concept shares one readout frame.

  python runs/lib/build_dom.py --ckpt-dir <dir> --step 2520 \
      --data-root <probe-train-data> --out runs/dom_baseline \
      [--push-repo kaushikreddyxyz/nanochat-d12-injections --push-subdir baseline]

Per-layer file dom_layer{L:02d}.npz: W_dom[54,d], mu[d], sd[d], auroc[54], dprime[54],
meter[54,4] (pos_mu,pos_sd,neg_mu,neg_sd of the val projection), concepts[54],
families[54], layer. dom_index.json holds the best layer per concept. Fit uses
mixed/<class>.train.jsonl; eval uses mixed/<class>.val.jsonl. See dom_probe.py.
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

import dom_probe as dp          # noqa: E402
import concepts as reg          # noqa: E402

FAMILIES = ["color_wheel", "continents", "directions", "months",
            "moon_phases", "seasons", "weekdays"]


def global_concepts():
    """54 concepts + their family, in global store-column order (asserted 0..53)."""
    concepts, families, cols = [], [], []
    for fam in FAMILIES:
        f = reg.get_family(fam)
        for c in f.store_order:
            concepts.append(c)
            families.append(fam)
            cols.append(f.store_column(c))
    assert cols == list(range(len(cols))), f"store columns not a 0..n cover: {cols}"
    return concepts, families


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
    m = mask.reshape(-1)
    return [o.reshape(-1, d).cpu().numpy()[m] for o in outs]


def sanity_ce(model, enc, bos_id, path, device, n_docs=30):
    """Guard against a tokenizer that does not match the checkpoint: a mismatch
    indexes the wrong embeddings and CE blows up toward ln(vocab)~10 nats."""
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


def _class_path(data_root, family, cls, split):
    return os.path.join(data_root, "data", family, "final", "mixed", f"{cls}.{split}.jsonl")


def fit_all(model, enc, bos_id, data_root, device, args):
    d = int(model.config.n_embd)
    L = len(model.transformer.h)
    concepts, families = global_concepts()
    C = len(concepts)                                     # 54
    accum = dp.DomAccum(L, d, concepts)

    # --- train pass: one global standardization, per-concept pos/neg means ---
    for fam in FAMILIES:
        f = reg.get_family(fam)
        for cls in f.store_order:
            ci = f.store_column(cls)
            docs = _load_docs(enc, _class_path(data_root, fam, cls, "train"), args.limit_docs)
            for ids_bt, mask, y in dp.batch_docs(docs, bos_id, args.max_tokens, args.max_rows):
                outs = dp.capture_layers(model, ids_bt, device)
                accum.add(ci, _gather(outs, mask, d), y)
        print(f"  fit {fam} done", flush=True)
    res = accum.finalize()
    W, mu, sd = res["W_dom"], res["mu"], res["sd"]        # [C,L,d], [L,d], [L,d]

    # --- val pass: per (concept, layer) AUROC / d' / meter of the projection ---
    au = np.full((C, L), np.nan)
    dpr = np.full((C, L), np.nan)
    meter = np.zeros((C, L, 4))
    for fam in FAMILIES:
        f = reg.get_family(fam)
        for cls in f.store_order:
            ci = f.store_column(cls)
            docs = _load_docs(enc, _class_path(data_root, fam, cls, "val"), args.limit_docs)
            pj = [([], []) for _ in range(L)]
            for ids_bt, mask, y in dp.batch_docs(docs, bos_id, args.max_tokens, args.max_rows):
                outs = dp.capture_layers(model, ids_bt, device)
                rows = _gather(outs, mask, d)
                pos = y >= dp.BINARIZE_AT
                for l, X in enumerate(rows):
                    p = dp.project(X, mu[l], sd[l], W[ci, l])
                    pj[l][0].append(p[pos])
                    pj[l][1].append(p[~pos])
            for l in range(L):
                P = np.concatenate(pj[l][0]) if pj[l][0] else np.zeros(0)
                N = np.concatenate(pj[l][1]) if pj[l][1] else np.zeros(0)
                au[ci, l] = dp.auroc(P, N)
                dpr[ci, l] = dp.dprime(P, N)
                meter[ci, l] = [P.mean() if P.size else 0.0, P.std() if P.size > 1 else 1.0,
                                N.mean() if N.size else 0.0, N.std() if N.size > 1 else 1.0]

    best = np.nanargmax(au, axis=1).astype(np.int64)
    return {"concepts": concepts, "families": families, "L": L, "d": d,
            "W": W, "mu": mu, "sd": sd, "auroc": au, "dprime": dpr, "meter": meter,
            "best_layer": best, "pos_n": res["pos_n"], "neg_n": res["neg_n"]}


def save_stacked(out_dir, r):
    os.makedirs(out_dir, exist_ok=True)
    concepts = np.array(r["concepts"])
    families = np.array(r["families"])
    for l in range(r["L"]):
        np.savez_compressed(
            os.path.join(out_dir, f"dom_layer{l:02d}.npz"),
            W_dom=r["W"][:, l, :].astype(np.float32),
            mu=r["mu"][l].astype(np.float32), sd=r["sd"][l].astype(np.float32),
            auroc=r["auroc"][:, l], dprime=r["dprime"][:, l],
            meter=r["meter"][:, l, :], concepts=concepts, families=families,
            layer=np.int64(l), binarize_at=np.float32(dp.BINARIZE_AT))
    idx = {
        "layers": r["L"], "d": r["d"], "binarize_at": dp.BINARIZE_AT,
        "concepts": r["concepts"], "families": r["families"],
        "standardization": "global per-layer mu/sd over all train tokens (all families)",
        "layer_file": "dom_layer{L:02d}.npz  (W_dom[54,d], mu[d], sd[d], auroc[54], dprime[54], meter[54,4])",
        "best_layer": {c: int(r["best_layer"][i]) for i, c in enumerate(r["concepts"])},
        "auroc_at_best": {c: round(float(r["auroc"][i, r["best_layer"][i]]), 4)
                          for i, c in enumerate(r["concepts"])},
        "auroc_per_layer": {c: [round(float(x), 4) for x in r["auroc"][i]]
                            for i, c in enumerate(r["concepts"])},
        "pos_tokens": {c: int(r["pos_n"][i, 0]) for i, c in enumerate(r["concepts"])},
    }
    with open(os.path.join(out_dir, "dom_index.json"), "w") as f:
        json.dump(idx, f, indent=2)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-rows", type=int, default=64)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--push-repo", default=None)
    ap.add_argument("--push-subdir", default="baseline")
    args = ap.parse_args()
    args.limit_docs = args.limit_docs or None

    import torch
    from nanochat.checkpoint_manager import build_model
    dev = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model, tokenizer, _meta = build_model(args.ckpt_dir, args.step, dev, "eval")
    enc = tokenizer.enc
    bos_id = tokenizer.get_bos_token_id()
    print(f"model d={model.config.n_embd} n_layer={len(model.transformer.h)} device={dev}", flush=True)

    ce = sanity_ce(model, enc, bos_id,
                   _class_path(args.data_root, "seasons", "autumn", "val"), dev)
    print(f"sanity CE (nats) = {ce:.3f}  (val_bpb ref 0.859; expect <~4)", flush=True)
    if ce > 5.0:
        raise SystemExit(f"ABORT: CE {ce:.2f} nats too high -- tokenizer/checkpoint mismatch")

    t0 = time.time()
    r = fit_all(model, enc, bos_id, args.data_root, dev, args)
    idx = save_stacked(args.out, r)
    print(f"fit+save {time.time()-t0:.0f}s -> {r['L']} layer files in {args.out}", flush=True)
    top = sorted(idx["auroc_at_best"].items(), key=lambda kv: -kv[1])
    print("best-layer AUROC (top/bottom 5):", flush=True)
    for c, a in top[:5] + top[-5:]:
        print(f"    {c:16s} L{idx['best_layer'][c]:<2d} auroc={a:.3f}", flush=True)

    if args.push_repo:
        from huggingface_hub import HfApi
        HfApi().upload_folder(folder_path=args.out, repo_id=args.push_repo,
                              path_in_repo=args.push_subdir, repo_type="model",
                              allow_patterns=["dom_layer*.npz", "dom_index.json"])
        print(f"pushed -> {args.push_repo}/{args.push_subdir}/", flush=True)


if __name__ == "__main__":
    main()
