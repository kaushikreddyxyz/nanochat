#!/usr/bin/env python3
"""Phase 2 (pod, one process per arm): weekday entry point for runs/lib/probe_arms.py —
binds the weekday ProbeSpec (site 'weekdays', after_block 3) + the shared eval harness.
Run from repo root: python runs/weekdays/probes/train_probes.py --arm trainable --device cuda
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "runs", "lib"))

import probe_arms  # noqa: E402
from probe_lib import ProbeSpec  # noqa: E402

SITE_NAME = "weekdays"
AFTER_BLOCK = 3
D_MODEL = 768
HF_REPO = "kaushikreddyxyz/weekday-geometry-d12"
STEP = 2520


def _harness():
    ev = os.path.join(REPO, "runs", "lib", "eval")
    if ev not in sys.path:
        sys.path.insert(0, ev)
    import harness
    return harness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["baseline", "trainable", "sphere", "orthogonal"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data", default=os.path.join(HERE, "probe_data.pt"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--heldout-every", type=int, default=10)
    ap.add_argument("--max-rows", type=int, default=32)
    # 16k tokens/batch: the [B,T,32k-padded-vocab] fp32 logits (+ CE reshape copies) peak
    # ~6-8 GB/process — 4 arm processes fit one 80GB card. 64k OOM'd the parallel run.
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--hf-repo", default=HF_REPO)
    ap.add_argument("--step", type=int, default=STEP)
    args = ap.parse_args()

    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    docs, meta = blob["docs"], blob["meta"]
    print(f"[{args.arm}] data: {meta['n_docs']} docs, {meta['n_tokens']} tokens, "
          f"active_frac={meta['active_frac']:.3f}")

    spec = ProbeSpec.from_family("weekdays", AFTER_BLOCK, D_MODEL, site_name=SITE_NAME)
    probe_arms.run_arm(spec, _harness(), args.arm, docs, meta, args.out_dir,
                       args.hf_repo, args.step, device=args.device,
                       heldout_every=args.heldout_every, max_rows=args.max_rows,
                       max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
