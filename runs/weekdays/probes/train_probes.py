#!/usr/bin/env python3
"""Phase 2 (pod, one process per arm — parallelizable): train ridge probes on an
arm's post-block-3 (+site) residual stream against the injected weekday acts,
injection ON vs OFF, plus (for injected arms) the baseline+bolt-on control:
the arm's CHECKPOINT direction attached to the BASELINE model (a model that
never trained with the injection) — the probe-recovery CEILING.

Conditions by --arm:
  baseline   : baseline_off (vanilla forward, block-3 hook)
  trainable  : trainable_off, trainable_on, bolt_trainable_on
  sphere     : sphere_off,    sphere_on,    bolt_sphere_on
  orthogonal : orthogonal_off, orthogonal_on, bolt_orthogonal_on

Writes results/probe_{condition}.npz (+ .json) and, for each injected arm, the
checkpoint's direction matrix results/direction_{arm}.npy (the D the probes are
later compared against — saved from the SAME loaded checkpoint, no re-download).

Run from the nanochat repo root (pod):
    python runs/weekdays/probes/train_probes.py --arm trainable --device cuda
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import probe_lib  # noqa: E402


def _harness():
    ev = os.path.join(REPO, "runs", "weekdays", "eval")
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
    ap.add_argument("--max-tokens", type=int, default=65536)
    args = ap.parse_args()

    H = _harness()
    os.makedirs(args.out_dir, exist_ok=True)
    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    docs, meta = blob["docs"], blob["meta"]
    bos_id = meta["bos_id"]
    print(f"[{args.arm}] data: {meta['n_docs']} docs, {meta['n_tokens']} tokens, "
          f"active_frac={meta['active_frac']:.3f}")

    def run(name, model, gate_scale, use_acts):
        print(f"[{args.arm}] condition {name} (gate_scale={gate_scale}, acts={use_acts})")
        res = probe_lib.run_condition(
            model, docs, bos_id, gate_scale=gate_scale, use_acts=use_acts,
            device=args.device, forward_metrics=H.forward_metrics,
            heldout_every=args.heldout_every, max_rows=args.max_rows,
            max_tokens=args.max_tokens,
            log=lambda s: print(f"[{args.arm}] {s}"))
        res["condition"] = name
        res["arm"] = args.arm
        res["data_meta"] = meta
        js = probe_lib.save_condition(os.path.join(args.out_dir, f"probe_{name}.npz"), res)
        with open(os.path.join(args.out_dir, f"probe_{name}.json"), "w") as f:
            json.dump(js, f, indent=1)
        print(f"[{args.arm}] wrote probe_{name}.npz/.json  ce_mean={res['ce_mean']:.4f}")

    if args.arm == "baseline":
        model, _ = H.load_model("baseline", args.device)
        run("baseline_off", model, 0.0, False)
        return

    # injected arm: off / on, then the baseline+bolt-on ceiling control
    model, _ = H.load_model(args.arm, args.device)
    D = model.injection_sites[probe_lib.SITE_NAME].direction.detach().float().cpu().numpy()
    np.save(os.path.join(args.out_dir, f"direction_{args.arm}.npy"), D)
    run(f"{args.arm}_off", model, 0.0, True)
    run(f"{args.arm}_on", model, 1.0, True)
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    base, _ = H.load_model("baseline", args.device)
    H.attach_site(base, D, gate=H.GATE)          # checkpoint D, trained gate
    run(f"bolt_{args.arm}_on", base, 1.0, True)


if __name__ == "__main__":
    main()
