"""Arm-level probe driver + probe-vs-direction comparison, for any concept family.
The per-family entry point supplies a ProbeSpec and an eval harness exposing
load_model/attach_site/site_params/forward_metrics. Import: sys.path.insert this dir.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import probe_lib  # noqa: E402

# Conditions per arm. The baseline arm has no site, so it only has the vanilla pass;
# "bolt_{arm}_on" attaches the trained arm's D to the BASELINE model — the ceiling
# control separating "the direction is readable" from "this arm learned to read it".
BASELINE_ARM = "baseline"


def run_arm(spec, harness, arm, docs, meta, out_dir, hf_repo, step, *, device="cuda",
            heldout_every=10, max_rows=32, max_tokens=16384, log=print):
    """Probe one arm end to end; writes probe_{condition}.npz/.json (+ direction_{arm}.npy
    for injected arms) into out_dir. Returns the list of condition names written."""
    os.makedirs(out_dir, exist_ok=True)
    written = []

    def run(name, model, loudness_scale, use_acts):
        log(f"[{arm}] condition {name} (loudness_scale={loudness_scale}, acts={use_acts})")
        res = probe_lib.run_condition(
            model, docs, meta["bos_id"], spec, loudness_scale=loudness_scale, use_acts=use_acts,
            device=device, forward_metrics=harness.forward_metrics,
            heldout_every=heldout_every, max_rows=max_rows, max_tokens=max_tokens,
            log=lambda s: log(f"[{arm}] {s}"))
        res.update(condition=name, arm=arm, data_meta=meta)
        js = probe_lib.save_condition(os.path.join(out_dir, f"probe_{name}.npz"), res)
        with open(os.path.join(out_dir, f"probe_{name}.json"), "w") as f:
            json.dump(js, f, indent=1)
        log(f"[{arm}] wrote probe_{name}.npz/.json  ce_mean={res['ce_mean']:.4f}")
        written.append(name)

    if arm == BASELINE_ARM:
        model, _ = harness.load_model(BASELINE_ARM, device, hf_repo, step)
        run(f"{BASELINE_ARM}_off", model, 0.0, False)
        return written

    model, _ = harness.load_model(arm, device, hf_repo, step)
    site = harness.site_params(model, spec.site_name)
    np.save(os.path.join(out_dir, f"direction_{arm}.npy"), site["direction"])
    run(f"{arm}_off", model, 0.0, True)
    run(f"{arm}_on", model, 1.0, True)
    del model
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()

    base, _ = harness.load_model(BASELINE_ARM, device, hf_repo, step)
    harness.attach_site(base, name=spec.site_name, **site)  # the arm's own D + loudness
    run(f"bolt_{arm}_on", base, 1.0, True)
    return written


# --------------------------------------------------------------------------- #
# analysis-side loaders / geometry
# --------------------------------------------------------------------------- #
def unit(M):
    return M / np.linalg.norm(M, axis=-1, keepdims=True)


def load_condition(res_dir, name):
    """One condition's npz+json as {json, all:{...}, act:{...}}. ``V`` is the ridge
    decoder readout (covariance-whitened); ``enc`` is the raw cross-covariance
    encoding estimate — where the signal LIVES, which is the geometric claim."""
    z = np.load(os.path.join(res_dir, f"probe_{name}.npz"))
    js = json.load(open(os.path.join(res_dir, f"probe_{name}.json")))
    out = {"json": js}
    for pop in ("all", "act"):
        p = js["pops"][pop]
        out[pop] = {"V": z[f"{pop}_V"], "enc": z[f"{pop}_Cxy"].T,
                    "r2": np.array(p["r2_heldout"]), "r2_mean": p["r2_heldout_mean"],
                    "acc": p.get("argmax_acc_te_act"), "n_train": p["n_train"]}
    return out


def compare_to_direction(rows, D):
    """rows, D: [r, d]. Matched per-channel cosine, the full r x r matrix, and the
    principal angles between the two row spaces."""
    from scipy.linalg import subspace_angles
    C = unit(rows) @ unit(D).T
    return {"matched": np.diag(C), "matrix": C,
            "principal_angles_deg": np.degrees(subspace_angles(rows.T, D.T)),
            "matched_mean": float(np.diag(C).mean())}
