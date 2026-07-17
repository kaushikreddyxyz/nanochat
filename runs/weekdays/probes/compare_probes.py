#!/usr/bin/env python3
"""Local analysis: compare the trained probes (per arm, injection ON vs OFF,
plus baseline bolt-on ceilings) against the injection direction matrices D.

Inputs (pulled from the pod into runs/weekdays/probes/results/):
    probe_{condition}.npz/.json   conditions: baseline_off, {arm}_{off,on},
                                  bolt_{arm}_on   (arm in trainable/sphere/orthogonal)
    direction_{arm}.npy           the arm CHECKPOINT's D [7, 768]

Per condition the npz holds, for populations 'all' (every token) and 'act'
(tokens the injection fires on): W (std-space ridge), V = W/sigma (raw-space
DECODER readout rows, covariance-whitened), Cxy (raw cross-covariance =
ENCODING-direction estimate, whitening-free), mu/sigma/ybar.

Outputs: probe_report.json + figures/figP*.png. CPU, deterministic.
Run: python3 runs/weekdays/probes/compare_probes.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import subspace_angles

HERE = os.path.dirname(os.path.abspath(__file__))
WEEKDAYS = os.path.dirname(HERE)
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

STORE = ["friday", "monday", "saturday", "sunday", "thursday", "tuesday", "wednesday"]
CALENDAR = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
CAL_ROWS = [STORE.index(d) for d in CALENDAR]
CAL_LABELS = [d[:3].capitalize() for d in CALENDAR]
ARMS = ["trainable", "sphere", "orthogonal"]
PAL = {"trainable": "#1f77b4", "sphere": "#2ca02c", "orthogonal": "#d62728",
       "baseline": "#7f7f7f", "off": "#9edae5", "on": None, "bolt": "#555555"}


def unit(M):
    return M / np.linalg.norm(M, axis=-1, keepdims=True)


def to_cal(M):
    M = np.asarray(M)
    return M[np.ix_(CAL_ROWS, CAL_ROWS)] if M.ndim == 2 else M[CAL_ROWS]


def load_cond(name):
    z = np.load(os.path.join(RES, f"probe_{name}.npz"))
    js = json.load(open(os.path.join(RES, f"probe_{name}.json")))
    out = {"json": js}
    for pop in ("all", "act"):
        out[pop] = {
            "V": z[f"{pop}_V"],                 # [7, 768] decoder rows
            "enc": z[f"{pop}_Cxy"].T,           # [7, 768] encoding rows
            "r2": np.array(js["pops"][pop]["r2_heldout"]),
            "r2_mean": js["pops"][pop]["r2_heldout_mean"],
            "acc": js["pops"][pop].get("argmax_acc_te_act"),
            "n_train": js["pops"][pop]["n_train"],
        }
    return out


def cmp_vs_D(rows, D):
    """rows, D: [7, 768]. Matched per-day cosine, full 7x7, subspace angles."""
    Ru, Du = unit(rows), unit(D)
    C = Ru @ Du.T
    ang = np.degrees(subspace_angles(rows.T, D.T))
    return {"matched": np.diag(C), "matrix": C, "principal_angles_deg": ang,
            "matched_mean": float(np.diag(C).mean())}


def main():
    os.makedirs(FIG, exist_ok=True)
    conds = {}
    names = (["baseline_off"]
             + [f"{a}_{g}" for a in ARMS for g in ("off", "on")]
             + [f"bolt_{a}_on" for a in ARMS])
    for n in names:
        conds[n] = load_cond(n)
    D = {a: np.load(os.path.join(RES, f"direction_{a}.npy")).astype(np.float64) for a in ARMS}

    # frozen arms' checkpoint D must equal the committed npz (sanity)
    for a, f in (("sphere", "direction_sphere.npz"), ("orthogonal", "direction_orthogonal.npz")):
        ref = np.load(os.path.join(WEEKDAYS, f))["D"].astype(np.float64)
        assert np.abs(D[a] - ref).max() < 1e-5, f"checkpoint D != {f}"

    report = {"conditions": {}, "vs_direction": {}, "notes": {
        "decoder_V": "ridge readout rows V=W/sigma (covariance-whitened)",
        "encoder_Cxy": "raw stream-target cross-covariance rows (signal location)",
        "populations": {"all": "every non-BOS token", "act": "tokens with any weekday z>=2"},
        "store_order": STORE}}

    for n, c in conds.items():
        report["conditions"][n] = {
            pop: {"r2_heldout": c[pop]["r2"].tolist(), "r2_mean": c[pop]["r2_mean"],
                  "argmax_acc": c[pop]["acc"], "n_train": c[pop]["n_train"]}
            for pop in ("all", "act")}
        report["conditions"][n]["ce_mean"] = c["json"]["ce_mean"]

    # probe-vs-D for every (condition, arm-of-D) pair that matters
    for a in ARMS:
        rec = {}
        for label, cond in (("on", f"{a}_on"), ("off", f"{a}_off"),
                            ("bolt", f"bolt_{a}_on"), ("baseline_off", "baseline_off")):
            for pop in ("all", "act"):
                for kind in ("enc", "V"):
                    r = cmp_vs_D(conds[cond][pop][kind], D[a])
                    rec[f"{label}_{pop}_{kind}"] = {
                        "matched_per_day": r["matched"].tolist(),
                        "matched_mean": r["matched_mean"],
                        "matrix": r["matrix"].tolist(),
                        "principal_angles_deg": r["principal_angles_deg"].tolist()}
        # ON vs OFF probe rotation (per-day cosine between the two readouts)
        for pop in ("all", "act"):
            for kind in ("enc", "V"):
                on_u = unit(conds[f"{a}_on"][pop][kind])
                off_u = unit(conds[f"{a}_off"][pop][kind])
                rec[f"on_vs_off_{pop}_{kind}_per_day"] = np.sum(on_u * off_u, 1).tolist()
        report["vs_direction"][a] = rec

    with open(os.path.join(HERE, "probe_report.json"), "w") as f:
        json.dump(report, f, indent=1)

    # ---------------- figures ---------------- #
    # P1: heldout R2 (act population) per condition, grouped by arm
    fig, ax = plt.subplots(figsize=(9, 4.8))
    groups = [("baseline", ["baseline_off"])] + [(a, [f"{a}_off", f"{a}_on", f"bolt_{a}_on"])
                                                 for a in ARMS]
    xt, xl = [], []
    x = 0
    for arm, cs in groups:
        for n in cs:
            r2 = conds[n]["act"]["r2_mean"]
            kind = "off" if n.endswith("_off") else ("bolt" if n.startswith("bolt") else "on")
            col = PAL[arm] if kind == "on" else (PAL["off"] if kind == "off" else PAL["bolt"])
            ax.bar(x, r2, color=col, edgecolor="k", lw=0.4)
            acc = conds[n]["act"]["acc"]
            ax.text(x, max(r2, 0) + 0.01, f"acc {acc:.2f}" if acc is not None else "",
                    ha="center", fontsize=7)
            xt.append(x); xl.append(n.replace("_", "\n"))
            x += 1
        x += 0.6
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("heldout R2 (mean over 7 days, active tokens)")
    ax.set_title("Probe quality at the injection layer: OFF vs ON vs bolt-on ceiling")
    fig.savefig(os.path.join(FIG, "figP1_probe_r2.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # P2: per-day cos(encoding probe row, D row) — ON / OFF / bolt, one panel per arm
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    w = 0.27
    xs = np.arange(7)
    for ax, a in zip(axes, ARMS):
        for i, (label, cond, col) in enumerate((
                ("ON", f"{a}_on", PAL[a]), ("OFF", f"{a}_off", PAL["off"]),
                ("bolt-on ceiling", f"bolt_{a}_on", PAL["bolt"]))):
            m = np.diag(unit(conds[cond]["all"]["enc"]) @ unit(D[a]).T)
            ax.bar(xs + (i - 1) * w, to_cal(m), width=w, color=col, edgecolor="k",
                   lw=0.3, label=label)
        ax.set_xticks(xs); ax.set_xticklabels(CAL_LABELS, fontsize=8)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(a)
        ax.set_ylim(-0.3, 1.05)
    axes[0].set_ylabel("cos(probe encoding row, D row)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Where the probe finds the weekday signal vs the injection direction "
                 "(encoding estimate, all tokens)")
    fig.savefig(os.path.join(FIG, "figP2_encoding_vs_D.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # P3: 7x7 heatmaps cos(enc probe ON, D) per arm, calendar order
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, a in zip(axes, ARMS):
        C = to_cal(unit(conds[f"{a}_on"]["all"]["enc"]) @ unit(D[a]).T)
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.set_xticklabels(CAL_LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(CAL_LABELS, fontsize=8)
        ax.set_title(f"{a}: probe rows x D rows")
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(C[i, j]) > 0.55 else "black")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="cosine")
    fig.suptitle("ON encoding probe (rows) vs injection D (cols), calendar order")
    fig.savefig(os.path.join(FIG, "figP3_enc_heatmaps.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # P4: decoder V vs D matched-day mean + on-vs-off readout rotation
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    xs = np.arange(3)
    for i, (label, key, col) in enumerate((("ON", "on", None), ("OFF", "off", PAL["off"]),
                                           ("bolt", "bolt", PAL["bolt"]))):
        for kind, alpha, hatch in (("enc", 1.0, None), ("V", 0.55, "//")):
            vals = [report["vs_direction"][a][f"{key}_all_{kind}"]["matched_mean"]
                    for a in ARMS]
            axes[0].bar(xs + (i - 1) * 0.27 + (0.0 if kind == "enc" else 0.0), vals,
                        width=0.27, alpha=alpha, hatch=hatch,
                        color=[PAL[a] if key == "on" else col or PAL[a] for a in ARMS],
                        edgecolor="k", lw=0.3,
                        label=f"{label} {kind}" if True else None)
    axes[0].set_xticks(xs); axes[0].set_xticklabels(ARMS)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("mean matched-day cosine vs D")
    axes[0].set_title("encoder (solid) vs decoder (hatched)")
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h[:6], l[:6], fontsize=6, ncol=2)
    for a in ARMS:
        axes[1].plot(range(7),
                     to_cal(np.array(report["vs_direction"][a]["on_vs_off_all_enc_per_day"])),
                     "-o", color=PAL[a], label=a)
    axes[1].set_xticks(range(7)); axes[1].set_xticklabels(CAL_LABELS, fontsize=8)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_ylabel("cos(ON enc row, OFF enc row)")
    axes[1].set_title("does injection move where the signal lives?")
    axes[1].legend(fontsize=8)
    fig.savefig(os.path.join(FIG, "figP4_decoder_and_rotation.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # console headline
    print("=" * 72)
    for a in ARMS:
        r = report["vs_direction"][a]
        print(f"{a:11s} enc-vs-D matched mean: ON {r['on_all_enc']['matched_mean']:+.3f}  "
              f"OFF {r['off_all_enc']['matched_mean']:+.3f}  "
              f"bolt {r['bolt_all_enc']['matched_mean']:+.3f} | "
              f"act-R2 ON {conds[f'{a}_on']['act']['r2_mean']:.3f} "
              f"OFF {conds[f'{a}_off']['act']['r2_mean']:.3f}")
    print(f"baseline_off act-R2 {conds['baseline_off']['act']['r2_mean']:.3f}  "
          f"argmax acc {conds['baseline_off']['act']['acc']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
