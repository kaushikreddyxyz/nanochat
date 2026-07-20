"""figP5-figP7 summary figures (probe-vs-D alignment incl. baseline null; per-day
probe quality for all 10 conditions; ON<->OFF probe rotation). Reads
probe_report.json only; writes into figures/.
Run: python3 runs/weekdays/probes/make_summary_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
REPORT = json.loads((HERE / "probe_report.json").read_text())
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)

# store NAME order -> calendar order
STORE = REPORT["notes"]["store_order"]  # friday, monday, ... (name-sorted)
CAL = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
CAL_IDX = [STORE.index(d) for d in CAL]
CAL_LAB = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

ARMS = ["trainable", "sphere", "orthogonal"]
ARM_COLOR = {"trainable": "#1f77b4", "sphere": "#2ca02c", "orthogonal": "#d62728"}
OFF_COLOR = "#add8e6"
BOLT_COLOR = "#555555"

VSD = REPORT["vs_direction"]
CONDS = REPORT["conditions"]


def per_day(arm, cond, pop, kind):
    """matched-day cosines vs D in calendar order. cond: on|off|bolt|baseline_off"""
    key = f"{cond}_{pop}_{kind}"
    return np.array(VSD[arm][key]["matched_per_day"])[CAL_IDX]


# ---------------------------------------------------------------- figP5
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
bar_specs = [  # (condition key, label, facecolor fn, hatch)
    ("on", "ON", lambda a: ARM_COLOR[a], None),
    ("off", "OFF", lambda a: OFF_COLOR, None),
    ("bolt", "bolt-on (baseline+D, untrained)", lambda a: BOLT_COLOR, None),
    ("baseline_off", "baseline null (no D ever)", lambda a: "white", "//"),
]
for row, kind in enumerate(["enc", "V"]):
    for col, pop in enumerate(["all", "act"]):
        ax = axes[row, col]
        for ai, arm in enumerate(ARMS):
            for bi, (cond, label, fc, hatch) in enumerate(bar_specs):
                vals = per_day(arm, cond, pop, kind)
                x = ai + (bi - 1.5) * 0.19
                ax.bar(x, vals.mean(), width=0.17, color=fc(arm), hatch=hatch,
                       edgecolor="black", linewidth=0.6,
                       label=label if ai == 0 else None)
                ax.plot(np.full(7, x) + np.linspace(-0.03, 0.03, 7), vals,
                        ".", color="black", ms=4, alpha=0.7, zorder=3)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(ARMS)
        kname = "encoding rows (cross-covariance)" if kind == "enc" else "decoder rows V=W/σ"
        pname = "all tokens" if pop == "all" else "active tokens (any z≥2)"
        ax.set_title(f"{kname} — {pname}", fontsize=11)
        ax.set_ylim(-0.15, 0.9)
        if col == 0:
            ax.set_ylabel("matched-day cos(probe row, D row)")
axes[0, 0].legend(fontsize=9, loc="upper left")
fig.suptitle("Probe-vs-D alignment, every condition (dots = the 7 days)", fontsize=14)
fig.tight_layout()
fig.savefig(FIGS / "figP5_vsD_all_conditions.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- figP6
ORDER = [
    ("baseline_off", "baseline (no site)"),
    ("trainable_off", "trainable OFF"),
    ("trainable_on", "trainable ON"),
    ("bolt_trainable_on", "baseline + trainable D"),
    ("sphere_off", "sphere OFF"),
    ("sphere_on", "sphere ON"),
    ("bolt_sphere_on", "baseline + sphere D"),
    ("orthogonal_off", "orthogonal OFF"),
    ("orthogonal_on", "orthogonal ON"),
    ("bolt_orthogonal_on", "baseline + orthogonal D"),
]
r2 = np.array([np.array(CONDS[k]["act"]["r2_heldout"])[CAL_IDX] for k, _ in ORDER])
acc = np.array([[CONDS[k]["act"]["argmax_acc"]] for k, _ in ORDER])
ce = np.array([[CONDS[k]["ce_mean"]] for k, _ in ORDER])

fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(12.5, 6), gridspec_kw={"width_ratios": [7, 1, 1]})
im1 = ax1.imshow(r2, cmap="Reds", vmin=0.0, vmax=0.55, aspect="auto")
for i in range(r2.shape[0]):
    for j in range(7):
        ax1.text(j, i, f"{r2[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                 color="white" if r2[i, j] > 0.42 else "black")
ax1.set_xticks(range(7)); ax1.set_xticklabels(CAL_LAB)
ax1.set_yticks(range(len(ORDER))); ax1.set_yticklabels([lab for _, lab in ORDER], fontsize=9)
ax1.set_title("heldout R² per day (active tokens)", fontsize=11)

im2 = ax2.imshow(acc, cmap="Blues", vmin=1 / 7, vmax=0.6, aspect="auto")
for i in range(acc.shape[0]):
    ax2.text(0, i, f"{acc[i, 0]:.3f}", ha="center", va="center", fontsize=8.5)
ax2.set_xticks([0]); ax2.set_xticklabels(["acc"]); ax2.set_yticks([])
ax2.set_title("day argmax\n(chance .143)", fontsize=9)

im3 = ax3.imshow(ce, cmap="Purples_r", aspect="auto")
for i in range(ce.shape[0]):
    ax3.text(0, i, f"{ce[i, 0]:.4f}", ha="center", va="center", fontsize=8)
ax3.set_xticks([0]); ax3.set_xticklabels(["CE"]); ax3.set_yticks([])
ax3.set_title("mean CE\n(lower=better)", fontsize=9)

fig.suptitle("Probe quality, all 10 conditions", fontsize=14)
fig.tight_layout()
fig.savefig(FIGS / "figP6_perday_quality.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- figP7
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for col, pop in enumerate(["all", "act"]):
    ax = axes[col]
    for ai, arm in enumerate(ARMS):
        for bi, kind in enumerate(["enc", "V"]):
            vals = np.array(VSD[arm][f"on_vs_off_{pop}_{kind}_per_day"])[CAL_IDX]
            x = ai + (bi - 0.5) * 0.32
            ax.bar(x, vals.mean(), width=0.28,
                   color=OFF_COLOR if kind == "enc" else ARM_COLOR[arm],
                   edgecolor="black", linewidth=0.6,
                   label=("encoding rows" if kind == "enc" else "decoder rows V")
                         if ai == 0 else None)
            ax.plot(np.full(7, x) + np.linspace(-0.05, 0.05, 7), vals,
                    ".", color="black", ms=4, alpha=0.7, zorder=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels(ARMS)
    ax.set_title("all tokens" if pop == "all" else "active tokens", fontsize=11)
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("cos(ON probe row, OFF probe row)")
axes[0].legend(fontsize=9, loc="lower left")
fig.suptitle("Probe rotation ON↔OFF: signal location is stable, the whitened readout rotates", fontsize=13)
fig.tight_layout()
fig.savefig(FIGS / "figP7_probe_rotation.png", dpi=140)
plt.close(fig)

print("wrote", *[p.name for p in sorted(FIGS.glob('figP[567]*.png'))])
