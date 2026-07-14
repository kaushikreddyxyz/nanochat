"""Figures for the injection on/off eval results (results/{summary,causal_summary}.json).

Writes PNGs to figures/. Pure post-processing — no model, no network.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

ARMS = ["trainable", "sphere", "orthogonal"]
COL = {"baseline": "#7f7f7f", "trainable": "#1f77b4", "sphere": "#2ca02c", "orthogonal": "#d62728"}
LBL = {"trainable": "trainable", "sphere": "sphere (realistic)", "orthogonal": "orthogonal (null)"}

S = json.load(open(os.path.join(RES, "summary.json")))
C = json.load(open(os.path.join(RES, "causal_summary.json")))


def v(metric, arm, gate):
    return S[metric][arm][gate]["value"]


def n_of(metric, arm="baseline", gate="on"):
    return S[metric][arm][gate]["n"]


# ---------------------------------------------------------------- fig 1: bpb buckets
def fig1():
    buckets = [("valbpb_injected", f"injected tokens\n(n={n_of('valbpb_injected'):,} = "
                f"{100*n_of('valbpb_injected')/n_of('valbpb_overall'):.1f}%)"),
               ("valbpb_after", f"token after injected\n(n={n_of('valbpb_after'):,})"),
               ("valbpb_rest", f"all other tokens\n(n={n_of('valbpb_rest'):,})")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    for ax, (m, title) in zip(axes, buckets):
        base = v(m, "baseline", "on") * 1000
        ys = np.arange(len(ARMS))
        for i, arm in enumerate(ARMS):
            on, off = v(m, arm, "on") * 1000, v(m, arm, "off") * 1000
            ax.plot([off, on], [i, i], color=COL[arm], lw=1.5, zorder=2)
            ax.scatter([on], [i], color=COL[arm], s=70, zorder=3, label="_")
            ax.scatter([off], [i], facecolor="white", edgecolor=COL[arm], s=70, lw=1.8, zorder=3)
        ax.axvline(base, color=COL["baseline"], ls="--", lw=1.2)
        ax.text(base, 2.62, " baseline", color=COL["baseline"], fontsize=8, va="bottom",
                ha="center", rotation=0)
        ax.set_yticks(ys, [LBL[a] for a in ARMS] if ax is axes[0] else ["", "", ""])
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("val bits-per-byte (×10⁻³)", fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.set_ylim(-0.6, 3.0)
        ax.invert_yaxis()
    # legend proxy
    axes[0].scatter([], [], color="k", s=60, label="injection ON")
    axes[0].scatter([], [], facecolor="white", edgecolor="k", s=60, lw=1.6, label="injection OFF (gate 0)")
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Use + reliance, cleanly localized: on < baseline < off — but only where the injection fires",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_bpb_buckets.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 2: dose-response
def fig2():
    # absolute day-logits per dose recomputed from the per-item files (paired design)
    import math
    dose_conds = [("off", 0.0), ("dose@0.25", 0.25), ("dose@0.5", 0.5), ("clean_on", 1.0),
                  ("dose@2", 2.0), ("dose@4", 4.0)]
    meta = C["meta"]
    def curves(fname):
        d = json.load(open(os.path.join(RES, fname)))
        order = None
        txt = {k: [0.0, 0] for _, k in dose_conds}
        oth = {k: [0.0, 0] for _, k in dose_conds}
        for rec in d["items"]:
            it = rec["item"]
            if it["family"] != "mention":
                continue
            for cname, k in dose_conds:
                cond = rec["conds"].get(cname)
                if not cond or "day_logits" not in cond:
                    continue
                L = cond["day_logits"]
                if order is None:
                    order = meta["store_order"] if meta["store_order"][L.index(max(L))] == cond["argmax_day"]                         else meta["calendar_order"]
                i = order.index(it["text_answer"])
                txt[k][0] += L[i]; txt[k][1] += 1
                oth[k][0] += (sum(L) - L[i]) / 6; oth[k][1] += 1
        ks = [k for _, k in dose_conds]
        return ks, [txt[k][0] / max(txt[k][1], 1) for k in ks], [oth[k][0] / max(oth[k][1], 1) for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    ax = axes[0]
    for arm in ARMS:
        ks, t, o = curves(f"causal_{arm}.json")
        ax.plot(ks, t, "-o", color=COL[arm], ms=4.5, lw=1.8, label=f"{LBL[arm]}: text day")
        ax.plot(ks, o, ":", color=COL[arm], alpha=0.55, lw=1.3,
                label="mean of other 6 days" if arm == "trainable" else "_")
    ax.set_xlabel("gate multiplier (0 = off, 1 = trained loudness 0.0273)")
    ax.set_ylabel("mean logit at answer position (absolute)")
    ax.set_title("Absolute: mean logit of the TEXT day (solid)\nvs the other six days (dotted), per dose",
                 fontsize=9.5)
    ax.legend(fontsize=7.8, frameon=False)
    ax.tick_params(labelsize=8.5)

    ax = axes[1]
    dr = C["dose_response"]["correct_vs_off"]
    doses = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    for arm in ARMS:
        ax.plot(doses, [dr[arm][str(d)] for d in doses], "-o", color=COL[arm], label=LBL[arm], ms=4.5, lw=1.8)
    for arm in ARMS:
        ax.plot(doses, [dr[f"baseline_{arm}"][str(d)] for d in doses], "--", color=COL[arm], alpha=0.35,
                lw=1.2, label="controls (untrained + direction)" if arm == "trainable" else "_")
    ax.set_ylim(-1.15, 0.45)
    sph4 = dr["sphere"]["4.0"]
    if sph4 < -1.1:
        ax.annotate(f"sphere @4x: {sph4:.2f}", xy=(4.0, -1.12), xytext=(2.2, -0.95), fontsize=8.5,
                    color=COL["sphere"], arrowprops=dict(arrowstyle="->", color=COL["sphere"], lw=1))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("gate multiplier")
    ax.set_ylabel("paired \u0394 logit(text day) vs SAME item at gate 0")
    ax.set_title("Paired delta vs off (the same items, gate 0):\ntrained arms respond, controls flat",
                 fontsize=9.5)
    ax.legend(fontsize=7.8, frameon=False, loc="lower left")
    ax.tick_params(labelsize=8.5)
    fig.suptitle("Dose\u2013response on day-mention items (n=280, paired within item)", fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_dose_response.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3: counterfactual flips
def fig3():
    conds = [("cf_swap_onehot_near", "swap near\n(X→X+1)"),
             ("cf_swap_onehot_far", "swap far\n(X→X+3)"),
             ("cf_dose_onehot_far@2", "swap far\n@2× gate"),
             ("cf_dose_onehot_far@4", "swap far\n@4× gate")]
    x = np.arange(len(conds))
    w = 0.24
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for j, arm in enumerate(ARMS):
        ys = [C["conditions"][c][arm]["flip_rate"] for c, _ in conds]
        ax.bar(x + (j - 1) * w, ys, w, color=COL[arm], label=LBL[arm])
    # control band per condition (min..max over the 3 baseline controls)
    for i, (c, _) in enumerate(conds):
        vals = [C["conditions"][c][f"baseline_{a}"]["flip_rate"] for a in ARMS]
        ax.fill_between([x[i] - 0.42, x[i] + 0.42], min(vals), max(vals),
                        color="k", alpha=0.13, zorder=0,
                        label="control range" if i == 0 else "_")
    ax.set_xticks(x, [t for _, t in conds], fontsize=8.5)
    ax.set_ylabel("flip rate  (answer moves text-day → injected-day)")
    ax.set_title("Counterfactual injection rarely overrides the text\n"
                 "(flip rates vs untrained-control band; n=280/condition)", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False)
    ax.tick_params(labelsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_counterfactual.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 4: capability on/off
def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    panels = [("core_metric", "CORE (22 tasks)", None),
              ("weekday_accuracy", "weekday_v1 accuracy (n=422)", 1 / 7),
              ("weekday_answer_ce", "weekday_v1 answer CE (lower = better)", None)]
    for ax, (m, title, chance) in zip(axes, panels):
        base = v(m, "baseline", "on")
        xs = np.arange(len(ARMS))
        for i, arm in enumerate(ARMS):
            on, off = v(m, arm, "on"), v(m, arm, "off")
            ax.plot([i, i], [off, on], color=COL[arm], lw=1.5, zorder=2)
            ax.scatter([i], [on], color=COL[arm], s=70, zorder=3)
            ax.scatter([i], [off], facecolor="white", edgecolor=COL[arm], s=70, lw=1.8, zorder=3)
        ax.axhline(base, color=COL["baseline"], ls="--", lw=1.2)
        ax.text(2.35, base, "baseline", color=COL["baseline"], fontsize=8, va="center")
        if chance:
            ax.axhline(chance, color="k", ls=":", lw=1)
            ax.text(2.35, chance, "chance", fontsize=8, va="center")
        if m == "weekday_accuracy":  # binomial SE band around baseline
            se = np.sqrt(base * (1 - base) / 422)
            ax.fill_between([-0.5, 2.5], base - se, base + se, color="k", alpha=0.08, zorder=0)
        ax.set_xticks(xs, [a for a in ARMS], fontsize=8.5)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(-0.5, 2.9)
        ax.tick_params(labelsize=8)
    axes[0].scatter([], [], color="k", s=60, label="ON")
    axes[0].scatter([], [], facecolor="white", edgecolor="k", s=60, lw=1.6, label="OFF")
    axes[0].legend(fontsize=8.5, frameon=False, loc="lower left")
    fig.suptitle("Capability on/off: CORE unaffected by inference-time injection; weekday accuracy at floor "
                 "(±1 SE band); answer-CE consistently improves with injection ON", fontsize=10.5, y=1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_capability.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 5: implant null
def fig5():
    conds = ["implant_off", "implant_zeroacts_on", "implant_onehot@1", "implant_onehot@2",
             "implant_onehot@4", "implant_emp@1"]
    x = np.arange(len(conds))
    w = 0.24
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for j, arm in enumerate(ARMS):
        ys = [C["conditions"][c][arm]["mean_p_cf"] for c in conds]
        ax.bar(x + (j - 1) * w, ys, w, color=COL[arm], label=LBL[arm])
    ax.axhline(1 / 7, color="k", ls=":", lw=1.2)
    ax.text(len(conds) - 0.4, 1 / 7 + 0.004, "chance (1/7)", fontsize=8.5)
    ax.set_xticks(x, [c.replace("implant_", "") for c in conds], fontsize=8, rotation=15)
    ax.set_ylabel("P(injected day) at answer")
    ax.set_ylim(0, 0.25)
    ax.set_title("Implant (day-free contexts): exact chance-level null everywhere —\n"
                 "too clean; verify readout wiring before citing", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False)
    ax.tick_params(labelsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_implant.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)




# ---------------------------------------------------------------- fig 6/7: open-ended generation
def _load_opengen():
    return json.load(open(os.path.join(RES, "opengen_summary.json")))


def fig6():
    og = _load_opengen()
    nat = og["natural_propensity_FIRST"]
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    arms4 = ["baseline", "trainable", "sphere", "orthogonal"]
    x = np.arange(7)
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for j, arm in enumerate(arms4):
        r = nat[arm]["per_day_base_rate"]
        ax.bar(x + (j - 1.5) * w, [r[d] for d in days], w, color=COL[arm],
               label=f"{arm} (mentions a day in {100*nat[arm]['day_mention_rate']:.1f}% of samples)")
    ax.set_xticks(x, [d[:3] for d in days])
    ax.set_ylabel("P(first day mentioned = d) per sample")
    ax.set_title("Natural propensity (NO injection): free generation from 30 day-free prompts,\n"
                 "510 samples/arm — day mentions are rare and near-uniform", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_natural_propensity.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig7():
    og = _load_opengen()
    eff = og["injection_effects_vs_none"]
    conds = ["inject_all/onehot/@1", "inject_all/onehot/@2", "inject_all/onehot/@4",
             "inject_all/empirical/@1",
             "inject_frontier/onehot/@1", "inject_frontier/onehot/@2", "inject_frontier/onehot/@4",
             "inject_last/onehot/@1", "inject_last/onehot/@2", "inject_last/onehot/@4"]
    rows = ["trainable", "sphere", "orthogonal", "baseline_trainable", "baseline_sphere", "baseline_orthogonal"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.7))
    mats = [np.array([[eff[r][c]["mean_d_p_first_eq_Y_vs_none"] for c in conds] for r in rows]),
            np.array([[eff[r][c]["mean_logit_p_Y"] - 1 / 7 for c in conds] for r in rows])]
    titles = ["Δ P(first generated day = injected Y) vs no-injection",
              "logit-readout P(Y) − chance (1/7) at first position"]
    for ax, M, t, vmax in zip(axes, mats, titles, [0.02, 0.05]):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(conds)),
                      [c.replace("inject_", "").replace("/onehot", "").replace("/empirical", " emp")
                       for c in conds], rotation=40, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(rows)),
                      [r.replace("baseline_", "ctrl:") for r in rows], fontsize=8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i,j]:+.3f}", ha="center", va="center", fontsize=6.3,
                        color="black")
        ax.set_title(t, fontsize=9.5)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("Injection in DAY-FREE contexts does not move free generation or the logit readout — "
                 "every cell ≈ 0 (n=210 template×day combos/cell, 16 samples each)", fontsize=10.5, y=1.06)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_opengen_effects.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print("wrote:", sorted(os.listdir(OUT)))
