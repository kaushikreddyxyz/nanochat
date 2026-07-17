#!/usr/bin/env python3
"""Structural analysis of the TRAINABLE weekday-injection direction matrix.

After 2,520 AdamW steps with a trainable 7x768 direction (orthonormal init,
seed 1337), what geometry did the optimizer sculpt? This script downloads the
checkpoint, extracts D, and measures: (a) the 7x7 inter-day cosine geometry,
(b) PCA / effective rank, (c) circular (calendar-ring) structure, (d) drift
from init, each against three references (init/orthogonal, sphere arm, gemma).

Deterministic, CPU-only, $0. Run from the nanochat repo root:
    python3 runs/weekdays/direction_analysis/analyze_direction.py

Writes (all under runs/weekdays/direction_analysis/):
    direction_report.json   every number behind the figures
    figures/*.png           six matplotlib (Agg) figures

CHANNEL-ORDER TRAP (bitten before): D's 7 rows are in STORE order
(friday,monday,saturday,sunday,thursday,tuesday,wednesday == climbmix cols
47..53). CALENDAR order (monday..sunday) is different; every circular /
calendar-ordered quantity remaps store-row -> calendar position EXPLICITLY via
CAL_ROWS and is asserted.
"""
from __future__ import annotations

import io
import gzip
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import subspace_angles

# nanochat repo root (this file: runs/weekdays/direction_analysis/) onto sys.path
# so `import nanochat` works when run by path from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from nanochat.injection.sites import orthonormal_direction

# --------------------------------------------------------------------------- #
# Pinned constants (store order, calendar mapping, campaign palette).
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
WEEKDAYS_DIR = os.path.dirname(HERE)
FIG_DIR = os.path.join(HERE, "figures")

HF_REPO = "kaushikreddyxyz/weekday-geometry-d12"
ARM = "trainable"
STEP = 2520
R, N_EMBD, SEED = 7, 768, 1337
GATE = 0.0273

# STORE order == D row order == climbmix cols 47..53.
STORE = ["friday", "monday", "saturday", "sunday", "thursday", "tuesday", "wednesday"]
STORE_IDX = {"friday": 47, "monday": 48, "saturday": 49, "sunday": 50,
             "thursday": 51, "tuesday": 52, "wednesday": 53}
# CALENDAR order (monday=0 ... sunday=6) -- the circle phase order.
CALENDAR = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
CAL_POS = {d: i for i, d in enumerate(CALENDAR)}
CAL_ROWS = [STORE.index(d) for d in CALENDAR]      # store-row index of each calendar day
CAL_OF_ROW = [CAL_POS[d] for d in STORE]           # calendar pos of each store row
LBL3 = {d: d[:3].capitalize() for d in STORE}      # Mon, Tue, ...
CAL_LABELS = [LBL3[d] for d in CALENDAR]           # [Mon,Tue,Wed,Thu,Fri,Sat,Sun]

# Sanity: the remap is an involution-free bijection and inverts cleanly.
assert [STORE[r] for r in CAL_ROWS] == CALENDAR
assert CAL_ROWS == [1, 5, 6, 4, 0, 2, 3], CAL_ROWS

PAL = {"trainable": "#1f77b4", "sphere": "#2ca02c",
       "orthogonal": "#d62728", "init": "#7f7f7f", "gemma": "#9467bd"}
WEEKEND = {"saturday", "sunday"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def unit_rows(M):
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def cosmat(M):
    U = unit_rows(np.asarray(M, np.float64))
    return U @ U.T


def to_cal(M):
    """Reorder a store-order [7,7] matrix (or [7] vector) into calendar order."""
    M = np.asarray(M)
    if M.ndim == 2:
        return M[np.ix_(CAL_ROWS, CAL_ROWS)]
    return M[CAL_ROWS]


def circ_dist(a, b, n=7):
    d = abs(a - b)
    return min(d, n - d)


def offdiag(M):
    return M[~np.eye(M.shape[0], dtype=bool)]


def profile(cos, order_of_row=CAL_OF_ROW):
    """cosine-vs-calendar-circular-distance: mean/std/values at dist 1,2,3."""
    buckets = {1: [], 2: [], 3: []}
    for i in range(7):
        for j in range(i + 1, 7):
            buckets[circ_dist(order_of_row[i], order_of_row[j])].append(float(cos[i, j]))
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "n": len(v), "values": v} for k, v in buckets.items()}


def eff_rank(sv):
    """Participation ratio effective rank from singular values."""
    e = np.asarray(sv, np.float64) ** 2
    return float(e.sum() ** 2 / (e ** 2).sum())


def sign_fix(V):
    """Deterministic sign for singular vectors: largest-|component| made +."""
    out = V.copy()
    for i in range(out.shape[0]):
        j = np.argmax(np.abs(out[i]))
        if out[i, j] < 0:
            out[i] *= -1.0
    return out


def fit_rho(prof):
    """Uniform-calendar-phase shared+circle fit (closed form, build_manifold.py).
    model_cos(k) = rho + (1-rho)*cos(2*pi*k/7); alpha=sqrt(rho), beta=sqrt(1-rho).
    Linear LS in rho over the 3 mean points; residual reported vs all 21 pairs."""
    ks = [1, 2, 3]
    c = {k: float(np.cos(2 * np.pi * k / 7)) for k in ks}
    g = {k: prof[k]["mean"] for k in ks}
    a = np.array([1.0 - c[k] for k in ks])
    b = np.array([g[k] - c[k] for k in ks])
    rho_raw = float((a @ b) / (a @ a))
    rho = float(np.clip(rho_raw, 0.0, 1.0))
    fitted = {k: rho + (1.0 - rho) * c[k] for k in ks}
    allp, allpred = [], []
    for k in ks:
        for v in prof[k]["values"]:
            allp.append(v); allpred.append(fitted[k])
    allp, allpred = np.array(allp), np.array(allpred)
    ss_res = float(np.sum((allp - allpred) ** 2))
    ss_tot = float(np.sum((allp - allp.mean()) ** 2))
    return {"rho_raw": rho_raw, "rho": rho, "alpha": float(np.sqrt(rho)),
            "beta": float(np.sqrt(1.0 - rho)), "c_k": c, "fitted_profile": fitted,
            "profile_max_abs_resid": float(max(abs(fitted[k] - g[k]) for k in ks)),
            "r2_against_21_pairs": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")}


def shared_circle_fit(U):
    """Free-phase shared+circle model fit to UNIT rows U [7,768]:
        u0 = normalized mean direction (shared 'weekday-ness')
        residual R = U - (U.u0) u0 ; its top-2 right singular vecs = circle plane
        phase_d = atan2(coord2, coord1)   (the FREE per-day phase)
    Returns the fit, its rank-3 reconstruction residual, and the calendar-order
    verdict on the recovered phase ordering (forward, reverse, star-windings)."""
    u0 = U.mean(0); u0 = u0 / np.linalg.norm(u0)
    alpha = U @ u0                                   # per-row shared projection
    Rres = U - np.outer(alpha, u0)
    Uu, Sv, Vt = np.linalg.svd(Rres, full_matrices=False)
    Vt = sign_fix(Vt)
    u1, u2 = Vt[0], Vt[1]
    c1, c2 = U @ u1, U @ u2
    phase = np.arctan2(c2, c1)                        # free phase per store row
    beta = np.sqrt(c1 ** 2 + c2 ** 2)
    recon = np.outer(alpha, u0) + np.outer(c1, u1) + np.outer(c2, u2)
    resid = U - recon
    r2 = float(1.0 - (resid ** 2).sum() / (U ** 2).sum())

    # calendar verdict: sort store rows by phase, read their calendar positions.
    order_rows = list(np.argsort(phase % (2 * np.pi)))
    order_cal = [CAL_OF_ROW[r] for r in order_rows]

    def is_cyclic_rotation(seq, target):
        n = len(target)
        for s in range(n):
            if all(seq[i] == target[(s + i) % n] for i in range(n)):
                return True
        return False

    fwd = list(range(7)); rev = list(range(6, -1, -1))
    star2 = [(2 * i) % 7 for i in range(7)]           # winding-by-2 heptagram
    star3 = [(3 * i) % 7 for i in range(7)]
    verdict = {
        "matches_calendar_forward": bool(is_cyclic_rotation(order_cal, fwd)),
        "matches_calendar_reverse": bool(is_cyclic_rotation(order_cal, rev)),
        "matches_star_winding2": bool(is_cyclic_rotation(order_cal, star2)),
        "matches_star_winding3": bool(is_cyclic_rotation(order_cal, star3)),
    }
    verdict["matches_any_clean_ring"] = bool(any(verdict.values()))
    return {
        "u0_shared": u0, "alpha_shared_per_row": alpha, "beta_circle_per_row": beta,
        "phase_rad_per_row": phase, "plane_u1": u1, "plane_u2": u2,
        "coord1": c1, "coord2": c2, "rank3_r2": r2,
        "rank3_resid_norm_per_row": np.linalg.norm(resid, axis=1),
        "phase_order_rows": order_rows, "phase_order_calpos": order_cal,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# load: imitate harness._download + _load_tensorfile (gz + _orig_mod strip),
# pull the direction tensor directly (no GPT build needed for a weight analysis).
# --------------------------------------------------------------------------- #
def load_learned_direction():
    from huggingface_hub import hf_hub_download
    stem = f"{ARM}/model_{STEP:06d}.pt"
    try:
        gz = hf_hub_download(HF_REPO, stem + ".gz", repo_type="model")
        with open(gz, "rb") as f:
            sd = torch.load(io.BytesIO(gzip.decompress(f.read())), map_location="cpu")
    except Exception:
        p = hf_hub_download(HF_REPO, stem, repo_type="model")
        sd = torch.load(p, map_location="cpu")
    meta_p = hf_hub_download(HF_REPO, f"{ARM}/meta_{STEP:06d}.json", repo_type="model")
    meta = json.load(open(meta_p))
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    D = sd["injection_sites.weekdays.direction"].float().numpy().astype(np.float64)
    gate = float(sd["injection_sites.weekdays.gate"].float().item())
    assert D.shape == (R, N_EMBD), D.shape
    cfg = meta["injection_sites_config"][0]
    assert cfg["trainable_direction"] and cfg["direction_init"] == "orthonormal" \
        and cfg["direction_seed"] == SEED and cfg["r"] == R
    return D, gate, meta


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _heat(ax, M, title, labels):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(7):
        for j in range(7):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.55 else "black")
    return im


def fig_cosine_matrices(cos_learned, cos_gemma, cos_sphere):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    _heat(axes[0], to_cal(cos_learned), "trainable (learned D)", CAL_LABELS)
    _heat(axes[1], to_cal(cos_gemma), "gemma-2b L8 (measured)", CAL_LABELS)
    im = _heat(axes[2], to_cal(cos_sphere), "sphere arm (frozen D)", CAL_LABELS)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="cosine")
    fig.suptitle("7x7 inter-day cosine, CALENDAR order (Mon..Sun)", fontsize=12)
    p = os.path.join(FIG_DIR, "fig1_cosine_matrices.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def fig_norms_drift(row_norms, drift_cos, princ_angles):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(7)
    # (a) row norms in calendar order, store-idx annotated
    rn_cal = to_cal(row_norms)
    axes[0].bar(x, rn_cal, color=PAL["trainable"])
    for i, d in enumerate(CALENDAR):
        axes[0].text(i, rn_cal[i], f"c{STORE_IDX[d]}", ha="center", va="bottom", fontsize=7)
    axes[0].axhline(1.0, color=PAL["init"], ls="--", lw=1, label="init norm = 1")
    axes[0].set_xticks(x); axes[0].set_xticklabels(CAL_LABELS)
    axes[0].set_ylabel("||row||"); axes[0].set_title("(a) learned row norms (init was unit)")
    axes[0].legend(fontsize=8)
    # (b) per-row cos(learned, init)
    dc_cal = to_cal(drift_cos)
    axes[1].bar(x, dc_cal, color=PAL["trainable"])
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylim(-1, 1)
    axes[1].set_xticks(x); axes[1].set_xticklabels(CAL_LABELS)
    axes[1].set_ylabel("cosine"); axes[1].set_title("(b) cos(learned row, init row) -- drift")
    # (c) principal angles between 7-dim learned & init subspaces
    axes[2].bar(np.arange(7), princ_angles, color=PAL["init"])
    axes[2].axhline(90, color="k", ls=":", lw=1, label="90 = orthogonal")
    axes[2].set_ylim(0, 95)
    axes[2].set_xlabel("principal angle #"); axes[2].set_ylabel("degrees")
    axes[2].set_title("(c) learned vs init subspace angles")
    axes[2].legend(fontsize=8)
    p = os.path.join(FIG_DIR, "fig2_norms_drift.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def fig_scree(ev_raw, ev_cen):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    k = np.arange(1, 8)
    for ax, ev, title in [(axes[0], ev_raw, "raw rows"),
                          (axes[1], ev_cen, "mean-centered rows")]:
        ax.bar(k, ev, color=PAL["trainable"], alpha=0.8, label="per-PC")
        ax.plot(k, np.cumsum(ev), "-o", color=PAL["orthogonal"], label="cumulative")
        ax.axhline(1.0, color="k", lw=0.5)
        for i in (1, 2):     # annotate cumulative at k=2,3
            ax.annotate(f"k{i+1}: {np.cumsum(ev)[i]:.2f}", (k[i], np.cumsum(ev)[i]),
                        textcoords="offset points", xytext=(0, 8), fontsize=8)
        ax.set_xlabel("component k"); ax.set_ylabel("explained variance share")
        ax.set_title(f"PCA scree -- {title}"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    p = os.path.join(FIG_DIR, "fig3_pca_scree.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def fig_circle_projection(coords_pc, ev_cen):
    """Scatter mean-centered rows on PC1-PC2 and PC2-PC3, connected calendar order."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    cmap = plt.get_cmap("twilight")
    for ax, (a, b) in zip(axes, [(0, 1), (1, 2)]):
        pts = coords_pc[:, [a, b]]
        loop = CAL_ROWS + [CAL_ROWS[0]]              # close the ring in calendar order
        ax.plot(pts[loop, 0], pts[loop, 1], "-", color=PAL["init"], lw=1, zorder=1)
        for pos, r in enumerate(CAL_ROWS):
            ax.scatter(*pts[r], color=cmap(pos / 7.0), s=120, zorder=2, edgecolor="k", lw=0.5)
            ax.annotate(LBL3[STORE[r]], pts[r], textcoords="offset points",
                        xytext=(6, 4), fontsize=9)
        ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
        ax.set_xlabel(f"PC{a+1} ({ev_cen[a]*100:.0f}%)")
        ax.set_ylabel(f"PC{b+1} ({ev_cen[b]*100:.0f}%)")
        ax.set_title(f"mean-centered rows, PC{a+1}-PC{b+1}")
        ax.set_aspect("equal", "datalim")
    fig.suptitle("Ring test: days connected in CALENDAR order (Mon->Sun)", fontsize=12)
    p = os.path.join(FIG_DIR, "fig4_circle_projections.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def fig_circular_profile(pr_learned, pr_gemma, pr_sphere):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ks = [1, 2, 3]
    for name, pr, col in [("trainable (learned)", pr_learned, PAL["trainable"]),
                          ("gemma-2b L8", pr_gemma, PAL["gemma"]),
                          ("sphere arm", pr_sphere, PAL["sphere"])]:
        m = [pr[k]["mean"] for k in ks]
        s = [pr[k]["std"] for k in ks]
        ax.errorbar(ks, m, yerr=s, marker="o", capsize=4, color=col, label=name)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(ks); ax.set_xlabel("calendar circular distance")
    ax.set_ylabel("mean inter-day cosine")
    ax.set_title("Cosine vs calendar distance (nearer days -> higher = ring)")
    ax.legend()
    p = os.path.join(FIG_DIR, "fig5_circular_profile.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def fig_free_phases(fit):
    """Free-phase ring (left) vs ideal uniform calendar ring (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), subplot_kw={"aspect": "equal"})
    cmap = plt.get_cmap("twilight")
    # left: learned free phases on the unit circle (radius = per-row circle beta, normalized)
    ph = fit["phase_rad_per_row"]
    beta = fit["beta_circle_per_row"]
    rad = beta / beta.max()
    for r in range(7):
        x, y = rad[r] * np.cos(ph[r]), rad[r] * np.sin(ph[r])
        axes[0].plot([0, x], [0, y], color=PAL["init"], lw=0.6, zorder=1)
        axes[0].scatter(x, y, color=cmap(CAL_OF_ROW[r] / 7.0), s=140, zorder=2,
                        edgecolor="k", lw=0.5)
        axes[0].annotate(LBL3[STORE[r]], (x, y), textcoords="offset points",
                         xytext=(6, 4), fontsize=9)
    axes[0].add_patch(plt.Circle((0, 0), 1, fill=False, color="grey", ls=":"))
    axes[0].set_title("learned FREE phases (color = calendar pos)")
    axes[0].set_xlim(-1.2, 1.2); axes[0].set_ylim(-1.2, 1.2)
    # right: ideal uniform calendar phases
    for pos, d in enumerate(CALENDAR):
        th = 2 * np.pi * pos / 7
        x, y = np.cos(th), np.sin(th)
        axes[1].scatter(x, y, color=cmap(pos / 7.0), s=140, edgecolor="k", lw=0.5)
        axes[1].annotate(LBL3[d], (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    loop = list(range(7)) + [0]
    axes[1].plot([np.cos(2 * np.pi * p / 7) for p in loop],
                 [np.sin(2 * np.pi * p / 7) for p in loop], color=PAL["sphere"], lw=1)
    axes[1].set_title("ideal uniform calendar ring")
    axes[1].set_xlim(-1.2, 1.2); axes[1].set_ylim(-1.2, 1.2)
    v = fit["verdict"]
    fig.suptitle(f"Free-phase order matches a clean calendar ring: "
                 f"{'YES' if v['matches_any_clean_ring'] else 'NO'}  "
                 f"(order by calpos: {fit['phase_order_calpos']})", fontsize=11)
    p = os.path.join(FIG_DIR, "fig6_free_phases.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    np.set_printoptions(suppress=True, precision=4)

    # --- load learned D + references ---
    D, gate, meta = load_learned_direction()
    init = orthonormal_direction(R, N_EMBD, SEED).numpy().astype(np.float64)   # == orthogonal arm
    orth_npz = np.load(os.path.join(WEEKDAYS_DIR, "direction_orthogonal.npz"))["D"].astype(np.float64)
    init_repro_err = float(np.abs(init - orth_npz).max())
    assert init_repro_err < 1e-6, init_repro_err
    sphere = np.load(os.path.join(WEEKDAYS_DIR, "direction_sphere.npz"))["D"].astype(np.float64)
    gemma_cos = np.array(json.load(open(os.path.join(WEEKDAYS_DIR, "manifold_validation.json")))
                         ["gemma_cosine_matrix"], np.float64)

    Un = unit_rows(D)
    cos_learned = Un @ Un.T
    cos_sphere = cosmat(sphere)
    row_norms = np.linalg.norm(D, axis=1)

    # --- (1) cosine geometry ---
    od = offdiag(cos_learned)
    cos_stats = {"offdiag_mean": float(od.mean()), "offdiag_std": float(od.std()),
                 "offdiag_min": float(od.min()), "offdiag_max": float(od.max())}

    # --- (2) drift from init + subspace rotation ---
    init_u = unit_rows(init)
    drift_cos = np.sum(Un * init_u, axis=1)                    # per-row (same store order)
    angles = np.degrees(subspace_angles(D.T, init.T))          # 7 principal angles
    subspace_overlap = float(np.mean(np.cos(np.radians(angles)) ** 2))

    # --- (3) PCA (raw + mean-centered) ---
    sv_raw = np.linalg.svd(D, compute_uv=False)
    ev_raw = (sv_raw ** 2 / (sv_raw ** 2).sum())
    mu = D.mean(0)
    Dc = D - mu
    Uc, sv_cen, Vt_cen = np.linalg.svd(Dc, full_matrices=False)
    ev_cen = (sv_cen ** 2 / (sv_cen ** 2).sum())
    coords_pc = Uc * sv_cen                                    # row scores in PC space
    shared_norm = float(np.linalg.norm(mu))
    shared_energy_frac = float(shared_norm ** 2 * R / (np.linalg.norm(D) ** 2))  # ||mean||^2*R / total

    # --- (4) circular structure ---
    pr_learned = profile(cos_learned)
    pr_gemma = profile(gemma_cos)
    pr_sphere = profile(cos_sphere)
    rho_fit = fit_rho(pr_learned)                              # uniform-phase closed form
    sc_fit = shared_circle_fit(Un)                             # free phases

    # --- (5) extras: sphere-u0 alignment, weekend clustering, effective rank ---
    sphere_u0 = unit_rows(sphere).mean(0); sphere_u0 /= np.linalg.norm(sphere_u0)
    u0_align = float(abs(sc_fit["u0_shared"] @ sphere_u0))
    # circle-plane alignment: learned plane (u1,u2) vs sphere plane (top-2 PC of centered sphere)
    sph_c = sphere - sphere.mean(0)
    _, _, sph_vt = np.linalg.svd(sph_c, full_matrices=False)
    plane_angles = np.degrees(subspace_angles(np.stack([sc_fit["plane_u1"], sc_fit["plane_u2"]]).T,
                                              sph_vt[:2].T))
    we_idx = [STORE.index(d) for d in WEEKEND]
    wd_idx = [i for i in range(7) if i not in we_idx]
    weekend_cos = float(cos_learned[we_idx[0], we_idx[1]])
    wd_pairs = [cos_learned[i, j] for a, i in enumerate(wd_idx) for j in wd_idx[a + 1:]]
    we_wd_pairs = [cos_learned[i, j] for i in we_idx for j in wd_idx]

    # --- assemble report ---
    report = {
        "meta": {"hf_repo": HF_REPO, "arm": ARM, "step": STEP, "gate_scalar": gate,
                 "val_bpb": meta.get("val_bpb"), "store_order": STORE, "store_idx": STORE_IDX,
                 "calendar_order": CALENDAR, "calendar_rows": CAL_ROWS,
                 "cal_pos_of_store_row": CAL_OF_ROW,
                 "init_reproduces_orthogonal_npz_maxabs": init_repro_err},
        "cosine": {
            "learned_store": cos_learned.tolist(), "learned_calendar": to_cal(cos_learned).tolist(),
            "gemma_store": gemma_cos.tolist(), "gemma_calendar": to_cal(gemma_cos).tolist(),
            "sphere_store": cos_sphere.tolist(), "sphere_calendar": to_cal(cos_sphere).tolist(),
            "learned_offdiag_stats": cos_stats,
        },
        "row_norms": {
            "learned_store": row_norms.tolist(),
            "learned_by_day": {d: float(row_norms[i]) for i, d in enumerate(STORE)},
            "min": float(row_norms.min()), "max": float(row_norms.max()),
            "mean": float(row_norms.mean()), "init_norm": 1.0,
            "growth_ratio_mean": float(row_norms.mean()),  # init rows were unit norm
        },
        "drift_from_init": {
            "per_row_cos_store": drift_cos.tolist(),
            "per_row_cos_by_day": {d: float(drift_cos[i]) for i, d in enumerate(STORE)},
            "mean_abs_cos": float(np.abs(drift_cos).mean()),
            "principal_angles_deg": angles.tolist(),
            "subspace_mean_cos2_overlap": subspace_overlap,
        },
        "pca": {
            "raw_explained_var": ev_raw.tolist(),
            "raw_cumulative": np.cumsum(ev_raw).tolist(),
            "centered_explained_var": ev_cen.tolist(),
            "centered_cumulative": np.cumsum(ev_cen).tolist(),
            "centered_k1": float(ev_cen[0]), "centered_k2_cum": float(np.cumsum(ev_cen)[1]),
            "centered_k3_cum": float(np.cumsum(ev_cen)[2]),
            "raw_k1": float(ev_raw[0]), "raw_k2_cum": float(np.cumsum(ev_raw)[1]),
            "raw_k3_cum": float(np.cumsum(ev_raw)[2]),
            "shared_mean_norm": shared_norm, "mean_row_norm": float(row_norms.mean()),
            "shared_energy_fraction": shared_energy_frac,
            "eff_rank_raw": eff_rank(sv_raw), "eff_rank_centered": eff_rank(sv_cen),
        },
        "circular": {
            "profile_learned": pr_learned, "profile_gemma": pr_gemma, "profile_sphere": pr_sphere,
            "uniform_phase_fit": rho_fit,
            "free_phase_fit": {
                "phase_deg_by_day": {d: float(np.degrees(sc_fit["phase_rad_per_row"][i]) % 360)
                                     for i, d in enumerate(STORE)},
                "alpha_shared_by_day": {d: float(sc_fit["alpha_shared_per_row"][i])
                                        for i, d in enumerate(STORE)},
                "beta_circle_by_day": {d: float(sc_fit["beta_circle_per_row"][i])
                                       for i, d in enumerate(STORE)},
                "phase_order_days": [STORE[r] for r in sc_fit["phase_order_rows"]],
                "phase_order_calpos": sc_fit["phase_order_calpos"],
                "rank3_r2": sc_fit["rank3_r2"],
                "rank3_resid_norm_per_row": sc_fit["rank3_resid_norm_per_row"].tolist(),
                "verdict": sc_fit["verdict"],
            },
        },
        "extras": {
            "learned_shared_vs_sphere_shared_cos": u0_align,
            "learned_plane_vs_sphere_plane_angles_deg": plane_angles.tolist(),
            "weekend_sat_sun_cos": weekend_cos,
            "mean_weekday_weekday_cos": float(np.mean(wd_pairs)),
            "mean_weekend_weekday_cos": float(np.mean(we_wd_pairs)),
            "monday_shared_alpha": float(sc_fit["alpha_shared_per_row"][STORE.index("monday")]),
        },
    }
    with open(os.path.join(HERE, "direction_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # --- figures ---
    figs = [
        fig_cosine_matrices(cos_learned, gemma_cos, cos_sphere),
        fig_norms_drift(row_norms, drift_cos, angles),
        fig_scree(ev_raw, ev_cen),
        fig_circle_projection(coords_pc, ev_cen),
        fig_circular_profile(pr_learned, pr_gemma, pr_sphere),
        fig_free_phases(sc_fit),
    ]

    # --- console headline ---
    print("=" * 70)
    print(f"learned D: rows {D.shape}, norms {row_norms.min():.1f}..{row_norms.max():.1f} "
          f"(init unit); gate {gate}")
    print(f"off-diag cosine: mean {cos_stats['offdiag_mean']:+.4f} "
          f"[{cos_stats['offdiag_min']:+.3f}, {cos_stats['offdiag_max']:+.3f}]")
    print(f"drift: per-row cos(learned,init) mean|.| {np.abs(drift_cos).mean():.3f}; "
          f"subspace angles {angles.min():.1f}-{angles.max():.1f} deg, overlap {subspace_overlap:.4f}")
    print(f"PCA centered: k1 {ev_cen[0]:.3f}, k2cum {np.cumsum(ev_cen)[1]:.3f}, "
          f"k3cum {np.cumsum(ev_cen)[2]:.3f}; eff_rank raw {eff_rank(sv_raw):.2f}")
    print(f"shared/mean norm {shared_norm:.1f} vs row {row_norms.mean():.1f} "
          f"(energy frac {shared_energy_frac:.3f})")
    print(f"circular profile learned d1/d2/d3: "
          f"{pr_learned[1]['mean']:.4f}/{pr_learned[2]['mean']:.4f}/{pr_learned[3]['mean']:.4f} "
          f"(gemma {pr_gemma[1]['mean']:.3f}/{pr_gemma[2]['mean']:.3f}/{pr_gemma[3]['mean']:.3f})")
    print(f"uniform-phase fit: rho {rho_fit['rho']:.3f} alpha {rho_fit['alpha']:.3f} "
          f"beta {rho_fit['beta']:.3f} R2vs21 {rho_fit['r2_against_21_pairs']:.3f}")
    print(f"free-phase order (calpos): {sc_fit['phase_order_calpos']} -> "
          f"clean ring: {sc_fit['verdict']['matches_any_clean_ring']}")
    print("figures:", [os.path.basename(p) for p in figs])
    print("=" * 70)


if __name__ == "__main__":
    main()
