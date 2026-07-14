#!/usr/bin/env python3
"""Experiment 3 (weekday-geometry): build a FROZEN injection direction whose 7
rows form a REALISTIC weekday manifold -- a 1-sphere (circle) embedded in a
low-dim subspace of nanochat's residual stream, with positive inter-day cosine
similarity calibrated to gemma-2-2b's ACTUAL measured weekday geometry at L8.

Pipeline (deterministic, offline, CPU, $0):
  1. MEASURE gemma's weekday geometry at L8. Replicate the exact raw-space read
     direction recipe from attribution/verify_reconstruction.py's Geometry:
         V = W / nat_std   (raw-space read dirs, [K, D])
         u_c = V_c / ||V_c||   (unit rows)
     for the 7 weekday concepts. L8 is axis-0 index 1 of the [3, K, D] arrays
     (store layers == [6, 8, 14]). Compute the 7x7 cosine matrix, then the
     cosine-vs-circular-distance profile in CALENDAR order.
  2. FIT the realistic manifold. Model each day's unit row as
         m_d = alpha*u0 + beta*(cos(theta_d)*u1 + sin(theta_d)*u2)
     with u0,u1,u2 orthonormal (seeded random in R^n_embd) and
         theta_d = 2*pi * (CALENDAR position of day d) / 7.
     Rows are unit norm iff alpha^2 + beta^2 == 1, giving the closed-form cosine
         cos(m_i, m_j) = alpha^2 + beta^2 * cos(theta_i - theta_j)
                       = rho + (1 - rho) * cos(2*pi*k/7)     (rho = alpha^2)
     at circular distance k. Least-squares fit rho (== alpha^2) to gemma's
     3-point mean profile.
  3. CHANNEL-ORDER TRAP: the direction matrix rows are in STORE order
     (friday first: friday=47, monday=48, saturday=49, sunday=50, thursday=51,
     tuesday=52, wednesday=53 in main_block_concepts), but theta_d uses CALENDAR
     position (monday=0 ... sunday=6). The two orders are DIFFERENT; the mapping
     is applied explicitly (CAL_POS) and tested. A wrong mapping silently
     destroys the experiment.
  4. EMIT direction_sphere.npz (D [7, n_embd] float32, store-channel order) and
     manifold_validation.json.

Run: python runs/weekdays/build_manifold.py   (from the nanochat repo root, or
anywhere -- attribution/out is discovered by walking up).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Pinned config (shared across the 4-run weekday-geometry experiment).
# --------------------------------------------------------------------------- #
N_EMBD = 768            # nanochat depth=12, aspect_ratio=64 -> model_dim=768
SEED = 1337             # seeds the orthonormal (u0,u1,u2) basis in R^N_EMBD
LAYERS = [6, 8, 14]     # store layer axis (probe_set.json "layers")
LI_L8 = 1               # axis-0 index of L8 in the [3,K,D] arrays

# Weekday channel order == STORE order == ascending main_block_concepts index.
WEEKDAY_STORE_ORDER = ["friday", "monday", "saturday", "sunday",
                       "thursday", "tuesday", "wednesday"]
STORE_IDX = {"friday": 47, "monday": 48, "saturday": 49, "sunday": 50,
             "thursday": 51, "tuesday": 52, "wednesday": 53}
# Calendar order (for the circle phase). monday=0 ... sunday=6.
CALENDAR = ["monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday"]
CAL_POS = {d: i for i, d in enumerate(CALENDAR)}

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
def find_attr_out(start: Path) -> Path:
    """Walk up from ``start`` to the superproject's attribution/out."""
    p = start
    for _ in range(8):
        cand = p / "attribution" / "out" / "probe_set.json"
        if cand.exists():
            return cand.parent
        p = p.parent
    raise FileNotFoundError("could not locate attribution/out/probe_set.json above "
                            f"{start} (expected in the oracle-encodings superproject)")


def circular_dist(pos_i: int, pos_j: int, n: int = 7) -> int:
    d = abs(pos_i - pos_j)
    return min(d, n - d)


def measure_gemma_geometry(attr_out: Path):
    """Replicate verify_reconstruction.Geometry for the 7 weekday rows at L8.
    Returns (u_week [7, D] store order, cos_gemma [7,7] store order, profile dict)."""
    meta = json.load(open(attr_out / "probe_set.json"))
    concepts = list(meta["main_block_concepts"])
    assert list(meta["layers"]) == LAYERS, (meta["layers"], LAYERS)

    # Verify the pinned store indices against the actual concept list.
    for d in WEEKDAY_STORE_ORDER:
        assert concepts[STORE_IDX[d]] == d, \
            f"store index {STORE_IDX[d]} is {concepts[STORE_IDX[d]]!r}, expected {d!r}"
        assert concepts.index(d) == STORE_IDX[d], \
            f"{d!r} first appears at index {concepts.index(d)}, pinned {STORE_IDX[d]}"

    arr = np.load(attr_out / "probe_set_arrays.npz")
    W = np.asarray(arr["W"], np.float64)              # [3, K, D] std-space read weights
    nat_std = np.asarray(arr["nat_std"], np.float64)  # [3, D]
    V = W[LI_L8] / nat_std[LI_L8][None, :]            # [K, D] raw-space read dirs @ L8
    vnorm = np.linalg.norm(V, axis=1)                 # [K]
    U = V / vnorm[:, None]                            # [K, D] unit rows

    widx = [STORE_IDX[d] for d in WEEKDAY_STORE_ORDER]   # store-order concept indices
    u_week = U[widx]                                     # [7, D] store order
    cos_gemma = u_week @ u_week.T                        # [7, 7] store order (unit rows)

    # cosine-vs-circular-distance profile, grouped by CALENDAR circular distance.
    cal_of_row = [CAL_POS[d] for d in WEEKDAY_STORE_ORDER]
    buckets = {1: [], 2: [], 3: []}
    for i in range(7):
        for j in range(i + 1, 7):
            k = circular_dist(cal_of_row[i], cal_of_row[j])
            buckets[k].append(float(cos_gemma[i, j]))
    profile = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                   "n": len(v), "values": [float(x) for x in v]}
               for k, v in buckets.items()}
    return u_week, cos_gemma, profile, cal_of_row


def fit_rho(profile: dict):
    """LS fit of rho (== alpha^2, with alpha^2 + beta^2 == 1) to the 3-point mean
    profile. model_cos(k) = rho + (1 - rho)*c_k, c_k = cos(2*pi*k/7).
    Linear in rho: minimize sum_k (rho*(1 - c_k) + c_k - g_k)^2."""
    ks = [1, 2, 3]
    c = {k: float(np.cos(2 * np.pi * k / 7)) for k in ks}
    g = {k: profile[k]["mean"] for k in ks}
    a = np.array([1.0 - c[k] for k in ks])
    b = np.array([g[k] - c[k] for k in ks])
    rho_raw = float((a @ b) / (a @ a))
    rho = float(np.clip(rho_raw, 0.0, 1.0))
    alpha = float(np.sqrt(rho))
    beta = float(np.sqrt(1.0 - rho))
    fitted = {k: rho + (1.0 - rho) * c[k] for k in ks}
    # honesty: residual of the fitted circle model vs ALL 21 individual pairs
    all_pairs, all_pred = [], []
    for k in ks:
        for v in profile[k]["values"]:
            all_pairs.append(v)
            all_pred.append(fitted[k])
    all_pairs = np.array(all_pairs); all_pred = np.array(all_pred)
    ss_res = float(np.sum((all_pairs - all_pred) ** 2))
    ss_tot = float(np.sum((all_pairs - all_pairs.mean()) ** 2))
    r2_pairs = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rho_raw": rho_raw, "rho": rho, "alpha": alpha, "beta": beta,
            "c_k": c, "fitted_profile": fitted,
            "profile_max_abs_resid": float(max(abs(fitted[k] - g[k]) for k in ks)),
            "r2_against_21_pairs": r2_pairs,
            "within_distance_std": {k: profile[k]["std"] for k in ks}}


def build_direction(alpha: float, beta: float, seed: int = SEED, n_embd: int = N_EMBD):
    """D [7, n_embd] float32 in STORE order. Row d = alpha*u0 + beta*(cos th*u1 +
    sin th*u2), th from the day's CALENDAR position. u0,u1,u2 orthonormal (seeded)."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((n_embd, 3)))  # [n_embd, 3], orthonormal cols
    u0, u1, u2 = basis[:, 0], basis[:, 1], basis[:, 2]
    D = np.zeros((7, n_embd), np.float64)
    thetas = np.zeros(7)
    for row, day in enumerate(WEEKDAY_STORE_ORDER):
        th = 2.0 * np.pi * CAL_POS[day] / 7.0
        thetas[row] = th
        m = alpha * u0 + beta * (np.cos(th) * u1 + np.sin(th) * u2)
        D[row] = m / np.linalg.norm(m)   # exact unit norm (guards fp drift)
    return D.astype(np.float32), thetas, basis


def main():
    attr_out = find_attr_out(HERE)
    print(f"[build] attribution/out -> {attr_out}")

    u_week, cos_gemma, profile, cal_of_row = measure_gemma_geometry(attr_out)
    print("[build] gemma weekday cosine profile @ L8 (calendar circular distance):")
    for k in (1, 2, 3):
        print(f"        dist {k}: mean={profile[k]['mean']:.4f} std={profile[k]['std']:.4f} "
              f"(n={profile[k]['n']})")

    fit = fit_rho(profile)
    print(f"[build] fit: rho_raw={fit['rho_raw']:.4f} rho={fit['rho']:.4f} "
          f"alpha={fit['alpha']:.4f} beta={fit['beta']:.4f}")
    print("[build] fitted circle-model profile: " +
          ", ".join(f"d{k}={fit['fitted_profile'][k]:.4f}" for k in (1, 2, 3)))
    print(f"[build] fitted-vs-gemma max |resid| (3-pt): {fit['profile_max_abs_resid']:.4f}; "
          f"circle-model R^2 vs 21 individual pairs: {fit['r2_against_21_pairs']:.4f}")

    D, thetas, basis = build_direction(fit["alpha"], fit["beta"])
    cos_constructed = (D.astype(np.float64) @ D.astype(np.float64).T)

    # constructed profile (same grouping) and cross-checks
    buckets = {1: [], 2: [], 3: []}
    for i in range(7):
        for j in range(i + 1, 7):
            buckets[circular_dist(cal_of_row[i], cal_of_row[j])].append(float(cos_constructed[i, j]))
    constructed_profile = {k: float(np.mean(v)) for k, v in buckets.items()}
    # constructed profile MUST equal the fitted closed-form model to ~fp
    max_constructed_vs_model = max(
        abs(constructed_profile[k] - fit["fitted_profile"][k]) for k in (1, 2, 3))
    # max |constructed - gemma| over off-diagonal (both in store order)
    off = ~np.eye(7, dtype=bool)
    max_constructed_vs_gemma = float(np.max(np.abs(cos_constructed - cos_gemma)[off]))
    unit_err = float(np.max(np.abs(np.diag(cos_constructed) - 1.0)))
    print(f"[build] constructed profile: " +
          ", ".join(f"d{k}={constructed_profile[k]:.4f}" for k in (1, 2, 3)))
    print(f"[build] |constructed - fitted model| max = {max_constructed_vs_model:.2e}; "
          f"row unit-norm max err = {unit_err:.2e}")
    print(f"[build] |constructed - gemma| max off-diagonal = {max_constructed_vs_gemma:.4f}")

    # --- emit artifacts ---
    calendar_map = {day: {"store_row": row, "store_idx": STORE_IDX[day],
                          "calendar_pos": CAL_POS[day], "theta": float(thetas[row])}
                    for row, day in enumerate(WEEKDAY_STORE_ORDER)}
    meta = {
        "experiment": "exp3-sphere",
        "description": "frozen weekday 1-sphere manifold direction, gemma-L8-calibrated",
        "n_embd": N_EMBD, "r": 7, "seed": SEED,
        "gemma_layer": 8, "layer_axis_index": LI_L8,
        "alpha": fit["alpha"], "beta": fit["beta"],
        "rho": fit["rho"], "rho_raw": fit["rho_raw"],
        "weekday_store_order": WEEKDAY_STORE_ORDER,
        "store_idx": STORE_IDX,
        "calendar_map": calendar_map,
        "gemma_profile": {k: profile[k]["mean"] for k in (1, 2, 3)},
        "fitted_profile": fit["fitted_profile"],
        "constructed_profile": constructed_profile,
        "c_k": fit["c_k"],
    }
    npz_path = HERE / "direction_sphere.npz"
    np.savez(npz_path, D=D,
             weekday_store_order=np.array(WEEKDAY_STORE_ORDER),
             thetas=thetas.astype(np.float64),
             calendar_pos=np.array([CAL_POS[d] for d in WEEKDAY_STORE_ORDER], np.int64),
             store_idx=np.array([STORE_IDX[d] for d in WEEKDAY_STORE_ORDER], np.int64),
             alpha=np.float64(fit["alpha"]), beta=np.float64(fit["beta"]),
             seed=np.int64(SEED), meta=json.dumps(meta))
    print(f"[build] wrote {npz_path}  (D {D.shape} {D.dtype})")

    validation = {
        "gemma_layer": 8,
        "row_col_order": WEEKDAY_STORE_ORDER,
        "gemma_cosine_matrix": cos_gemma.tolist(),
        "constructed_cosine_matrix": cos_constructed.tolist(),
        "gemma_profile": {str(k): profile[k] for k in (1, 2, 3)},
        "fit": {kk: fit[kk] for kk in ("rho_raw", "rho", "alpha", "beta",
                                       "fitted_profile", "profile_max_abs_resid",
                                       "r2_against_21_pairs", "within_distance_std")},
        "constructed_profile": constructed_profile,
        "max_abs_constructed_minus_fitted_model": float(max_constructed_vs_model),
        "max_abs_constructed_minus_gemma_offdiag": max_constructed_vs_gemma,
        "row_unit_norm_max_err": unit_err,
        "calendar_map": calendar_map,
        "seed": SEED, "n_embd": N_EMBD,
    }
    val_path = HERE / "manifold_validation.json"
    json.dump(validation, open(val_path, "w"), indent=2)
    print(f"[build] wrote {val_path}")


if __name__ == "__main__":
    main()
