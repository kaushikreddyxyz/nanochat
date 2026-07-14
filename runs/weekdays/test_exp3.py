#!/usr/bin/env python3
"""Plain-assert tests for Experiment 3 (weekday sphere manifold). Run:
    python runs/weekdays/test_exp3.py
Covers: rows unit-norm; store<->calendar mapping; adjacent calendar days carry
the max off-diagonal cosine; constructed profile matches the fitted circle model
to 1e-6; D loads back from the npz bit-exact; and REPORTS (does not hard-fail)
the fitted-vs-gemma residual."""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WEEKDAY_STORE_ORDER = ["friday", "monday", "saturday", "sunday",
                       "thursday", "tuesday", "wednesday"]
STORE_IDX = {"friday": 47, "monday": 48, "saturday": 49, "sunday": 50,
             "thursday": 51, "tuesday": 52, "wednesday": 53}
CALENDAR = ["monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday"]
CAL_POS = {d: i for i, d in enumerate(CALENDAR)}


def circular_dist(a, b, n=7):
    d = abs(a - b)
    return min(d, n - d)


def main():
    npz = np.load(HERE / "direction_sphere.npz")
    D = np.asarray(npz["D"], np.float64)
    meta = json.loads(str(npz["meta"]))
    val = json.load(open(HERE / "manifold_validation.json"))

    # --- shapes / dtype ---
    assert npz["D"].dtype == np.float32, npz["D"].dtype
    assert D.shape == (7, meta["n_embd"]), D.shape
    assert meta["r"] == 7 and meta["n_embd"] == 768, meta

    # --- rows unit norm ---
    norms = np.linalg.norm(D, axis=1)
    assert np.max(np.abs(norms - 1.0)) < 1e-5, norms
    print(f"[ok] all 7 rows unit-norm (max |‖row‖-1| = {np.max(np.abs(norms - 1.0)):.2e})")

    # --- store<->calendar mapping (the CHANNEL-ORDER TRAP) ---
    order = list(map(str, npz["weekday_store_order"]))
    assert order == WEEKDAY_STORE_ORDER, order
    # store indices ascend friday(47)..wednesday(53)
    assert [int(x) for x in npz["store_idx"]] == [STORE_IDX[d] for d in WEEKDAY_STORE_ORDER]
    # monday is store row 1 with theta == 0 (calendar pos 0)
    assert order[1] == "monday", order
    assert CAL_POS["monday"] == 0
    assert abs(float(npz["thetas"][1]) - 0.0) < 1e-12, npz["thetas"][1]
    # every row's theta matches its calendar position
    for row, day in enumerate(order):
        exp_theta = 2 * math.pi * CAL_POS[day] / 7
        assert abs(float(npz["thetas"][row]) - exp_theta) < 1e-12, (day, npz["thetas"][row])
    print("[ok] store<->calendar map correct (monday@row1, theta=0; all thetas match calendar pos)")

    # --- adjacent calendar days carry the max off-diagonal cosine ---
    C = D @ D.T
    cal_of_row = [CAL_POS[d] for d in order]
    for i in range(7):
        offdiag = [(circular_dist(cal_of_row[i], cal_of_row[j]), C[i, j], j)
                   for j in range(7) if j != i]
        max_cos = max(c for _, c, _ in offdiag)
        # the row's argmax off-diagonal neighbours must be its calendar distance-1 days
        argmax_dists = {d for d, c, _ in offdiag if abs(c - max_cos) < 1e-9}
        assert argmax_dists == {1}, (order[i], offdiag)
    print("[ok] every row's max off-diagonal cosine is at calendar distance 1")

    # --- constructed profile matches the fitted circle model to 1e-6 ---
    fitted = {int(k): v for k, v in meta["fitted_profile"].items()}
    buckets = {1: [], 2: [], 3: []}
    for i in range(7):
        for j in range(i + 1, 7):
            buckets[circular_dist(cal_of_row[i], cal_of_row[j])].append(C[i, j])
    for k in (1, 2, 3):
        got = float(np.mean(buckets[k]))
        assert abs(got - fitted[k]) < 1e-6, (k, got, fitted[k])
        # within a distance bucket ALL pairs are equal (that's the circle property)
        assert np.max(np.abs(np.array(buckets[k]) - got)) < 1e-6, (k, buckets[k])
    print("[ok] constructed cosine profile matches fitted circle model to <1e-6")

    # --- npz -> D bit-exact roundtrip ---
    reloaded = np.load(HERE / "direction_sphere.npz")["D"]
    assert reloaded.dtype == np.float32
    assert np.array_equal(reloaded, npz["D"]), "D not bit-exact on reload"
    # matches the matrix recorded in validation json
    assert np.max(np.abs(np.array(val["constructed_cosine_matrix"]) - C)) < 1e-9
    print("[ok] D reloads bit-exact; matches manifold_validation.json")

    # --- REPORT (do not hard-fail): fitted-vs-gemma residual ---
    gemma = {int(k): v for k, v in meta["gemma_profile"].items()}
    print("\n[report] gemma vs fitted circle-model profile (cosine by calendar distance):")
    for k in (1, 2, 3):
        print(f"         dist {k}: gemma={gemma[k]:.4f}  fitted={fitted[k]:.4f}  "
              f"|resid|={abs(gemma[k]-fitted[k]):.4f}")
    print(f"[report] 3-pt max |resid| = {val['fit']['profile_max_abs_resid']:.4f}; "
          f"circle-model R^2 vs 21 pairs = {val['fit']['r2_against_21_pairs']:.4f}")
    if val["fit"]["profile_max_abs_resid"] > 0.15:
        print("[report] NOTE: residual is LARGE -- gemma's weekday geometry is a near-uniform "
              "positive clump, not a clean circle. The circle model is imposed by design (exp3), "
              "calibrated (alpha/beta) but NOT a good description of gemma. See NOTES_3_sphere.md.")

    print("\nALL STRUCTURAL ASSERTIONS PASSED.")


if __name__ == "__main__":
    main()
