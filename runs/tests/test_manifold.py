"""manifold tests: circle construction for arbitrary k, the closed-form cosine
identity, the rho fit, and orthogonal-control invariants — pure math, no artifacts.
Reproduction of the COMMITTED direction files lives in test_arm_configs.py, table-driven
over every family."""
import numpy as np
import pytest

import concepts
import manifold


@pytest.mark.parametrize("k,expected", [(3, [1]), (4, [1, 2]), (7, [1, 2, 3]),
                                        (8, [1, 2, 3, 4]), (12, [1, 2, 3, 4, 5, 6])])
def test_distances_cover_the_cycle(k, expected):
    assert manifold.distances(k) == expected


def test_circular_distance_wraps():
    assert manifold.circular_distance(0, 6, 7) == 1
    assert manifold.circular_distance(0, 3, 7) == 3
    assert manifold.circular_distance(0, 2, 4) == 2      # antipode on an even cycle
    assert manifold.circular_distance(3, 0, 4) == 1


@pytest.mark.parametrize("family", ["seasons", "weekdays", "months", "moon_phases",
                                    "directions"])
def test_circle_direction_obeys_the_closed_form_cosine(family):
    fam = concepts.get_family(family)
    alpha, beta = 0.6, np.sqrt(1 - 0.36)
    D, thetas, basis = manifold.build_circle_direction(family, alpha, beta, n_embd=64)
    assert D.shape == (fam.r, 64) and D.dtype == np.float32
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-12)
    assert np.max(np.abs(np.linalg.norm(D, axis=1) - 1.0)) < 1e-6

    # theta comes from the CYCLE position, not the store row
    for row, name in enumerate(fam.store_order):
        assert np.isclose(thetas[row], 2 * np.pi * fam.cycle_position(name) / fam.r)

    C = D.astype(np.float64) @ D.astype(np.float64).T
    rho = alpha ** 2
    cyc = fam.cycle_position_of_store_row()
    for i in range(fam.r):
        for j in range(fam.r):
            if i == j:
                continue
            d = manifold.circular_distance(cyc[i], cyc[j], fam.r)
            assert np.isclose(C[i, j], rho + (1 - rho) * np.cos(2 * np.pi * d / fam.r),
                              atol=1e-6)


@pytest.mark.parametrize("family", ["seasons", "weekdays"])
def test_nearest_cycle_neighbours_carry_the_max_cosine(family):
    fam = concepts.get_family(family)
    D, _, _ = manifold.build_circle_direction(family, 0.7, np.sqrt(1 - 0.49), n_embd=64)
    C = D.astype(np.float64) @ D.astype(np.float64).T
    cyc = fam.cycle_position_of_store_row()
    for i in range(fam.r):
        off = [(manifold.circular_distance(cyc[i], cyc[j], fam.r), C[i, j])
               for j in range(fam.r) if j != i]
        best = max(c for _, c in off)
        assert {d for d, c in off if abs(c - best) < 1e-9} == {1}


def test_seasons_is_a_four_point_circle_with_antipodal_opposites():
    fam = concepts.get_family("seasons")
    assert fam.r == 4
    alpha = 0.5
    D, _, _ = manifold.build_circle_direction("seasons", alpha, np.sqrt(1 - alpha ** 2),
                                              n_embd=32)
    C = D.astype(np.float64) @ D.astype(np.float64).T
    rho = alpha ** 2
    # spring/autumn and summer/winter are the two antipodal pairs: cos = 2*rho - 1
    for a, b in (("spring", "autumn"), ("summer", "winter")):
        assert np.isclose(C[fam.store_row(a), fam.store_row(b)], 2 * rho - 1, atol=1e-6)
    # adjacent seasons sit at 90 degrees on the circle: cos = rho
    for a, b in (("spring", "summer"), ("summer", "autumn"), ("autumn", "winter"),
                 ("winter", "spring")):
        assert np.isclose(C[fam.store_row(a), fam.store_row(b)], rho, atol=1e-6)


@pytest.mark.parametrize("k,profile_cos", [(4, {1: 0.5, 2: 0.1}),
                                           (7, {1: 0.6, 2: 0.5, 3: 0.45})])
def test_fit_circle_rho_recovers_a_synthetic_circle(k, profile_cos):
    rho_true = 0.4
    prof = {d: {"mean": rho_true + (1 - rho_true) * np.cos(2 * np.pi * d / k),
                "std": 0.0, "n": 1, "values": []}
            for d in manifold.distances(k)}
    fit = manifold.fit_circle_rho(prof, k)
    assert np.isclose(fit["rho"], rho_true, atol=1e-12)
    assert np.isclose(fit["alpha"] ** 2 + fit["beta"] ** 2, 1.0, atol=1e-12)
    assert fit["profile_max_abs_resid"] < 1e-12
    # a non-circular profile still yields a clipped, unit-norm (alpha, beta)
    fit2 = manifold.fit_circle_rho(
        {d: {"mean": v, "std": 0.0, "n": 1, "values": []} for d, v in profile_cos.items()}, k)
    assert 0.0 <= fit2["rho"] <= 1.0
    assert np.isclose(fit2["alpha"] ** 2 + fit2["beta"] ** 2, 1.0, atol=1e-12)


@pytest.mark.parametrize("family", ["seasons", "weekdays"])
def test_orthogonal_control_rows_are_orthonormal(family):
    D = manifold.build_orthogonal_direction(family, n_embd=128)
    r = concepts.get_family(family).r
    assert D.shape == (r, 128) and D.dtype == np.float32
    G = D.astype(np.float64) @ D.astype(np.float64).T
    assert np.max(np.abs(G - np.eye(r))) < 1e-6


def test_cosine_profile_buckets_by_cycle_distance_not_store_row():
    fam = concepts.get_family("weekdays")
    D, _, _ = manifold.build_circle_direction("weekdays", 0.7, np.sqrt(0.51), n_embd=64)
    C = D.astype(np.float64) @ D.astype(np.float64).T
    by_cycle = manifold.cosine_profile(C, fam.cycle_position_of_store_row(), 7)
    by_store = manifold.cosine_profile(C, list(range(7)), 7)
    # within a true cycle bucket every pair is identical; store-row bucketing smears
    assert all(v["std"] < 1e-8 for v in by_cycle.values())
    assert max(v["std"] for v in by_store.values()) > 1e-3
