"""Cyclic-manifold direction builder for any k-concept family: measure the donor's
(gemma-2) cosine profile, LS-fit a k-point circle to it, emit direction_sphere.npz +
an orthogonal control + validation json. CPU-only. ``import manifold`` after
sys.path-inserting this dir; ``build_family(...)`` is the entry point.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import concepts as concept_registry  # noqa: E402

STORE_LAYERS = [6, 8, 14]   # probe_set.json "layers"; axis 0 of the [3,K,D] arrays
DEFAULT_SEED = 1337
DEFAULT_N_EMBD = 768        # nanochat depth=12, aspect_ratio=64


def find_attribution_out(start):
    """Walk up from ``start`` to the superproject's attribution/out."""
    p = Path(start).resolve()
    for _ in range(8):
        cand = p / "attribution" / "out" / "probe_set.json"
        if cand.exists():
            return cand.parent
        p = p.parent
    raise FileNotFoundError(f"no attribution/out/probe_set.json above {start} "
                            "(expected in the oracle-encodings superproject)")


def circular_distance(pos_i, pos_j, k):
    d = abs(pos_i - pos_j)
    return min(d, k - d)


def distances(k):
    """The circular distances a k-cycle can realise between distinct points."""
    return list(range(1, k // 2 + 1))


def cosine_profile(cos_matrix, cycle_pos_of_row, k):
    """Off-diagonal cosines bucketed by CYCLE circular distance. Rows/cols of
    ``cos_matrix`` are in STORE order; ``cycle_pos_of_row`` carries the map."""
    buckets = {d: [] for d in distances(k)}
    for i in range(k):
        for j in range(i + 1, k):
            d = circular_distance(cycle_pos_of_row[i], cycle_pos_of_row[j], k)
            buckets[d].append(float(cos_matrix[i, j]))
    return {d: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v),
                "values": [float(x) for x in v]}
            for d, v in buckets.items()}


def measure_donor_geometry(attr_out, family, layer=8):
    """Donor unit read-directions for one family at one gemma layer, in STORE row
    order. Replicates attribution/verify_reconstruction.py's Geometry recipe:
    V = W / nat_std, then row-normalise. Returns (U, cos, profile, cycle_pos_of_row)."""
    fam = concept_registry.get_family(family)
    attr_out = Path(attr_out)
    meta = json.load(open(attr_out / "probe_set.json"))
    store_concepts = list(meta["main_block_concepts"])
    assert list(meta["layers"]) == STORE_LAYERS, (meta["layers"], STORE_LAYERS)
    concept_registry.assert_matches_store_columns(store_concepts)
    li = STORE_LAYERS.index(layer)

    arr = np.load(attr_out / "probe_set_arrays.npz")
    W = np.asarray(arr["W"], np.float64)              # [3, K, D] std-space read weights
    nat_std = np.asarray(arr["nat_std"], np.float64)  # [3, D]
    V = W[li] / nat_std[li][None, :]                  # [K, D] raw-space read dirs
    U = V / np.linalg.norm(V, axis=1)[:, None]
    U_fam = U[list(fam.store_index)]                  # [r, D] STORE row order
    cos = U_fam @ U_fam.T
    cycle_pos = fam.cycle_position_of_store_row()
    return U_fam, cos, cosine_profile(cos, cycle_pos, fam.r), cycle_pos


def fit_circle_rho(profile, k):
    """LS fit of rho (== alpha^2, alpha^2 + beta^2 == 1) to the mean profile.
    model_cos(d) = rho + (1 - rho)*c_d, c_d = cos(2*pi*d/k). Linear in rho:
    minimize sum_d (rho*(1 - c_d) + c_d - g_d)^2."""
    ds = distances(k)
    c = {d: float(np.cos(2 * np.pi * d / k)) for d in ds}
    g = {d: profile[d]["mean"] for d in ds}
    a = np.array([1.0 - c[d] for d in ds])
    b = np.array([g[d] - c[d] for d in ds])
    denom = float(a @ a)
    assert denom > 0, f"degenerate circle fit at k={k} (all cos(2pi d/k) == 1)"
    rho_raw = float((a @ b) / denom)
    rho = float(np.clip(rho_raw, 0.0, 1.0))
    fitted = {d: rho + (1.0 - rho) * c[d] for d in ds}
    # honesty: residual of the fitted circle model vs ALL individual pairs
    obs, pred = [], []
    for d in ds:
        for v in profile[d]["values"]:
            obs.append(v)
            pred.append(fitted[d])
    obs, pred = np.array(obs), np.array(pred)
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2)) if obs.size else 0.0
    # A circle predicts cosine strictly DECREASING in cycle distance. When the donor
    # profile is not, the fit still returns a valid unit-norm (alpha, beta) but the
    # cyclic ordering is contradicted by the donor — the sphere arm is then a purely
    # imposed geometry, not a calibrated one. Callers must not silently ignore this.
    means = [g[d] for d in ds]
    return {"rho_raw": rho_raw, "rho": rho,
            "alpha": float(np.sqrt(rho)), "beta": float(np.sqrt(1.0 - rho)),
            "donor_profile_decreasing_in_distance": bool(
                all(x > y for x, y in zip(means, means[1:]))),
            "c_d": c, "fitted_profile": fitted,
            "profile_max_abs_resid": float(max(abs(fitted[d] - g[d]) for d in ds)),
            "r2_against_all_pairs": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "n_pairs": int(obs.size),
            "within_distance_std": {d: profile[d]["std"] for d in ds}}


def build_circle_direction(family, alpha, beta, seed=DEFAULT_SEED, n_embd=DEFAULT_N_EMBD):
    """D [r, n_embd] float32 in STORE row order. Row = alpha*u0 + beta*(cos th*u1 +
    sin th*u2), th from the concept's CYCLE position. u0,u1,u2 orthonormal (seeded)."""
    fam = concept_registry.get_family(family)
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((n_embd, 3)))   # orthonormal columns
    u0, u1, u2 = basis[:, 0], basis[:, 1], basis[:, 2]
    D = np.zeros((fam.r, n_embd), np.float64)
    thetas = np.zeros(fam.r)
    for row, name in enumerate(fam.store_order):
        th = 2.0 * np.pi * fam.cycle_position(name) / fam.r
        thetas[row] = th
        m = alpha * u0 + beta * (np.cos(th) * u1 + np.sin(th) * u2)
        D[row] = m / np.linalg.norm(m)     # exact unit norm (guards fp drift)
    return D.astype(np.float32), thetas, basis


def build_orthogonal_direction(family, seed=DEFAULT_SEED, n_embd=DEFAULT_N_EMBD):
    """Null-geometry control: r mutually orthogonal unit rows, from the framework's
    own generator so the file route is bit-identical to direction_init='orthonormal'."""
    from nanochat.injection.sites import orthonormal_direction
    fam = concept_registry.get_family(family)
    D = orthonormal_direction(fam.r, n_embd, seed)
    return np.ascontiguousarray(D.numpy(), dtype=np.float32)


def build_family(family, out_dir, *, attr_out=None, layer=8, seed=DEFAULT_SEED,
                 n_embd=DEFAULT_N_EMBD, orthogonal=True, log=print):
    """Emit direction_sphere.npz + manifold_validation.json (and the orthogonal
    control) for one cyclic family. Returns the validation dict."""
    fam = concept_registry.get_family(family)
    assert fam.is_cyclic, f"family {family!r} has no pinned cycle order"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    attr_out = Path(attr_out) if attr_out else find_attribution_out(out_dir)
    log(f"[manifold] {family}: attribution/out -> {attr_out}")

    _, cos_donor, profile, cycle_pos = measure_donor_geometry(attr_out, family, layer)
    for d in distances(fam.r):
        log(f"[manifold]   donor L{layer} dist {d}: mean={profile[d]['mean']:.4f} "
            f"std={profile[d]['std']:.4f} (n={profile[d]['n']})")

    fit = fit_circle_rho(profile, fam.r)
    log(f"[manifold]   fit: rho={fit['rho']:.4f} alpha={fit['alpha']:.4f} "
        f"beta={fit['beta']:.4f} (circle-model R^2 vs {fit['n_pairs']} pairs: "
        f"{fit['r2_against_all_pairs']:.2f})")
    if not fit["donor_profile_decreasing_in_distance"]:
        log(f"[manifold]   WARNING: donor cosine is NOT decreasing in cycle distance — "
            f"{family}'s donor geometry contradicts the circle ordering; the sphere arm "
            f"is an IMPOSED geometry here, not a calibrated one.")

    D, thetas, _ = build_circle_direction(family, fit["alpha"], fit["beta"], seed, n_embd)
    cos_built = D.astype(np.float64) @ D.astype(np.float64).T
    built_profile = {d: v["mean"] for d, v in
                     cosine_profile(cos_built, cycle_pos, fam.r).items()}
    max_vs_model = max(abs(built_profile[d] - fit["fitted_profile"][d])
                       for d in distances(fam.r))
    off = ~np.eye(fam.r, dtype=bool)
    max_vs_donor = float(np.max(np.abs(cos_built - cos_donor)[off]))
    unit_err = float(np.max(np.abs(np.diag(cos_built) - 1.0)))
    log(f"[manifold]   |built - fitted model| max = {max_vs_model:.2e}; "
        f"row unit-norm max err = {unit_err:.2e}")
    log(f"[manifold]   |built - donor| max off-diagonal = {max_vs_donor:.4f}")

    cycle_map = {name: {"store_row": row, "store_column": fam.store_index[row],
                        "cycle_position": fam.cycle_position(name),
                        "theta": float(thetas[row])}
                 for row, name in enumerate(fam.store_order)}
    meta = {"family": family, "r": fam.r, "n_embd": n_embd, "seed": seed,
            "gemma_layer": layer, "layer_axis_index": STORE_LAYERS.index(layer),
            "alpha": fit["alpha"], "beta": fit["beta"],
            "rho": fit["rho"], "rho_raw": fit["rho_raw"],
            "store_order": list(fam.store_order),
            "store_index": list(fam.store_index),
            "cycle_order": list(fam.cycle_order),
            "cycle_map": cycle_map,
            "donor_profile": {d: profile[d]["mean"] for d in distances(fam.r)},
            "fitted_profile": fit["fitted_profile"],
            "built_profile": built_profile,
            "c_d": fit["c_d"]}
    npz_path = out_dir / "direction_sphere.npz"
    np.savez(npz_path, D=D,
             store_order=np.array(list(fam.store_order)),
             thetas=thetas.astype(np.float64),
             cycle_position=np.array(cycle_pos, np.int64),
             store_index=np.array(list(fam.store_index), np.int64),
             alpha=np.float64(fit["alpha"]), beta=np.float64(fit["beta"]),
             seed=np.int64(seed), meta=json.dumps(meta))
    log(f"[manifold]   wrote {npz_path}  (D {D.shape} {D.dtype})")

    validation = {"family": family, "gemma_layer": layer,
                  "row_col_order": list(fam.store_order),
                  "cycle_order": list(fam.cycle_order),
                  "donor_cosine_matrix": cos_donor.tolist(),
                  "built_cosine_matrix": cos_built.tolist(),
                  "donor_profile": {str(d): profile[d] for d in distances(fam.r)},
                  "fit": {k: fit[k] for k in ("rho_raw", "rho", "alpha", "beta",
                                              "donor_profile_decreasing_in_distance",
                                              "fitted_profile", "profile_max_abs_resid",
                                              "r2_against_all_pairs", "n_pairs",
                                              "within_distance_std")},
                  "built_profile": built_profile,
                  "max_abs_built_minus_fitted_model": float(max_vs_model),
                  "max_abs_built_minus_donor_offdiag": max_vs_donor,
                  "row_unit_norm_max_err": unit_err,
                  "cycle_map": cycle_map, "seed": seed, "n_embd": n_embd}
    json.dump(validation, open(out_dir / "manifold_validation.json", "w"), indent=2)
    log(f"[manifold]   wrote {out_dir / 'manifold_validation.json'}")

    if orthogonal:
        Do = build_orthogonal_direction(family, seed, n_embd)
        np.savez(out_dir / "direction_orthogonal.npz", D=Do)
        norms = np.linalg.norm(Do, axis=1)
        cos_o = (Do / norms[:, None]) @ (Do / norms[:, None]).T
        ortho_val = {"family": family, "shape": [fam.r, n_embd], "dtype": "float32",
                     "direction_seed": seed,
                     "generator": "nanochat.injection.sites.orthonormal_direction",
                     "channel_order": list(fam.store_order),
                     "row_norms": [float(x) for x in norms],
                     "cosine_matrix": [[float(v) for v in row] for row in cos_o],
                     "max_abs_off_diagonal_cosine":
                         float(np.abs(cos_o - np.eye(fam.r)).max()),
                     "max_abs_row_norm_error": float(np.abs(norms - 1.0).max())}
        json.dump(ortho_val, open(out_dir / "orthogonal_validation.json", "w"), indent=2)
        log(f"[manifold]   wrote {out_dir / 'direction_orthogonal.npz'} + "
            f"orthogonal_validation.json (max |off-diag cos| = "
            f"{ortho_val['max_abs_off_diagonal_cosine']:.3e})")

    return validation
