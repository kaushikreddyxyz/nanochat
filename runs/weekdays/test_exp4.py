#!/usr/bin/env python3
"""Plain-assert checks for Experiment 4 (orthogonal / null weekday geometry).

Run from the repo root:  PYTHONPATH=. python3 runs/weekdays/test_exp4.py
(or:  python3 -m pytest runs/weekdays/test_exp4.py)

Verifies:
  1. the frozen direction is 7 mutually orthogonal unit rows — cosine matrix is
     identity to 1e-5, rows are unit norm;
  2. direction_orthogonal.npz round-trips bit-exact and equals what the framework
     seed route (orthonormal_direction) builds in-model — so the seed and file
     mechanisms are interchangeable;
  3. exp4_config.json parses into an InjectionCfg with trainable_direction=False
     and the pinned direction/gate/geometry fields.
"""
import json
import os

import numpy as np
import torch

from nanochat.injection.sites import (InjectionCfg, InjectionSite, classify_gate_spec,
                                       orthonormal_direction)

HERE = os.path.dirname(os.path.abspath(__file__))
R, N_EMBD, SEED = 7, 768, 1337
TOL = 1e-5

WEEKDAYS = ["friday", "monday", "saturday", "sunday", "thursday", "tuesday", "wednesday"]


def _load_npz_D():
    with np.load(os.path.join(HERE, "direction_orthogonal.npz")) as z:
        return np.ascontiguousarray(z["D"])


def test_cosine_matrix_is_identity_and_rows_unit():
    D = _load_npz_D()
    assert D.shape == (R, N_EMBD), D.shape
    assert D.dtype == np.float32, D.dtype
    norms = np.linalg.norm(D, axis=1)
    assert np.allclose(norms, 1.0, atol=TOL), f"row norms not unit: {norms}"
    unit = D / norms[:, None]
    cos = unit @ unit.T
    off = np.abs(cos - np.eye(R, dtype=cos.dtype))
    max_off = float(off.max())
    assert max_off < TOL, f"max |off-diag cosine| = {max_off} >= {TOL}"


def test_npz_bit_exact_roundtrip_and_matches_seed_route():
    D = _load_npz_D()
    # bit-exact round-trip through npz
    D2 = _load_npz_D()
    assert np.array_equal(D, D2), "npz did not round-trip bit-exact"
    # equals the framework's in-model seed route (direction_init="orthonormal")
    D_seed = orthonormal_direction(R, N_EMBD, SEED).numpy()
    assert np.array_equal(D, D_seed), \
        "npz != orthonormal_direction(7,768,1337) — seed and file routes would diverge"


def test_validation_json_agrees():
    with open(os.path.join(HERE, "orthogonal_validation.json")) as f:
        v = json.load(f)
    assert v["shape"] == [R, N_EMBD]
    assert v["direction_seed"] == SEED
    assert v["weekday_channel_order"] == WEEKDAYS
    assert v["max_abs_off_diagonal_cosine"] < TOL, v["max_abs_off_diagonal_cosine"]


def test_config_parses_into_frozen_orthonormal_cfg():
    with open(os.path.join(HERE, "exp4_config.json")) as f:
        spec = json.load(f)
    site = spec["sites"][0]
    # injection_train builds InjectionCfg(**d) with NO key stripping, so a stray
    # doc/underscore key in a SITE dict would crash the real run. Guard against it.
    assert not any(k.startswith("_") for k in site), \
        f"site dict has non-InjectionCfg keys: {[k for k in site if k.startswith('_')]}"
    cfg = InjectionCfg(**site)
    assert cfg.trainable_direction is False, "direction must be frozen for exp4"
    assert cfg.r == R
    assert cfg.after_block == 3
    assert cfg.direction_init == "orthonormal"
    assert cfg.direction_seed == SEED
    assert cfg.gate == "abs:0.0273"
    # sources keys must match site names
    assert set(spec["sources"]) == {c["name"] for c in spec["sites"]}


def test_site_builds_frozen_orthonormal_direction():
    # A built InjectionSite from the cfg reproduces the orthonormal null geometry
    # in-model and its direction is non-trainable. injection_train resolves the
    # gate spec ("abs:0.0273" -> 0.0273) via classify_gate_spec before building
    # the site; mirror that here.
    mode, val = classify_gate_spec("abs:0.0273")
    assert mode == "abs" and abs(val - 0.0273) < 1e-12, (mode, val)
    cfg = InjectionCfg(name="weekdays", r=R, after_block=3, gate=float(val),
                       trainable_direction=False, direction_init="orthonormal",
                       direction_seed=SEED)
    site = InjectionSite(cfg, N_EMBD)
    assert site.direction.requires_grad is False, "frozen direction must not require grad"
    D = site.direction.detach().numpy()
    D_npz = _load_npz_D()
    assert np.array_equal(D, D_npz), "in-model site direction != emitted npz"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nall {len(fns)} exp4 checks passed")
