"""Seasons scaffold tests: the configs parse and pin the right columns/r/after_block/
loudness, and the built direction artifacts match the registry."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from nanochat.injection.sites import classify_loudness_spec

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "runs" / "seasons"

import concepts  # noqa: E402
import manifold  # noqa: E402

SEASONS = concepts.get_family("seasons")
CONFIGS = {"trainable": HERE / "exp_trainable_L0.json",
           "sphere": HERE / "exp_sphere_L0.json"}
DRIVERS = {"trainable": HERE / "run_trainable_L0.sh",
           "sphere": HERE / "run_sphere_L0.sh"}


@pytest.fixture(params=sorted(CONFIGS))
def arm(request):
    return request.param


def _cfg(arm):
    return json.load(open(CONFIGS[arm]))


def test_site_pins_r4_at_block_zero(arm):
    sites = _cfg(arm)["sites"]
    assert len(sites) == 1
    site = sites[0]
    assert site["name"] == "seasons"
    assert site["r"] == 4 == SEASONS.r
    assert site["after_block"] == 0          # layer-0 injection, not the weekday 3


def test_source_columns_are_the_registry_store_order(arm):
    src = _cfg(arm)["sources"]["seasons"]
    assert src["concepts"] == list(SEASONS.store_order)
    assert src["concepts"] == ["autumn", "spring", "summer", "winter"]
    # the cycle order is a DIFFERENT permutation and must never appear here
    assert src["concepts"] != list(SEASONS.cycle_order)
    assert src["layer"] == 8
    assert src["kind"] == "probe-scores-runtime"
    assert src["align_policy"] == "max" and src["noise_sigma"] == 0.0


def test_source_wires_the_shared_library_class(arm):
    src = _cfg(arm)["sources"]["seasons"]
    path, _, cls = src["class"].rpartition(":")
    assert (REPO / path).is_file()
    assert cls == "ConceptProbeScoreSource"
    # the site's relu owns thresholding under dose; present_z would only clip on top
    assert src["kwargs"] == {"present_z": 0.0, "family": "seasons"}


def test_prefetch_block_matches_the_scored_store(arm):
    pf = _cfg(arm)["sources"]["seasons"]["prefetch"]
    assert _cfg(arm)["sources"]["seasons"]["score_shards_dir_or_repo"] == \
        "kaushikreddyxyz/climbmix-scored"
    assert _cfg(arm)["sources"]["seasons"]["shards"] == "0-184"
    assert len(pf["repos"]) == 8 and pf["per_repo"] == 25
    assert pf["repos"][0] == "kaushikreddyxyz/climbmix-scored"
    assert pf["repos"][-1] == "kaushikreddyxyz/climbmix-scored-overflow-7"


def test_loudness_is_the_donor_dial(arm):
    # Loudness is calibrated at startup from the dial; nothing is hand-set here.
    assert classify_loudness_spec(_cfg(arm)["sites"][0]["loudness"]) == ("dial", 1.0)


def test_direction_arms_differ_only_in_the_direction():
    tr, sp = _cfg("trainable")["sites"][0], _cfg("sphere")["sites"][0]
    assert tr["trainable_direction"] is True and tr["direction_init"] == "orthonormal"
    assert sp["trainable_direction"] is False
    assert sp["direction_init"] == "file:runs/seasons/direction_sphere.npz"
    assert _cfg("trainable")["sources"] == _cfg("sphere")["sources"]
    for k in ("name", "r", "after_block", "loudness"):
        assert tr[k] == sp[k]


def test_sphere_direction_file_exists_and_matches_the_site():
    path = REPO / _cfg("sphere")["sites"][0]["direction_init"][len("file:"):]
    assert path.is_file()
    D = np.load(path)["D"]
    assert D.shape == (4, 768) and D.dtype == np.float32
    assert np.max(np.abs(np.linalg.norm(D.astype(np.float64), axis=1) - 1.0)) < 1e-5


def test_built_direction_reproduces_from_the_library():
    npz = np.load(HERE / "direction_sphere.npz")
    meta = json.loads(str(npz["meta"]))
    assert meta["family"] == "seasons" and meta["r"] == 4
    assert meta["store_order"] == list(SEASONS.store_order)
    assert meta["cycle_order"] == list(SEASONS.cycle_order)
    assert meta["store_index"] == [43, 44, 45, 46]
    D, thetas, _ = manifold.build_circle_direction("seasons", meta["alpha"], meta["beta"],
                                                   seed=meta["seed"], n_embd=meta["n_embd"])
    assert np.array_equal(D, npz["D"])
    # thetas follow the CYCLE order: spring 0, summer pi/2, autumn pi, winter 3pi/2
    for name, expected in (("spring", 0.0), ("summer", np.pi / 2),
                           ("autumn", np.pi), ("winter", 3 * np.pi / 2)):
        assert np.isclose(thetas[SEASONS.store_row(name)], expected)


def test_orthogonal_control_matches_the_framework_generator():
    ref = np.load(HERE / "direction_orthogonal.npz")["D"]
    assert np.array_equal(ref, manifold.build_orthogonal_direction("seasons"))


def test_validation_json_records_the_donor_ordering_verdict():
    val = json.load(open(HERE / "manifold_validation.json"))
    assert val["family"] == "seasons"
    assert val["row_col_order"] == list(SEASONS.store_order)
    # gemma's season directions are NOT ordered like a circle (opposite seasons are
    # more aligned than adjacent ones) — the sphere arm is an imposed geometry here.
    assert val["fit"]["donor_profile_decreasing_in_distance"] is False
    assert val["donor_profile"]["2"]["mean"] > val["donor_profile"]["1"]["mean"]


def test_hf_target_is_the_unified_repo(arm):
    text = DRIVERS[arm].read_text()
    assert "kaushikreddyxyz/nanochat-d12-injections" in text
    assert f"HF_SUBDIR=${{HF_SUBDIR:-seasons_{arm}_L0}}" in text
    assert "runs/lib/hf_push.py" in text
