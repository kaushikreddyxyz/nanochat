"""Scaffold tests for EVERY arm of EVERY family: the exp configs parse into the
library's InjectionCfg and pin the registry's columns/r/loudness, the committed
direction artifacts reproduce from runs/lib/manifold.py, and the shared launcher accepts
each arm's documented invocation.
Table-driven over ARMS — adding a family means adding rows, never a new test file.
"""
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nanochat.injection.sites import (InjectionCfg, classify_loudness_spec,
                                      validate_direction_file_order)

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "runs" / "lib"

import concepts  # noqa: E402
import manifold  # noqa: E402

LAUNCHER = HERE / "launch_arm.sh"

# One row per trainable arm. `config=None` is the injection-free baseline (base_train).
ARMS = {
    "weekdays_baseline":  dict(family="weekdays", config=None, after_block=None),
    "weekdays_trainable": dict(family="weekdays", config="runs/weekdays/exp2_config.json",
                               after_block=3, trainable=True, direction="orthonormal"),
    "weekdays_sphere":    dict(family="weekdays", config="runs/weekdays/exp3_config.json",
                               after_block=3, trainable=False,
                               direction="file:runs/weekdays/direction_sphere.npz"),
    "weekdays_orthogonal": dict(family="weekdays", config="runs/weekdays/exp4_config.json",
                                after_block=3, trainable=False, direction="orthonormal"),
    "seasons_trainable":  dict(family="seasons", config="runs/seasons/exp_trainable_L0.json",
                               after_block=0, trainable=True, direction="orthonormal"),
    "seasons_sphere":     dict(family="seasons", config="runs/seasons/exp_sphere_L0.json",
                               after_block=0, trainable=False,
                               direction="file:runs/seasons/direction_sphere.npz"),
}
INJECTED = [k for k, v in ARMS.items() if v["config"]]
FAMILIES = sorted({v["family"] for v in ARMS.values()})

# The weekday artifacts predate runs/lib and carry the old key spelling; regenerating
# them would rewrite a COMPLETED campaign's committed record, so the test speaks both.
VALIDATION_KEYS = {"donor_cos": ("donor_cosine_matrix", "gemma_cosine_matrix"),
                   "donor_prof": ("donor_profile", "gemma_profile")}


def _cfg(arm):
    return json.load(open(REPO / ARMS[arm]["config"]))


def _pick(d, key):
    for k in VALIDATION_KEYS[key]:
        if k in d:
            return d[k]
    raise KeyError(f"{key}: none of {VALIDATION_KEYS[key]} in {sorted(d)}")


# --------------------------------------------------------------------------- #
# configs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", INJECTED)
def test_site_dict_is_accepted_by_the_library(arm):
    # injection_train builds sites via InjectionCfg(**site), so an unknown or renamed
    # field is a launch-time TypeError; catch it here instead.
    cfg = InjectionCfg(**_cfg(arm)["sites"][0])
    assert cfg.name and cfg.r > 0
    classify_loudness_spec(cfg.loudness)


@pytest.mark.parametrize("arm", INJECTED)
def test_site_matches_the_registry(arm):
    spec, fam = ARMS[arm], concepts.get_family(ARMS[arm]["family"])
    sites = _cfg(arm)["sites"]
    assert len(sites) == 1
    site = sites[0]
    assert site["name"] == spec["family"]
    assert site["r"] == fam.r
    assert site["after_block"] == spec["after_block"]
    assert site["trainable_direction"] is spec["trainable"]
    assert site["direction_init"] == spec["direction"]


@pytest.mark.parametrize("arm", INJECTED)
def test_source_columns_are_the_registry_store_order(arm):
    fam = concepts.get_family(ARMS[arm]["family"])
    src = _cfg(arm)["sources"][ARMS[arm]["family"]]
    assert src["concepts"] == list(fam.store_order)
    # the cycle order is a DIFFERENT permutation and must never appear here
    assert src["concepts"] != list(fam.cycle_order)
    assert src["layer"] == 8
    assert src["kind"] == "probe-scores-runtime"
    assert src["align_policy"] == "max" and src["noise_sigma"] == 0.0


@pytest.mark.parametrize("arm", INJECTED)
def test_source_wires_the_shared_library_class(arm):
    src = _cfg(arm)["sources"][ARMS[arm]["family"]]
    path, _, cls = src["class"].rpartition(":")
    assert (REPO / path).is_file()
    assert "/" in path, "'class' must use the collision-proof file-path form"
    assert cls == "ConceptProbeScoreSource"
    # The SITE owns thresholding under dose; a source-level present_z would only clip
    # dose events on top of the site's relu, so it must be off.
    assert src["kwargs"] == {"present_z": 0.0, "family": ARMS[arm]["family"]}
    site = _cfg(arm)["sites"][0]
    assert float(site.get("threshold", InjectionCfg.threshold)) == 2.0


@pytest.mark.parametrize("arm", INJECTED)
def test_class_suffix_satisfies_the_injection_train_assertion(arm):
    # injection_train asserts type(src).__name__ == spec["class"].rsplit(":", 1)[1].
    sys.path.insert(0, str(HERE))
    from probe_source import ConceptProbeScoreSource
    src = _cfg(arm)["sources"][ARMS[arm]["family"]]
    assert ConceptProbeScoreSource.__name__ == src["class"].rsplit(":", 1)[1]


@pytest.mark.parametrize("arm", INJECTED)
def test_prefetch_block_matches_the_scored_store(arm):
    src = _cfg(arm)["sources"][ARMS[arm]["family"]]
    assert src["score_shards_dir_or_repo"] == "kaushikreddyxyz/climbmix-scored"
    assert src["shards"] == "0-184"
    pf = src["prefetch"]
    assert len(pf["repos"]) == 8 and pf["per_repo"] == 25
    assert pf["repos"][0] == "kaushikreddyxyz/climbmix-scored"
    assert pf["repos"][-1] == "kaushikreddyxyz/climbmix-scored-overflow-7"


@pytest.mark.parametrize("arm", INJECTED)
def test_comment_is_one_line(arm):
    c = _cfg(arm)["_comment"]
    assert "\n" not in c and len(c) < 200, "config _comment names the arm; facts live in code"


@pytest.mark.parametrize("arm", INJECTED)
def test_loudness_is_a_dial_or_a_well_formed_absolute_target(arm):
    mode, value = classify_loudness_spec(_cfg(arm)["sites"][0]["loudness"])
    assert mode in ("dial", "abs") and value > 0


@pytest.mark.parametrize("arm", INJECTED)
def test_every_injected_arm_uses_the_donor_dial(arm):
    # dial 1.0 resolving to the weekday campaign's 0.0273 is pinned by
    # tests/test_injection_dose.py::test_dose_L_ref_reproduces_the_hand_set_weekday_gate.
    assert classify_loudness_spec(_cfg(arm)["sites"][0]["loudness"])[0] == "dial"


@pytest.mark.parametrize("family", FAMILIES)
def test_arms_of_a_family_differ_only_in_the_direction(family):
    arms = [a for a in INJECTED if ARMS[a]["family"] == family]
    ref = _cfg(arms[0])
    for a in arms[1:]:
        cfg = _cfg(a)
        assert cfg["sources"] == ref["sources"], f"{a}: source wiring drifted"
        for k in ("name", "r", "after_block"):
            assert cfg["sites"][0][k] == ref["sites"][0][k], f"{a}: site {k} drifted"


@pytest.mark.parametrize("family", FAMILIES)
def test_loudness_agrees_across_a_family(family):
    """Arms of a family differ only in geometry, so they must share one loudness."""
    loud = {a: _cfg(a)["sites"][0]["loudness"] for a in INJECTED
            if ARMS[a]["family"] == family}
    assert len(set(loud.values())) == 1, f"{family}: loudness diverged: {loud}"


# --------------------------------------------------------------------------- #
# committed direction artifacts reproduce from the library
# --------------------------------------------------------------------------- #
def _attr_out():
    try:
        return manifold.find_attribution_out(__file__)
    except FileNotFoundError:
        pytest.skip("attribution/out not available")


@pytest.mark.parametrize("arm", [a for a in INJECTED if ARMS[a]["direction"].startswith("file:")])
def test_file_direction_exists_and_matches_the_site(arm):
    fam = concepts.get_family(ARMS[arm]["family"])
    path = REPO / _cfg(arm)["sites"][0]["direction_init"][len("file:"):]
    assert path.is_file(), "existing checkpoint metas name this exact path"
    D = np.load(path)["D"]
    assert D.shape == (fam.r, 768) and D.dtype == np.float32
    assert np.max(np.abs(np.linalg.norm(D.astype(np.float64), axis=1) - 1.0)) < 1e-5


@pytest.mark.parametrize("arm", [a for a in INJECTED if ARMS[a]["direction"].startswith("file:")])
def test_file_direction_row_order_matches_the_arms_concept_list(arm):
    # Row i of D IS concept i. The npz declares its own row order and nothing read it,
    # so reordering a config's `concepts` would silently permute the geometry — the
    # permutation class of bug this project has been bitten by before.
    cfg = _cfg(arm)
    site = cfg["sites"][0]
    concept_list = cfg["sources"][site["name"]]["concepts"]
    prev = os.getcwd()
    os.chdir(REPO)                       # file: paths resolve from the launch CWD
    try:
        key = validate_direction_file_order(site["direction_init"], concept_list, site["name"])
        assert key, f"{arm}: direction npz declares no row order to check against"
        with pytest.raises(ValueError, match="row order"):
            validate_direction_file_order(site["direction_init"],
                                          list(reversed(concept_list)), site["name"])
    finally:
        os.chdir(prev)


@pytest.mark.parametrize("family", FAMILIES)
def test_committed_sphere_direction_is_bit_identical_to_the_library(family):
    attr_out = _attr_out()
    fam = concepts.get_family(family)
    ref = np.load(REPO / "runs" / family / "direction_sphere.npz")
    _, _, profile, _ = manifold.measure_donor_geometry(attr_out, family, layer=8)
    fit = manifold.fit_circle_rho(profile, fam.r)
    meta = json.loads(str(ref["meta"]))
    assert np.isclose(fit["alpha"], meta["alpha"], atol=1e-12)
    assert np.isclose(fit["beta"], meta["beta"], atol=1e-12)
    D, thetas, _ = manifold.build_circle_direction(family, fit["alpha"], fit["beta"],
                                                   seed=meta["seed"], n_embd=meta["n_embd"])
    assert np.array_equal(D, ref["D"]), f"{family} sphere direction is not bit-identical"
    assert np.allclose(thetas, ref["thetas"], atol=1e-15)


@pytest.mark.parametrize("family", FAMILIES)
def test_committed_orthogonal_control_is_bit_identical_to_the_library(family):
    ref = np.load(REPO / "runs" / family / "direction_orthogonal.npz")["D"]
    assert np.array_equal(manifold.build_orthogonal_direction(family), ref)


@pytest.mark.parametrize("family", FAMILIES)
def test_donor_profile_matches_the_committed_validation_json(family):
    attr_out = _attr_out()
    fam = concepts.get_family(family)
    val = json.load(open(REPO / "runs" / family / "manifold_validation.json"))
    _, cos, profile, _ = manifold.measure_donor_geometry(attr_out, family, layer=8)
    assert np.allclose(cos, np.array(_pick(val, "donor_cos")), atol=1e-12)
    for d in manifold.distances(fam.r):
        assert np.isclose(profile[d]["mean"], _pick(val, "donor_prof")[str(d)]["mean"],
                          atol=1e-12)
    assert val["row_col_order"] == list(fam.store_order)


def test_seasons_donor_geometry_contradicts_the_circle_ordering():
    # gemma's season directions are NOT ordered like a circle (opposite seasons are more
    # aligned than adjacent ones) — the seasons sphere arm is an IMPOSED geometry.
    val = json.load(open(REPO / "runs" / "seasons" / "manifold_validation.json"))
    assert val["fit"]["donor_profile_decreasing_in_distance"] is False
    assert _pick(val, "donor_prof")["2"]["mean"] > _pick(val, "donor_prof")["1"]["mean"]


# --------------------------------------------------------------------------- #
# the shared launcher
# --------------------------------------------------------------------------- #
def _header_invocations():
    """The usage-header lines are the ONLY published per-arm invocations, so parse them
    from the launcher itself — a stale header then fails these tests, not a run."""
    out = []
    for line in LAUNCHER.read_text().splitlines():
        m = re.match(r"^#\s+((?:[A-Z_]+=\S+\s+)+)bash runs/lib/launch_arm\.sh\s*$", line)
        if m:
            out.append(dict(kv.split("=", 1) for kv in shlex.split(m.group(1))))
    return out


def test_header_documents_every_arm():
    envs = _header_invocations()
    assert len(envs) == len(ARMS), f"header lists {len(envs)} arms, table has {len(ARMS)}"
    assert {e["MODEL_TAG"] for e in envs} == {e["MODEL_TAG"] for e in envs}, "duplicate MODEL_TAG"
    configs = {e.get("CONFIG") for e in envs}
    assert configs == {v["config"] for v in ARMS.values()}, "header CONFIGs != the arm table"


@pytest.mark.parametrize("env", _header_invocations(),
                         ids=lambda e: e["MODEL_TAG"])
def test_header_invocation_is_well_formed(env):
    assert env["MODEL_TAG"] and env.get("RUN_NAME")
    if env.get("CONFIG"):
        assert (REPO / env["CONFIG"]).is_file()


def test_launcher_and_bootstrap_parse():
    for sh in (LAUNCHER, REPO / "runs" / "weekdays" / "pod_bootstrap.sh",
               REPO / "runs" / "weekdays" / "probes" / "run_probes_pod.sh"):
        assert subprocess.run(["bash", "-n", str(sh)], capture_output=True).returncode == 0, sh


def test_launcher_requires_a_model_tag():
    proc = subprocess.run(["bash", str(LAUNCHER)], capture_output=True, text=True,
                          cwd=str(REPO), timeout=120,
                          env={k: v for k, v in os.environ.items() if k != "MODEL_TAG"})
    assert proc.returncode != 0 and "MODEL_TAG" in proc.stderr


def test_launcher_rejects_a_missing_config():
    proc = subprocess.run(["bash", str(LAUNCHER)], capture_output=True, text=True,
                          cwd=str(REPO), timeout=120,
                          env={**os.environ, "MODEL_TAG": "x", "CONFIG": "runs/nope.json"})
    assert proc.returncode == 1 and "CONFIG not found" in proc.stderr


def test_hf_targets_are_the_expected_repos():
    envs = {e["MODEL_TAG"]: e for e in _header_invocations()}
    for tag, e in envs.items():
        repo = e.get("HF_REPO", "kaushikreddyxyz/nanochat-d12-injections")
        expected = ("kaushikreddyxyz/weekday-geometry-d12" if tag.startswith("weekday")
                    else "kaushikreddyxyz/nanochat-d12-injections")
        assert repo == expected, f"{tag} pushes to {repo}"


# --------------------------------------------------------------------------- #
# budget / cross-arm consistency
# --------------------------------------------------------------------------- #
def test_budget_math_is_the_ratio12_derivation():
    text = LAUNCHER.read_text()
    iters = int(re.search(r"NUM_ITERATIONS=\$\{NUM_ITERATIONS:-(\d+)\}", text).group(1))
    seq = int(re.search(r"MAX_SEQ_LEN=\$\{MAX_SEQ_LEN:-(\d+)\}", text).group(1))
    dbs = int(re.search(r"DEVICE_BATCH_SIZE=\$\{DEVICE_BATCH_SIZE:-(\d+)\}", text).group(1))
    nproc = int(re.search(r"NPROC=\$\{NPROC:-(\d+)\}", text).group(1))
    assert (iters, seq, dbs, nproc) == (2520, 2048, 32, 8)
    assert iters * seq * dbs * nproc == 1_321_205_760


def test_every_arm_shares_one_pinned_training_config():
    # One launcher => the pinned knobs cannot drift between arms by construction; this
    # asserts the header never overrides any of them per-arm.
    pinned = {"DEPTH", "RATIO", "NUM_ITERATIONS", "MAX_SEQ_LEN", "DEVICE_BATCH_SIZE",
              "NPROC", "SEED", "PRECISION", "TRAIN_SHARDS"}
    for env in _header_invocations():
        assert not (pinned & set(env)), f"{env['MODEL_TAG']} overrides pinned knobs"
