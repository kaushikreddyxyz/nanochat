"""CPU tests for the dose site math and its loudness calibration: exact linear dose
response, bitwise-zero no-op on sub-threshold/zero/negative rows, D's row scale as a
pure gauge, the sphere direction's gram surviving row normalization, PER-EVENT (not
per-total) equalization with the min_events imputation, name-indexed concept subsets,
the subset L_ref, and the post-alignment realized-loudness check."""
import json
import os
import sys
from dataclasses import asdict

import numpy as np
import pytest
import torch

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(TESTS, ".."))
sys.path.insert(0, REPO)

from nanochat.injection.sites import (  # noqa: E402
    InjectionCfg,
    InjectionSite,
    calibrate_dose_gate,
    loudness_vector_hash,
    log_realized_loudness,
    realized_loudness_report,
    reassert_optimizability,
    build_sites,
)

B, T, N_EMBD, R = 2, 8, 64, 6
SPHERE_NPZ = os.path.join(REPO, "runs", "weekdays", "direction_sphere.npz")


def _dose_site(channel_scale=None, threshold=2.0, **kw):
    kw.setdefault("name", "acts")
    kw.setdefault("r", R)
    return InjectionSite(InjectionCfg(
        threshold=threshold,
        channel_scale=[1.0] * kw["r"] if channel_scale is None else channel_scale, **kw), N_EMBD)


# --------------------------------------------------------------------------- #
# 1. Dose response: strictly monotone, and exactly linear in (a - z0).
# --------------------------------------------------------------------------- #
def test_dose_response_is_exactly_linear_in_the_excess_over_threshold():
    x = torch.zeros(1, 2, N_EMBD)
    s = _dose_site(threshold=2.0)
    a = torch.zeros(1, 2, R)
    a[0, 0, 0], a[0, 1, 0] = 4.0, 2.6           # one active channel, two doses

    d = s(x, a) - x
    lo, hi = d[0, 1].norm().item(), d[0, 0].norm().item()
    assert hi > lo > 0.0
    assert hi / lo == pytest.approx((4.0 - 2.0) / (2.6 - 2.0), rel=1e-5)   # 3.333...


def test_dose_is_monotone_across_a_sweep():
    x = torch.zeros(1, 5, N_EMBD)
    s = _dose_site(threshold=2.0)
    a = torch.zeros(1, 5, R)
    a[0, :, 0] = torch.tensor([2.1, 2.6, 3.0, 4.0, 9.0])
    n = (s(x, a) - x)[0].norm(dim=-1)
    assert torch.all(n[1:] > n[:-1])


# --------------------------------------------------------------------------- #
# 2. Exact no-op (bitwise, not almost-equal).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fill", [0.0, 1.9, -3.0, -1e9])
def test_subthreshold_zero_and_negative_rows_inject_exactly_zero(fill):
    torch.manual_seed(2)
    x = torch.randn(B, T, N_EMBD)
    s = _dose_site(threshold=2.0, channel_scale=[7.0] * R)
    a = torch.full((B, T, R), fill)
    assert torch.equal(s(x, a), x)


def test_only_the_supra_threshold_rows_move():
    torch.manual_seed(3)
    x = torch.randn(1, 3, N_EMBD)
    s = _dose_site(threshold=2.0)
    a = torch.zeros(1, 3, R)
    a[0, 1, 2] = 5.0                              # only the middle token fires
    y = s(x, a)
    assert torch.equal(y[0, 0], x[0, 0]) and torch.equal(y[0, 2], x[0, 2])
    assert not torch.equal(y[0, 1], x[0, 1])


def test_zero_channel_scale_is_an_exact_no_op():
    torch.manual_seed(4)
    x = torch.randn(B, T, N_EMBD)
    s = _dose_site(channel_scale=[0.0] * R)
    assert torch.equal(s(x, torch.randn(B, T, R).abs() * 10), x)


# --------------------------------------------------------------------------- #
# 3. D's row scale is a pure gauge -> loudness is not trainable through it.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factor", ["uniform", "per_row"])
def test_scaling_direction_rows_leaves_the_injection_unchanged(factor):
    # Exact as algebra; in fp32 the row renorm rounds, so this is a tight-tolerance
    # check. The power-of-two case below is the bitwise one.
    torch.manual_seed(5)
    x = torch.randn(B, T, N_EMBD)
    a = torch.randn(B, T, R) * 3.0
    s = _dose_site()
    base = s(x, a)
    with torch.no_grad():
        m = 13.7 if factor == "uniform" else torch.arange(1, R + 1, dtype=torch.float32).reshape(R, 1) * 0.37
        s.direction.mul_(m)
    assert torch.allclose(s(x, a), base, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("factor", [2.0, 0.25, 1024.0])
def test_power_of_two_row_scaling_is_bitwise_neutral(factor):
    torch.manual_seed(6)
    x = torch.randn(B, T, N_EMBD)
    a = torch.randn(B, T, R) * 3.0
    s = _dose_site()
    base = s(x, a)
    with torch.no_grad():
        s.direction.mul_(factor)     # exact in binary fp -> the gauge is exactly a gauge
    assert torch.equal(s(x, a), base)


# --------------------------------------------------------------------------- #
# 4. The sphere direction's gram survives row normalization (read-only).
# --------------------------------------------------------------------------- #
def test_sphere_direction_gram_is_preserved_by_row_normalization():
    d = np.load(SPHERE_NPZ)["D"].astype(np.float64)
    d_hat = d / np.sqrt(np.maximum((d ** 2).mean(1, keepdims=True), 1e-8))
    assert np.allclose(np.sqrt((d_hat ** 2).mean(1)), 1.0)

    def cos_gram(m):
        u = m / np.linalg.norm(m, axis=1, keepdims=True)
        return u @ u.T

    g0, g1 = cos_gram(d), cos_gram(d_hat)
    off = ~np.eye(d.shape[0], dtype=bool)
    assert np.allclose(g0, g1, atol=1e-12)
    assert np.abs(g0[off]).max() == pytest.approx(0.8192, abs=1e-3)   # deliberate, must survive
    # Rows are already unit-L2, so unit-RMS is one uniform sqrt(n_embd) rescale.
    assert np.allclose(d_hat / d, np.sqrt(d.shape[1]), rtol=1e-6)


# --------------------------------------------------------------------------- #
# Calibration.
# --------------------------------------------------------------------------- #
CONCEPTS = [f"c{i}" for i in range(4)]


def _loudness(layer=8, p50=0.03, active=None):
    return {"concepts": list(CONCEPTS),
            "subspace_total": {"ridge": {str(layer): {"p50": p50, "p90": 2 * p50,
                                                      "p95": 3 * p50, "p99": 4 * p50,
                                                      "max": 5 * p50}}},
            "ridge": {"active_loudness": {str(layer): {"p50": active or [0.02] * 4}}}}


class _SyntheticSource:
    """Channel c fires on ``rates[c]`` of tokens at magnitude ``mags[c]`` (above the
    2.0 threshold the site uses), deterministically in (seed)."""
    name = "synthetic"
    layer = 8
    concepts = list(CONCEPTS)

    def __init__(self, rates, mags, n_docs=64, n_per_doc=40, threshold=2.0, jitter=0.0):
        self.r = len(rates)
        self.rates, self.mags, self.jitter = rates, mags, jitter
        self._n_docs, self._n_per, self._th = n_docs, n_per_doc, threshold

    def sample_activation_rows(self, k, seed):
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(min(k, self._n_docs)):
            z = np.zeros((self._n_per, self.r), np.float32)
            for c in range(self.r):
                fire = rng.random(self._n_per) < self.rates[c]
                m = self.mags[c] * (1.0 + self.jitter * rng.standard_normal(int(fire.sum())))
                z[fire, c] = self._th + np.abs(m)
            rows.append(z)
        pooled = np.concatenate(rows)
        return pooled, len(rows), pooled.shape[0]


def _identity_direction(r, n_embd=N_EMBD):
    d = np.zeros((r, n_embd), np.float32)
    for i in range(r):
        d[i, i] = 1.0
    return torch.from_numpy(d)


def test_calibration_equalizes_per_event_not_per_total():
    # (a) equal conditional magnitude, 10x different firing rate -> EQUAL scale.
    src = _SyntheticSource(rates=[0.5, 0.05, 0.5, 0.05], mags=[1.0, 1.0, 1.0, 1.0])
    scale, meta = calibrate_dose_gate(src, _loudness(), _identity_direction(4),
                                      dial=1.0, threshold=2.0, seed=0, log=lambda *_: None)
    assert scale[0] == pytest.approx(scale[1], rel=1e-3)
    assert meta["n_events"][0] > 5 * meta["n_events"][1]          # rates really did differ
    # A per-TOTAL rule would have scaled the rare channel up by ~sqrt(10); it must not.
    assert scale[1] / scale[0] < 1.05

    # (b) equal firing rate, 2x different conditional magnitude -> scale ratio 2.
    src = _SyntheticSource(rates=[0.4, 0.4, 0.4, 0.4], mags=[1.0, 2.0, 1.0, 2.0])
    scale, meta = calibrate_dose_gate(src, _loudness(), _identity_direction(4),
                                      dial=1.0, threshold=2.0, seed=0, log=lambda *_: None)
    assert scale[0] / scale[1] == pytest.approx(2.0, rel=1e-6)
    assert meta["event_rms"][1] / meta["event_rms"][0] == pytest.approx(2.0, rel=1e-6)


def test_calibration_hits_the_target_median_and_reports_both_ladders():
    src = _SyntheticSource(rates=[0.3, 0.3, 0.3, 0.3], mags=[1.0, 2.0, 3.0, 4.0])
    d = _identity_direction(4)
    scale, meta = calibrate_dose_gate(src, _loudness(p50=0.03), d, dial=2.0,
                                      threshold=2.0, seed=0, log=lambda *_: None)
    assert meta["target_median"] == pytest.approx(2.0 * 0.02)   # 0.02 = subset median, not p50
    assert meta["injected_loudness"]["p50"] == pytest.approx(meta["target_median"], rel=1e-6)
    assert meta["donor_loudness"]["p95"] == pytest.approx(3 * 0.03)
    assert [meta["injected_loudness"][q] for q in ("p50", "p90", "p95", "p99", "max")] == \
        sorted(meta["injected_loudness"][q] for q in ("p50", "p90", "p95", "p99", "max"))

    # The reported median is what the SITE actually injects (same D, same relu).
    site = InjectionSite(InjectionCfg(name="s", r=4, threshold=2.0, channel_scale=scale), N_EMBD)
    with torch.no_grad():
        site.direction.copy_(d)
    rows, _, _ = src.sample_activation_rows(256, 0)
    x = torch.ones(1, rows.shape[0], N_EMBD)
    delta = site(x, torch.from_numpy(rows).unsqueeze(0)) - x
    loud = delta[0].detach().pow(2).mean(-1).sqrt()
    assert float(loud[loud > 0].median()) == pytest.approx(meta["target_median"], rel=1e-4)


def test_min_events_guard_imputes_the_median_well_measured_event_rms():
    # Channels 0-2 fire often at E=1,2,4 (median 2); channel 3 is under min_events.
    src = _SyntheticSource(rates=[0.5, 0.5, 0.5, 0.002], mags=[1.0, 2.0, 4.0, 9.0])
    logs = []
    scale, meta = calibrate_dose_gate(src, _loudness(), _identity_direction(4), dial=1.0,
                                      threshold=2.0, seed=0, min_events=50, log=logs.append)
    assert meta["n_events"][3] < 50 and meta["n_imputed"] == 1
    assert any("FALLBACK" in m and "min_events=50" in m and "c3" in m for m in logs)
    # Unit-consistent: the weight stays 1/E, with E imputed as the median of {1,2,4}.
    assert meta["event_rms_imputed"] == pytest.approx(2.0, rel=1e-6)
    assert meta["event_rms"][3] == pytest.approx(2.0, rel=1e-6)   # imputed, NOT its 9.0 sample
    assert scale[3] == pytest.approx(scale[1], rel=1e-6)          # == the channel whose E is 2
    # And emphatically not the donor loudness, which is a different unit entirely.
    assert scale[3] != pytest.approx(0.02, rel=1e-3)


def test_dose_requires_active_loudness_for_its_dial_reference():
    loud = _loudness()
    del loud["ridge"]        # no longer used by the min_events fallback, but IS the L_ref source
    with pytest.raises(ValueError, match="active_loudness"):
        calibrate_dose_gate(_SyntheticSource([0.3] * 4, [1.0] * 4), loud,
                            _identity_direction(4), log=lambda *_: None)


def test_all_channels_under_min_events_is_a_hard_error():
    src = _SyntheticSource(rates=[0.002] * 4, mags=[1.0] * 4)
    with pytest.raises(RuntimeError, match="no well-measured channel"):
        calibrate_dose_gate(src, _loudness(), _identity_direction(4), threshold=2.0,
                            seed=0, min_events=50, log=lambda *_: None)


def test_dead_channel_gets_zero_scale_and_a_loud_log():
    src = _SyntheticSource(rates=[0.5, 0.5, 0.5, 0.0], mags=[1.0, 1.0, 1.0, 1.0])
    logs = []
    scale, meta = calibrate_dose_gate(src, _loudness(), _identity_direction(4),
                                      dial=1.0, threshold=2.0, seed=0, log=logs.append)
    assert scale[3] == 0.0 and meta["n_dead"] == 1
    assert any("DEAD" in m and "c3" in m for m in logs)


def test_calibration_is_deterministic_across_simulated_ranks():
    # jitter>0 so the sampled magnitudes (not just the firing pattern) depend on the
    # seed — otherwise E_c is exact and determinism would be vacuous.
    def rank(seed):
        src = _SyntheticSource([0.3] * 4, [1.0, 2.0, 3.0, 4.0], jitter=0.25)
        return calibrate_dose_gate(src, _loudness(), _identity_direction(4), dial=1.0,
                                   threshold=2.0, seed=seed, log=lambda *_: None)[0]

    assert rank(7) == rank(7)
    assert loudness_vector_hash(rank(7)) == loudness_vector_hash(rank(7))
    assert loudness_vector_hash(rank(8)) != loudness_vector_hash(rank(7))   # the seed really is in play


# A realistic 54-concept loudness.json with the 4 season columns at 43..46 and the 7
# weekday columns at 47..53 — the layout that made the old exact-order check unusable.
SEASONS = ["autumn", "spring", "summer", "winter"]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
FULL54 = [f"f{i}" for i in range(43)] + SEASONS + WEEKDAYS


def _loudness54(layer=8, per_concept=None, subspace_p50=0.19):
    pc = per_concept or [0.005] * 43 + [0.030, 0.022, 0.026, 0.028] + [0.0273] * 7
    return {"concepts": list(FULL54),
            "subspace_total": {"ridge": {str(layer): {"p50": subspace_p50, "p90": 2 * subspace_p50,
                                                      "p95": 3 * subspace_p50, "p99": 4 * subspace_p50,
                                                      "max": 5 * subspace_p50}}},
            "ridge": {"active_loudness": {str(layer): {"p50": list(pc)}}}}


def test_subset_sources_resolve_and_values_follow_names_not_positions():
    # The blocker: a 4-of-54 source used to hard-fail the exact-order check. It must
    # now resolve, and a SHUFFLED column order must pick up each concept's own value.
    shuffled = ["winter", "autumn", "summer", "spring"]
    src = _SyntheticSource([0.3] * 4, [1.0] * 4)
    src.concepts = shuffled
    _, meta = calibrate_dose_gate(src, _loudness54(), _identity_direction(4), dial=1.0,
                                  threshold=2.0, seed=0, log=lambda *_: None)
    want = {"autumn": 0.030, "spring": 0.022, "summer": 0.026, "winter": 0.028}
    assert meta["concept_index"] == [FULL54.index(c) for c in shuffled] == [46, 43, 45, 44]
    assert meta["donor_per_concept"] == pytest.approx([want[c] for c in shuffled], rel=1e-6)


def test_dose_L_ref_is_the_subset_median_not_the_54_concept_subspace_total():
    src = _SyntheticSource([0.3] * 4, [1.0] * 4)
    src.concepts = list(SEASONS)
    _, meta = calibrate_dose_gate(src, _loudness54(), _identity_direction(4), dial=1.0,
                                  threshold=2.0, seed=0, log=lambda *_: None)
    assert meta["L_ref"] == pytest.approx(float(np.median([0.030, 0.022, 0.026, 0.028])))
    assert meta["L_ref_subspace_total"] == pytest.approx(0.19)
    assert meta["L_ref"] != pytest.approx(meta["L_ref_subspace_total"], rel=1e-3)
    # The whole-subspace reference would have over-injected this 4-of-54 site ~7x.
    assert meta["L_ref_subspace_total"] / meta["L_ref"] > 3.0
    assert meta["target_median"] == pytest.approx(meta["L_ref"])   # dial 1.0


def test_dose_L_ref_reproduces_the_hand_set_weekday_gate():
    # The 7 weekday columns are all 0.0273 -> dial 1.0 reproduces the campaign's abs:0.0273.
    src = _SyntheticSource([0.3] * 7, [1.0] * 7)
    src.concepts = list(WEEKDAYS)
    _, meta = calibrate_dose_gate(src, _loudness54(), _identity_direction(7), dial=1.0,
                                  threshold=2.0, seed=0, log=lambda *_: None)
    assert meta["L_ref"] == pytest.approx(0.0273)


def test_calibration_rejects_an_unknown_concept_name():
    src = _SyntheticSource([0.3] * 4, [1.0] * 4)
    src.concepts = ["autumn", "spring", "summer", "not_a_concept"]
    with pytest.raises(ValueError, match="not_a_concept"):
        calibrate_dose_gate(src, _loudness54(), _identity_direction(4), log=lambda *_: None)


def test_abs_target_sets_the_median_directly_and_ignores_the_dial():
    src = _SyntheticSource([0.3] * 4, [1.0, 2.0, 3.0, 4.0])
    d = _identity_direction(4)
    scale, meta = calibrate_dose_gate(src, None, d, abs_target=0.07, threshold=2.0,
                                      seed=0, log=lambda *_: None)
    assert meta["mode"] == "abs" and meta["dial"] is None
    assert meta["target_median"] == 0.07
    assert meta["injected_loudness"]["p50"] == pytest.approx(0.07, rel=1e-9)
    # Same rows, dial mode: only the target differs, the per-channel MIX is identical.
    dial_scale, dial_meta = calibrate_dose_gate(src, _loudness(), d, dial=1.0, threshold=2.0,
                                                seed=0, log=lambda *_: None)
    ratio = [a / b for a, b in zip(scale, dial_scale)]
    assert ratio == pytest.approx([0.07 / dial_meta["target_median"]] * 4, rel=1e-6)


class _CoFireSource:
    """``n_cofire`` channels fire on the SAME tokens — the multi-family case (a date
    token is winter + january + friday at once)."""
    name = "cofire"
    layer = 8
    concepts = list(CONCEPTS)
    r = 4

    def __init__(self, n_cofire, rate=0.4, mag=4.0):
        self.n_cofire, self.rate, self.mag = n_cofire, rate, mag

    def sample_activation_rows(self, k, seed):
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(64):
            z = np.zeros((40, self.r), np.float32)
            fire = rng.random(40) < self.rate
            for c in range(self.n_cofire):
                z[fire, c] = self.mag
            rows.append(z)
        pooled = np.concatenate(rows)
        return pooled, 64, pooled.shape[0]


@pytest.mark.parametrize("n_cofire", [1, 2, 3, 4])
def test_calibration_targets_total_per_token_loudness_not_per_channel(n_cofire):
    """Co-firing must NOT multiply the injected total. The median is taken over
    rms(u @ D_hat) per token on the REAL firing pattern, so the calibration
    self-corrects for co-firing and multi-family needs no new machinery."""
    d = _identity_direction(4)
    scale, meta = calibrate_dose_gate(_CoFireSource(n_cofire), _loudness(), d, dial=1.0,
                                      threshold=2.0, seed=0, min_events=1, log=lambda *_: None)
    assert meta["injected_loudness"]["p50"] == pytest.approx(meta["target_median"], rel=1e-6)
    # Per-channel scale absorbs the co-firing: it shrinks as 1/sqrt(n) (quadrature).
    single = calibrate_dose_gate(_CoFireSource(1), _loudness(), d, dial=1.0, threshold=2.0,
                                 seed=0, min_events=1, log=lambda *_: None)[0]
    assert scale[0] == pytest.approx(single[0] / np.sqrt(n_cofire), rel=1e-6)

    # And the SITE reproduces that total on the same rows — not n_cofire x it.
    site = InjectionSite(InjectionCfg(name="s", r=4, threshold=2.0, channel_scale=scale), N_EMBD)
    with torch.no_grad():
        site.direction.copy_(d)
    rows, _, _ = _CoFireSource(n_cofire).sample_activation_rows(256, 0)
    x = torch.ones(1, rows.shape[0], N_EMBD)
    loud = (site(x, torch.from_numpy(rows).unsqueeze(0)) - x)[0].detach().pow(2).mean(-1).sqrt()
    assert float(loud[loud > 0].median()) == pytest.approx(meta["target_median"], rel=1e-4)


# --------------------------------------------------------------------------- #
# Per-concept (length-r) thresholds.
# --------------------------------------------------------------------------- #
def test_threshold_accepts_a_length_r_vector_in_the_site():
    z0 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    s = _dose_site(threshold=z0)
    assert s.threshold.shape == (R,)
    x = torch.ones(1, 1, N_EMBD)                   # rms(x) == 1, so the delta is bare
    a = torch.full((1, 1, R), 3.5)                 # channels 0,1,2 clear their knee
    u = torch.tensor([2.5, 1.5, 0.5, 0.0, 0.0, 0.0])
    d_hat = s.direction / s.direction.pow(2).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    assert torch.allclose(s(x, a) - x, u @ d_hat, rtol=1e-6, atol=1e-7)
    # A channel exactly at its own knee contributes bitwise nothing.
    a2 = torch.tensor(z0).reshape(1, 1, R)
    assert torch.equal(s(torch.zeros(1, 1, N_EMBD), a2), torch.zeros(1, 1, N_EMBD))


def test_threshold_vector_is_honoured_by_calibration():
    # Channel c fires at 2+c+1; with a per-channel knee of 2+c, each clears by exactly 1,
    # so per-event equalization gives every channel the SAME scale.
    src = _SyntheticSource([0.4] * 4, [1.0, 2.0, 3.0, 4.0])
    z0 = [2.0, 3.0, 4.0, 5.0]
    scale, meta = calibrate_dose_gate(src, _loudness(), _identity_direction(4), dial=1.0,
                                      threshold=z0, seed=0, log=lambda *_: None)
    assert meta["threshold"] == pytest.approx(z0)
    assert meta["event_rms"] == pytest.approx([1.0] * 4, rel=1e-6)
    assert scale == pytest.approx([scale[0]] * 4, rel=1e-6)


def test_threshold_vector_of_the_wrong_length_is_refused():
    with pytest.raises(AssertionError, match="threshold must be"):
        _dose_site(threshold=[2.0, 3.0])


# --------------------------------------------------------------------------- #
# Realized-loudness check (post-alignment).
# --------------------------------------------------------------------------- #
def _mean_pool_align(rows, k):
    """Collapse each group of k gemma rows into one nanochat row by MEAN — the
    align_policy='mean' shrinkage that makes calibration under-estimate."""
    n = (rows.shape[0] // k) * k
    return rows[:n].reshape(-1, k, rows.shape[1]).mean(1)


def test_realized_report_matches_the_site_on_the_same_rows():
    src = _SyntheticSource([0.3] * 4, [1.0, 2.0, 3.0, 4.0])
    d = _identity_direction(4)
    scale, meta = calibrate_dose_gate(src, _loudness(), d, dial=1.0, threshold=2.0,
                                      seed=0, log=lambda *_: None)
    rows, _, _ = src.sample_activation_rows(256, 0)
    rep = realized_loudness_report([rows], 2.0, scale, d, meta["target_median"],
                                   min_firing_tokens=10 ** 9)
    site = InjectionSite(InjectionCfg(name="s", r=4, threshold=2.0, channel_scale=scale), N_EMBD)
    with torch.no_grad():
        site.direction.copy_(d)
    x = torch.ones(1, rows.shape[0], N_EMBD)
    loud = (site(x, torch.from_numpy(rows).unsqueeze(0)) - x)[0].detach().pow(2).mean(-1).sqrt()
    loud = loud[loud > 0]
    assert rep["n_firing_tokens"] == loud.numel()
    assert rep["ladder"]["p50"] == pytest.approx(float(loud.median()), rel=1e-4)
    # Unaligned rows are exactly what calibration saw, so the target is hit.
    assert rep["within_tol"] and abs(rep["rel_dev"]) < 1e-3


def test_mean_pool_alignment_makes_the_site_quieter_than_calibration_predicted():
    # The reason the check exists. relu is convex, so relu(mean(z)-z0) <= mean(relu(z-z0)):
    # calibration scores each gemma token, the site scores the POOLED nanochat token, and
    # pooling a firing token with sub-threshold neighbours drags it under the knee. The
    # dial therefore comes out too SMALL, and the correction factor must exceed 1.
    src = _SyntheticSource([0.5] * 4, [2.0] * 4)          # fires at z=4.0, z0=2.0
    d = _identity_direction(4)
    scale, meta = calibrate_dose_gate(src, _loudness(), d, dial=1.0, threshold=2.0,
                                      seed=0, log=lambda *_: None)
    rows, _, _ = src.sample_activation_rows(256, 0)
    rep = realized_loudness_report([_mean_pool_align(rows, 4)], 2.0, scale, d,
                                   meta["target_median"], min_firing_tokens=10 ** 9)
    assert rep["realized_p50"] < meta["target_median"]     # quieter, as convexity predicts
    assert rep["correction"] > 1.0                         # the dial must be raised
    assert not rep["within_tol"]
    assert rep["correction"] == pytest.approx(meta["target_median"] / rep["realized_p50"], rel=1e-9)
    # Sanity: the SAME rows unpooled do hit the target, so pooling is the whole effect.
    assert realized_loudness_report([rows], 2.0, scale, d, meta["target_median"],
                                    min_firing_tokens=10 ** 9)["within_tol"]


def test_realized_check_warns_loudly_and_reports_a_correction_without_applying_it():
    d = _identity_direction(4)
    rows = np.full((2000, 4), 4.0, np.float32)          # a_eff = 2 everywhere
    scale = [1.0, 0.0, 0.0, 0.0]
    target = 0.03
    rep = realized_loudness_report([rows], 2.0, scale, d, target, min_firing_tokens=10 ** 9)
    logs = []
    log_realized_loudness(rep, "acts", {"p50": target}, {"p50": 0.03}, logs.append)
    assert not rep["within_tol"]
    assert rep["correction"] == pytest.approx(target / rep["realized_p50"], rel=1e-9)
    joined = "\n".join(logs)
    assert "WARNING" in joined and "NOT auto-corrected" in joined
    assert f"{rep['correction']:.4f}" in joined
    assert "realized=" in joined and "calib=" in joined and "donor=" in joined
    assert scale == [1.0, 0.0, 0.0, 0.0]                 # the caller's scale is untouched


def test_realized_check_stops_pulling_once_it_has_enough_firing_tokens():
    d = _identity_direction(4)
    pulled = []

    def batches():
        for i in range(100):
            pulled.append(i)
            yield np.full((512, 4), 4.0, np.float32)

    rep = realized_loudness_report(batches(), 2.0, [1.0] * 4, d, 0.03, min_firing_tokens=1000)
    assert rep["n_firing_tokens"] >= 1000
    assert len(pulled) == 2 and rep["n_batches"] == 2   # exactly what the caller must replay


def test_realized_check_reports_a_silent_site_instead_of_crashing():
    d = _identity_direction(4)
    rep = realized_loudness_report([np.zeros((256, 4), np.float32)], 2.0, [1.0] * 4, d, 0.03,
                                   min_firing_tokens=1000)
    assert rep["ladder"] is None and rep["n_firing_tokens"] == 0 and not rep["within_tol"]
    logs = []
    log_realized_loudness(rep, "acts", None, None, logs.append)
    assert any("NO token injected" in m for m in logs)


def test_realized_check_accepts_batched_BTr_tensors():
    d = _identity_direction(4)
    a = torch.full((2, 8, 4), 4.0)
    rep = realized_loudness_report([a], 2.0, [1.0] * 4, d, 0.03, min_firing_tokens=10 ** 9)
    assert rep["n_tokens"] == 16 and rep["n_firing_tokens"] == 16


def test_peek_and_replay_skips_no_batches_and_keeps_order():
    """Mirrors injection_train's peek/replay exactly: the check pulls batches, then
    chain() puts them back in front so training consumes every batch, in order."""
    import itertools
    d = _identity_direction(4)
    stream = ((i, np.full((512, 4), 4.0, np.float32)) for i in range(20))
    first = next(stream)
    peeked = []

    def peek_acts(max_batches=32):
        yield first[1]
        i = 0
        while i < max_batches:
            if i == len(peeked):
                peeked.append(next(stream))
            yield peeked[i][1]
            i += 1

    rep = realized_loudness_report(peek_acts(), 2.0, [1.0] * 4, d, 0.03, min_firing_tokens=1000)
    assert rep["n_batches"] == 2 and len(peeked) == 1     # first + one pulled
    train = itertools.chain(peeked, stream)
    assert [first[0]] + [b[0] for b in train] == list(range(20))   # nothing skipped, order intact


def test_peek_respects_the_max_batches_cap():
    d = _identity_direction(4)
    stream = (np.zeros((8, 4), np.float32) for _ in range(1000))   # never fires
    peeked = []

    def peek_acts(max_batches=5):
        yield next(stream)
        i = 0
        while i < max_batches:
            if i == len(peeked):
                peeked.append(next(stream))
            yield peeked[i]
            i += 1

    rep = realized_loudness_report(peek_acts(), 2.0, [1.0] * 4, d, 0.03, min_firing_tokens=10 ** 9)
    assert len(peeked) == 5 and rep["ladder"] is None    # capped, and reported as silent


# --------------------------------------------------------------------------- #
# Optimizability contract.
# --------------------------------------------------------------------------- #
def test_dose_extras_are_never_optimized_and_survive_a_reassert():
    sites = build_sites([dict(name="acts", r=R, after_block=0,
                              channel_scale=[1.0] * R, trainable_direction=True)], N_EMBD)
    s = sites["acts"]
    for p in (s.channel_scale, s.threshold):
        assert getattr(p, "_never_optimize", False) and p.requires_grad
        p._never_optimize = False                # simulate load_state_dict(assign=True)
        p.requires_grad_(False)
    reassert_optimizability(sites)
    for p in (s.channel_scale, s.threshold):
        assert p._never_optimize and p.requires_grad
    assert s.direction.requires_grad


def test_uncalibrated_site_cannot_be_built():
    cfg = InjectionCfg(name="acts", r=R)
    assert cfg.channel_scale is None          # must be calibrated, never silently 1.0
    with pytest.raises(AssertionError, match="channel_scale is unresolved"):
        InjectionSite(cfg, N_EMBD)
    with pytest.raises(AssertionError, match="length-6"):
        _dose_site(channel_scale=[1.0, 2.0])


def test_dose_cfg_round_trips_through_checkpoint_meta_and_assign_load():
    # The resume contract: channel_scale/threshold persist via asdict(cfg) into
    # injection_sites_config, rebuild the same site, and survive assign=True.
    cs = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66]
    cfg = InjectionCfg(name="acts", r=R, after_block=0, threshold=2.0, channel_scale=cs)
    meta = json.loads(json.dumps(asdict(cfg)))
    assert meta["channel_scale"] == cs and meta["threshold"] == 2.0 and meta["loudness"] == 1.0

    sites = build_sites([InjectionCfg(**meta)], N_EMBD)
    sd = sites.state_dict()
    assert {"acts.channel_scale", "acts.threshold"} <= set(sd)

    fresh = build_sites([InjectionCfg(**meta)], N_EMBD)
    fresh.load_state_dict(sd, assign=True)
    reassert_optimizability(fresh)
    s = fresh["acts"]
    assert torch.equal(s.channel_scale, torch.tensor(cs))
    assert s.channel_scale._never_optimize and s.threshold._never_optimize

    x, a = torch.randn(B, T, N_EMBD), torch.randn(B, T, R) * 4
    assert torch.equal(s(x, a), sites["acts"](x, a))


def test_dose_gradients_reach_a_trainable_direction_but_not_the_scale():
    s = _dose_site(trainable_direction=True)
    x = torch.randn(B, T, N_EMBD, requires_grad=True)
    a = torch.randn(B, T, R) * 4.0
    s(x, a).sum().backward()
    assert s.direction.grad is not None and s.direction.grad.abs().sum() > 0
    assert s.channel_scale.grad is not None      # loggable want-signal, never stepped
