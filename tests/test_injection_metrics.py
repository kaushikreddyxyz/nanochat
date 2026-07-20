"""Injection health metrics (nanochat/injection/metrics.py): correctness on synthetic
firing patterns, per-window reset, the DDP reduce path with simulated ranks, and the
guarantee that a metrics bug degrades to "no metrics" instead of killing the step."""
import numpy as np
import pytest
import torch

from nanochat.injection.metrics import HIST_BINS, HIST_HI, HIST_LO, InjectionMetrics, saturation_bounds
from nanochat.injection.sites import InjectionCfg, InjectionSite

N_EMBD = 32


def make_site(r=4, threshold=2.0, channel_scale=None, trainable=False, seed=1337):
    cfg = InjectionCfg(name="probes", r=r, threshold=threshold, direction_seed=seed,
                       trainable_direction=trainable,
                       channel_scale=list(channel_scale if channel_scale is not None else [1.0] * r))
    return InjectionSite(cfg, N_EMBD)


def make_metrics(site, **kw):
    kw.setdefault("targets", {"probes": 0.03})
    kw.setdefault("concepts", {"probes": [f"c{i}" for i in range(site.cfg.r)]})
    kw.setdefault("log_every", 10)
    return InjectionMetrics({"probes": site}, **kw)


def ref_loudness(a, site):
    """rms(u @ D_hat) per token, the long way — the quantity the site's injection has
    relative to rms(x), computed without the Gram shortcut metrics.py uses."""
    a = np.asarray(a, np.float64).reshape(-1, site.cfg.r)
    z0 = np.asarray(site.threshold.detach().numpy(), np.float64)
    w = np.asarray(site.channel_scale.detach().numpy(), np.float64)
    d = np.asarray(site.direction.detach().numpy(), np.float64)
    d_hat = d / np.sqrt(np.maximum((d ** 2).mean(1, keepdims=True), 1e-8))
    u = np.maximum(a - z0, 0.0) * w
    z = u @ d_hat
    return np.sqrt((z ** 2).mean(1))


class FakeSource:
    """Minimal ActivationSource stand-in exposing the bookkeeping metrics reads."""

    def __init__(self, r, concepts=None, counts=None, depth=None, quant=None):
        self.r = r
        self.concepts = concepts or [f"c{i}" for i in range(r)]
        self._counts = counts if counts is not None else {"ok": 0, "miss_shard": 0, "miss_row": 0, "drift": 0}
        self._depth = depth
        if quant is not None:
            self._zero, self._scale, self._mean, self._std = quant

    def stats(self):
        return dict(self._counts)

    def pool_depth_stats(self):
        return self._depth


# --------------------------------------------------------------------------- #
# Tier 1: known firing pattern -> expected rates and loudness
# --------------------------------------------------------------------------- #
def test_known_firing_pattern_gives_expected_rates():
    site = make_site(r=4, threshold=2.0)
    m = make_metrics(site)
    # 10 tokens. ch0 fires on 5, ch1 on 2, ch2 on 1, ch3 never (dead).
    a = torch.zeros(1, 10, 4)
    a[0, :5, 0] = 5.0     # a_eff = 3.0
    a[0, :2, 1] = 4.0     # a_eff = 2.0
    a[0, 0, 2] = 6.0      # a_eff = 4.0
    a[0, :, 3] = 1.0      # below threshold everywhere
    m.observe({"probes": a})
    out = m.flush(0)

    assert out["inj/probes/fire_rate/c0"] == pytest.approx(0.5)
    assert out["inj/probes/fire_rate/c1"] == pytest.approx(0.2)
    assert out["inj/probes/fire_rate/c2"] == pytest.approx(0.1)
    assert out["inj/probes/fire_rate/c3"] == pytest.approx(0.0)
    assert out["inj/probes/aeff_mean/c0"] == pytest.approx(3.0)
    assert out["inj/probes/aeff_mean/c1"] == pytest.approx(2.0)
    assert out["inj/probes/aeff_mean/c2"] == pytest.approx(4.0)
    assert out["inj/probes/aeff_mean/c3"] == pytest.approx(0.0)
    assert out["inj/probes/row_fire_rate"] == pytest.approx(0.5)   # rows 0-4 fire
    assert out["inj/probes/fire_rate_min"] == pytest.approx(0.0)
    assert out["inj/probes/fire_rate_max"] == pytest.approx(0.5)
    # 8 channel-firings over 5 firing rows
    assert out["inj/probes/cofire_mean"] == pytest.approx(8 / 5)
    assert out["inj/probes/zero_row_rate"] == pytest.approx(0.0)   # ch3=1.0 everywhere


def test_dead_channel_is_counted():
    site = make_site(r=4, threshold=2.0)
    m = make_metrics(site)
    a = torch.zeros(1, 8, 4)
    a[0, :, :3] = 5.0     # ch3 silent
    m.observe({"probes": a})
    out = m.flush(0)
    assert out["inj/probes/dead_channels"] == 1.0
    assert out["inj/probes/fire_rate/c3"] == 0.0


def test_realized_loudness_matches_direct_projection():
    site = make_site(r=5, threshold=2.0, channel_scale=[0.01, 0.02, 0.005, 0.03, 0.015])
    m = make_metrics(site, targets={"probes": 0.02})
    g = torch.Generator().manual_seed(7)
    a = torch.randn(2, 64, 5, generator=g) * 2.0 + 1.0
    m.observe({"probes": a})
    out = m.flush(0)

    ref = ref_loudness(a.numpy(), site)
    fired = ref[ref > 0]
    assert out["inj/probes/loudness_mean"] == pytest.approx(fired.mean(), rel=1e-5)
    # p50 comes from a log2 histogram, so it is exact only to the bin width
    bin_w = 2.0 ** ((HIST_HI - HIST_LO) / HIST_BINS)
    assert out["inj/probes/loudness_p50"] == pytest.approx(np.median(fired), rel=bin_w - 1)
    assert out["inj/probes/loudness_ratio_p50"] == pytest.approx(out["inj/probes/loudness_p50"] / 0.02)
    assert out["inj/probes/loudness_target"] == pytest.approx(0.02)


def test_loudness_equals_the_real_sites_injected_over_residual_rms():
    """The load-bearing shortcut: metrics never touch x, because the site injects
    rms(x) * (u @ D_hat) and so rms(injected)/rms(x) is independent of x. Pin that
    against the actual InjectionSite.forward on a residual with a non-trivial scale."""
    site = make_site(r=4, threshold=2.0, channel_scale=[0.02, 0.01, 0.03, 0.015])
    g = torch.Generator().manual_seed(3)
    a = torch.randn(2, 32, 4, generator=g) * 2.0 + 1.5
    x = torch.randn(2, 32, N_EMBD, generator=g) * 11.0        # deliberately not unit-scale
    with torch.no_grad():
        injected = site(x, a) - x
    rms = lambda t: t.pow(2).mean(-1).sqrt()
    per_token = (rms(injected) / rms(x)).reshape(-1).numpy()

    m = make_metrics(site, targets={"probes": 0.02})
    m.observe({"probes": a})
    out = m.flush(0)
    fired = per_token[per_token > 0]
    assert out["inj/probes/loudness_mean"] == pytest.approx(fired.mean(), rel=1e-4)


def test_loudness_ratio_is_one_when_realized_hits_target():
    site = make_site(r=3, threshold=2.0)
    g = torch.Generator().manual_seed(11)
    a = torch.randn(1, 512, 3, generator=g).abs() * 2.0 + 2.0
    target = float(np.median(ref_loudness(a.numpy(), site)))
    m = make_metrics(site, targets={"probes": target})
    m.observe({"probes": a})
    out = m.flush(0)
    assert out["inj/probes/loudness_ratio_p50"] == pytest.approx(1.0, rel=0.1)


def test_zero_rows_and_nonfinite_counted():
    site = make_site(r=3, threshold=2.0)
    m = make_metrics(site)
    a = torch.zeros(1, 10, 3)
    a[0, :4, :] = 5.0                 # 4 live rows, 6 exact-zero rows
    a[0, 9, 0] = float("nan")         # one non-finite entry (still a "zero" row otherwise)
    m.observe({"probes": a})
    out = m.flush(0)
    assert out["inj/probes/zero_row_rate"] == pytest.approx(0.5)   # nan row counts as non-zero
    assert out["inj/probes/nonfinite_acts"] == 1.0
    assert out["inj/probes/nonfinite_injected"] == 0.0             # nan is scrubbed before projection


def test_alignment_rates_from_source_stats():
    site = make_site(r=2, threshold=2.0)
    src = FakeSource(2, counts={"ok": 0, "miss_shard": 0, "miss_row": 0, "drift": 0},
                     depth={"p50": 1, "p90": 2, "p99": 4, "mean": 1.4, "zero_coverage_rate": 0.02})
    m = make_metrics(site, sources={"probes": src})
    src._counts.update({"ok": 90, "drift": 8, "miss_shard": 2})
    m.observe({"probes": torch.full((1, 4, 2), 5.0)})
    out = m.flush(0)
    assert out["inj/probes/align_ok_rate"] == pytest.approx(0.9)
    assert out["inj/probes/align_drift_rate"] == pytest.approx(0.08)
    assert out["inj/probes/align_miss_rate"] == pytest.approx(0.02)
    assert out["inj/probes/no_coverage_rate"] == pytest.approx(0.02)
    assert out["inj/probes/pool_depth_p90"] == 2
    # counters are cumulative on the source; the metric is the per-window DELTA
    src._counts.update({"ok": 190, "drift": 8, "miss_shard": 2})
    m.observe({"probes": torch.full((1, 4, 2), 5.0)})
    out2 = m.flush(10)
    assert out2["inj/probes/align_ok_rate"] == pytest.approx(1.0)
    assert out2["inj/probes/align_docs"] == pytest.approx(100.0)


def test_loss_split_over_injected_and_uninjected_tokens():
    site = make_site(r=2, threshold=2.0)
    m = make_metrics(site, loss_split=True)
    a = torch.zeros(1, 6, 2)
    a[0, :3, 0] = 5.0                        # tokens 0-2 inject, 3-5 do not
    per_tok = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    valid = torch.ones(6, dtype=torch.bool)
    m.observe({"probes": a}, per_token_loss=per_tok, valid=valid)
    out = m.flush(0)
    assert out["inj/probes/loss_injected"] == pytest.approx(2.0)
    assert out["inj/probes/loss_uninjected"] == pytest.approx(20.0)
    assert out["inj/probes/loss_gap"] == pytest.approx(-18.0)


def test_loss_split_honours_the_valid_mask():
    site = make_site(r=2, threshold=2.0)
    m = make_metrics(site, loss_split=True)
    a = torch.zeros(1, 4, 2)
    a[0, :2, 0] = 5.0
    per_tok = torch.tensor([1.0, 999.0, 10.0, 999.0])
    valid = torch.tensor([True, False, True, False])
    m.observe({"probes": a}, per_token_loss=per_tok, valid=valid)
    out = m.flush(0)
    assert out["inj/probes/loss_injected"] == pytest.approx(1.0)
    assert out["inj/probes/loss_uninjected"] == pytest.approx(10.0)


def test_loss_split_reduction_matches_the_default_mean():
    """--injection-loss-split trades reduction='mean' for reduction='none' + an explicit
    masked mean. Pin that the training signal is unchanged (to float tolerance), which is
    the whole basis for offering the flag."""
    from nanochat.gpt import GPT, GPTConfig
    cfg = GPTConfig(sequence_len=64, vocab_size=64, n_layer=2, n_head=2,
                    n_kv_head=2, n_embd=64, window_pattern="L")
    torch.manual_seed(0)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    torch.manual_seed(0)
    model.init_weights()
    x = torch.randint(0, 60, (2, 16))
    y = torch.randint(0, 60, (2, 16))
    y[0, 3] = -1                                      # an ignored target must not dilute the mean
    with torch.no_grad():
        per_tok = model(x, y, loss_reduction='none')
        valid = y.reshape(-1) != -1
        split = float(per_tok.sum() / valid.sum().clamp_min(1))
        assert split == pytest.approx(float(model(x, y)), rel=1e-6)


# --------------------------------------------------------------------------- #
# Tier 2: geometry
# --------------------------------------------------------------------------- #
def test_gram_offdiagonal_detects_parallel_rows():
    site = make_site(r=3, threshold=2.0, trainable=True)
    m = make_metrics(site)
    m.observe({"probes": torch.full((1, 4, 3), 5.0)})
    out_orth = m.flush(0)
    assert out_orth["inj/probes/gram_offdiag_absmax"] < 1e-4     # orthonormal init

    with torch.no_grad():                                        # rotate row 1 onto row 0
        site.direction[1] = site.direction[0] * 3.0
    m.observe({"probes": torch.full((1, 4, 3), 5.0)})
    out_par = m.flush(10)
    assert out_par["inj/probes/gram_offdiag_absmax"] == pytest.approx(1.0, abs=1e-4)
    assert out_par["inj/probes/d_row_rms_max"] / out_par["inj/probes/d_row_rms_min"] > 2.0


def test_cosine_drift_from_init():
    site = make_site(r=3, threshold=2.0, trainable=True)
    m = make_metrics(site)
    m.observe({"probes": torch.full((1, 4, 3), 5.0)})
    assert m.flush(0)["inj/probes/dir_cos_init_min"] == pytest.approx(1.0, abs=1e-5)

    with torch.no_grad():
        site.direction[0] = -site.direction[0]                   # a row rotated 180 degrees
    m.observe({"probes": torch.full((1, 4, 3), 5.0)})
    out = m.flush(10)
    assert out["inj/probes/dir_cos_init_min"] == pytest.approx(-1.0, abs=1e-5)
    assert out["inj/probes/dir_cos_init_mean"] == pytest.approx(1 / 3, abs=1e-5)


def test_row_rms_is_gauge_invariant_but_reported():
    site = make_site(r=2, threshold=2.0, trainable=True)
    m = make_metrics(site)
    with torch.no_grad():
        site.direction.mul_(7.0)                                  # pure gauge: injection unchanged
    m.observe({"probes": torch.full((1, 4, 2), 5.0)})
    out = m.flush(0)
    assert out["inj/probes/d_row_rms_mean"] == pytest.approx(7.0 / np.sqrt(N_EMBD), rel=1e-4)
    assert out["inj/probes/dir_cos_init_min"] == pytest.approx(1.0, abs=1e-5)


def test_gradient_signals_including_the_never_stepped_loudness_param():
    site = make_site(r=3, threshold=2.0, channel_scale=[0.1, 0.2, 0.3], trainable=True)
    m = make_metrics(site)
    site.direction.grad = torch.full_like(site.direction, 0.5)
    site.channel_scale.grad = torch.tensor([-1.0, -2.0, -3.0])    # negative => wants louder
    site.threshold.grad = torch.tensor(0.25)
    m.observe_grads()
    m.observe_grads()                                             # averaged over grad steps
    m.observe({"probes": torch.full((1, 4, 3), 5.0)})
    out = m.flush(0)
    assert out["inj/probes/grad_norm_direction"] == pytest.approx(0.5 * np.sqrt(3 * N_EMBD), rel=1e-5)
    assert out["inj/probes/grad_loudness/c1"] == pytest.approx(-2.0)
    assert out["inj/probes/grad_loudness_mean"] == pytest.approx(-2.0)
    assert out["inj/probes/grad_loudness_absmax"] == pytest.approx(3.0)
    assert out["inj/probes/grad_threshold_absmean"] == pytest.approx(0.25)
    w = np.array([0.1, 0.2, 0.3])
    assert out["inj/probes/want_louder"] == pytest.approx(
        -np.dot([-1.0, -2.0, -3.0], w) / np.linalg.norm(w))
    assert out["inj/probes/want_louder"] > 0


# --------------------------------------------------------------------------- #
# Tier 3: saturation
# --------------------------------------------------------------------------- #
def test_saturation_bounds_from_store_quantization():
    r = 2
    quant = (np.zeros(r), np.ones(r), np.zeros(r), np.ones(r))   # zero=0 scale=1 mean=0 std=1
    lo, hi = saturation_bounds(FakeSource(r, quant=quant), r)
    assert hi == pytest.approx([127.0, 127.0])
    assert lo == pytest.approx([-128.0, -128.0])

    site = make_site(r=r, threshold=2.0)
    m = make_metrics(site, sources={"probes": FakeSource(r, quant=quant)})
    a = torch.full((1, 5, r), 3.0)
    a[0, 0, 0] = 127.0                                            # one clipped entry of 10
    m.observe({"probes": a})
    assert m.flush(0)["inj/probes/saturation_rate"] == pytest.approx(0.1)


def test_saturation_absent_for_sources_without_quantization():
    site = make_site(r=2, threshold=2.0)
    m = make_metrics(site, sources={"probes": FakeSource(2)})
    m.observe({"probes": torch.full((1, 4, 2), 5.0)})
    assert "inj/probes/saturation_rate" not in m.flush(0)


# --------------------------------------------------------------------------- #
# Windowing, DDP, failure containment
# --------------------------------------------------------------------------- #
def test_accumulator_resets_per_window():
    site = make_site(r=2, threshold=2.0)
    m = make_metrics(site)
    a = torch.zeros(1, 10, 2)
    a[0, :5, 0] = 5.0
    m.observe({"probes": a})
    assert m.flush(0)["inj/probes/fire_rate/c0"] == pytest.approx(0.5)
    # nothing observed since: the window is empty, not carrying the previous one forward
    assert m.flush(10) == {}
    a2 = torch.zeros(1, 10, 2)
    a2[0, :1, 0] = 5.0
    m.observe({"probes": a2})
    assert m.flush(20)["inj/probes/fire_rate/c0"] == pytest.approx(0.1)


def test_should_log_respects_the_interval():
    site = make_site(r=2, threshold=2.0)
    m = make_metrics(site, log_every=50)
    assert m.should_log(0) and m.should_log(50) and m.should_log(100)
    assert not m.should_log(1) and not m.should_log(49)
    off = make_metrics(make_site(r=2), log_every=0)
    assert not off.enabled and not off.should_log(0)


def test_ddp_reduce_path_with_simulated_ranks():
    """Two ranks see different halves of the stream; the reduced metrics must equal the
    metrics of the pooled stream."""
    a0 = torch.zeros(1, 10, 3)
    a0[0, :5, 0] = 5.0
    a0[0, :1, 1] = 4.0
    a1 = torch.zeros(1, 10, 3)
    a1[0, :3, 0] = 6.0
    a1[0, :2, 2] = 4.0

    packets = {}

    def rank_reduce(rank):
        def _reduce(t):
            packets[rank] = t.clone()
            if len(packets) < 2:
                return t                          # first rank's flush is a placeholder
            return packets[0] + packets[1]
        return _reduce

    sites = [make_site(r=3, threshold=2.0) for _ in range(2)]
    m0 = make_metrics(sites[0], reduce_fn=rank_reduce(0))
    m1 = make_metrics(sites[1], reduce_fn=rank_reduce(1))
    m0.observe({"probes": a0})
    m1.observe({"probes": a1})
    m0.flush(0)                                    # populates packets[0]
    out = m1.flush(0)                              # sees the summed packet

    pooled = make_metrics(make_site(r=3, threshold=2.0))
    pooled.observe({"probes": torch.cat([a0, a1], dim=1)})
    ref = pooled.flush(0)
    for k in ("fire_rate/c0", "fire_rate/c1", "fire_rate/c2", "row_fire_rate",
              "cofire_mean", "aeff_mean/c0", "loudness_mean", "dead_channels"):
        assert out["inj/probes/" + k] == pytest.approx(ref["inj/probes/" + k], rel=1e-6), k


def test_flush_still_reduces_when_metrics_are_disabled():
    """DDP safety: a rank whose metrics died must still enter the collective, or the
    healthy ranks hang forever on the all_reduce."""
    site = make_site(r=2, threshold=2.0)
    calls = []
    m = make_metrics(site, reduce_fn=lambda t: (calls.append(t.numel()), t)[1])
    m.enabled = False                              # simulate a mid-run metrics failure
    assert m.should_log(0)
    assert m.flush(0) == {}
    assert len(calls) == 1


def test_exception_inside_metrics_does_not_kill_the_step():
    site = make_site(r=2, threshold=2.0)
    logged = []
    m = make_metrics(site, log=logged.append)

    class Boom(dict):
        def get(self, *a, **kw):
            raise RuntimeError("deliberate metrics bug")

    m.observe(Boom())                               # must not raise
    assert not m.enabled
    assert any("DISABLED" in str(s) for s in logged)
    m.observe({"probes": torch.full((1, 4, 2), 5.0)})   # still a no-op, still no raise
    m.observe_grads()
    assert m.flush(0) == {}


def test_exception_inside_flush_does_not_kill_the_step():
    site = make_site(r=2, threshold=2.0)
    logged = []
    m = make_metrics(site, log=logged.append)
    m.observe({"probes": torch.full((1, 4, 2), 5.0)})
    m.targets = None                                # break key construction
    assert m.flush(0) == {}
    assert not m.enabled
    assert any("DISABLED" in str(s) for s in logged)


def test_observe_ignores_sites_absent_from_the_acts_dict():
    site = make_site(r=2, threshold=2.0)
    m = make_metrics(site)
    m.observe({"other": torch.full((1, 4, 2), 5.0)})
    assert m.enabled
    assert m.flush(0) == {}


def test_enabled_metrics_banner_lists_what_is_available():
    site = make_site(r=2, threshold=2.0)
    bare = make_metrics(site)
    assert any("realized loudness" in s for s in bare.enabled_metrics())
    assert any("loss split OFF" in s for s in bare.enabled_metrics())
    assert not any("pooling depth" in s for s in bare.enabled_metrics())

    rich = make_metrics(make_site(r=2), loss_split=True,
                        sources={"probes": FakeSource(2, depth={"p50": 1}, quant=(
                            np.zeros(2), np.ones(2), np.zeros(2), np.ones(2)))})
    text = "; ".join(rich.enabled_metrics())
    assert "pooling depth" in text and "saturation" in text and "alignment" in text
    assert "loss split injected-vs-not" in text


def test_concept_names_come_from_the_source():
    site = make_site(r=2, threshold=2.0)
    m = InjectionMetrics({"probes": site}, sources={"probes": FakeSource(2, concepts=["monday", "friday"])},
                         targets={"probes": 0.03}, log_every=10)
    m.observe({"probes": torch.full((1, 4, 2), 5.0)})
    out = m.flush(0)
    assert "inj/probes/fire_rate/monday" in out and "inj/probes/aeff_mean/friday" in out


def test_multi_site_keys_do_not_collide():
    s1, s2 = make_site(r=2, threshold=2.0), make_site(r=3, threshold=2.0)
    s2.cfg.name = "acts"
    m = InjectionMetrics({"probes": s1, "acts": s2}, targets={"probes": 0.03, "acts": 0.05},
                         concepts={"probes": ["a", "b"], "acts": ["x", "y", "z"]}, log_every=10)
    a1 = torch.zeros(1, 10, 2); a1[0, :5, 0] = 5.0
    a2 = torch.zeros(1, 10, 3); a2[0, :2, 1] = 5.0
    m.observe({"probes": a1, "acts": a2})
    out = m.flush(0)
    assert out["inj/probes/fire_rate/a"] == pytest.approx(0.5)
    assert out["inj/acts/fire_rate/y"] == pytest.approx(0.2)
    assert out["inj/probes/loudness_target"] == pytest.approx(0.03)
    assert out["inj/acts/loudness_target"] == pytest.approx(0.05)
