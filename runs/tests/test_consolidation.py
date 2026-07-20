"""Cross-suite invariant tests binding the SHARED runs/lib/eval drivers to the weekday
family (things no per-suite test can pin): off-mechanism bit-equivalence, dose linearity
in channel_scale, the single per_token_ce position convention, one canonical store-order
source, attach_site provenance, the load_model file:->zeros rewrite,
empirical_patterns.json artifact sanity.
Run: python -m pytest runs/tests/test_consolidation.py
"""
import json
import os

import numpy as np
import torch

import causal
import harness
import run_evals
import weekday_evalset
import weekday_items
from conftest import LIB_EVAL, WEEKDAYS
from test_harness import SITE_SCALE, _tiny_gpt_with_site  # shared tiny model

FAMILY = "weekdays"
WEEKDAY_EVAL = os.path.join(WEEKDAYS, "eval")
causal.bind_items_module(os.path.join(WEEKDAY_EVAL, "weekday_items.py"))


# --------------------------------------------------------------------------- #
# 1. The four OFF mechanisms are bit-identical (and ON differs).
# --------------------------------------------------------------------------- #
def test_off_mechanisms_bit_identical():
    m = _tiny_gpt_with_site()
    torch.manual_seed(2)
    ids = torch.randint(0, 60, (1, 12))
    # supra-threshold everywhere, so ON genuinely injects on every row
    acts = np.random.RandomState(1).randn(1, 12, 7).astype(np.float32) + 6.0
    acts_t = torch.from_numpy(acts)

    vanilla = harness.forward_metrics(m, ids, acts=None)["logits"]
    off = harness.forward_metrics(m, ids, acts, loudness_scale=0.0)["logits"]
    with torch.inference_mode():
        # the CORE adapter's off path: direct model call under scaled_loudness(0)
        with harness.scaled_loudness(m, 0.0):
            adapter_off = m(ids, acts={"acts": acts_t}).float().cpu()
        # sub-threshold acts relu to bitwise zero — the source-side no-op
        acts_zero = m(ids, acts={"acts": acts_t * 0.0}).float().cpu()

    assert torch.equal(off, vanilla), "forward_metrics loudness_scale=0 != vanilla"
    assert torch.equal(adapter_off, vanilla), "scaled_loudness(0) direct call != vanilla"
    assert torch.equal(acts_zero, vanilla), "sub-threshold acts != vanilla"
    # channel_scale restored after the context manager
    assert torch.equal(m.injection_sites["acts"].channel_scale.detach(),
                       torch.full((7,), SITE_SCALE))
    on = harness.forward_metrics(m, ids, acts, loudness_scale=1.0)["logits"]
    assert not torch.equal(on, vanilla), "loudness_scale=1 with firing acts must differ"


# --------------------------------------------------------------------------- #
# 2. Dose semantics: linear in channel_scale AND in the activation magnitude.
# --------------------------------------------------------------------------- #
def _site(scale, threshold=0.0, r=7, n_embd=64):
    from nanochat.injection.sites import InjectionCfg, InjectionSite
    return InjectionSite(InjectionCfg(name="w", r=r, after_block=0, threshold=threshold,
                                      channel_scale=[scale] * r), n_embd)


def test_loudness_is_linear_in_channel_scale():
    """Site math is x + rms(x) * ((relu(a - z0) * channel_scale) @ D_hat): loudness
    multiplies OUTSIDE the direction. On x=0 the output IS the injected delta, so a 4x
    channel_scale must give a bit-exactly 4x delta (powers of 2 commute with fp
    rounding)."""
    s1, s4 = _site(0.03), _site(4 * 0.03)
    with torch.no_grad():
        s4.direction.copy_(s1.direction)
    x = torch.zeros(1, 8, 64)
    a = torch.randn(1, 8, 7, generator=torch.Generator().manual_seed(3)).abs() + 1.0
    d1, d4 = s1(x, a), s4(x, a)
    assert not torch.equal(d1, x), "site with firing acts must inject"
    assert torch.equal(d4, 4.0 * d1), "loudness is not linear in channel_scale"


def test_acts_magnitude_survives_and_zero_is_exact_noop():
    """The dose standard: magnitude is NOT renormalized away. At threshold 0 the site is
    exactly linear in the activation, so 4x acts inject 4x; zero/negative acts relu to
    bitwise 0 and the site is a strict no-op."""
    s = _site(0.03)
    gen = torch.Generator().manual_seed(4)
    x = torch.zeros(1, 8, 64)
    a = torch.randn(1, 8, 7, generator=gen).abs() + 1.0
    y, y4, y0 = s(x, a), s(x, 4.0 * a), s(x, 0.0 * a)
    assert torch.equal(y4, 4.0 * y), "4x acts must inject 4x (magnitude must survive)"
    assert torch.equal(y0, x), "zero acts must be an exact no-op"
    assert not torch.equal(y, x)
    assert torch.equal(s(x, -a), x), "negative acts must relu to a bitwise no-op"


# --------------------------------------------------------------------------- #
# 3. ONE position-indexing convention across all three suites (shared stub).
# --------------------------------------------------------------------------- #
def test_position_indexing_shared_stub():
    m = _tiny_gpt_with_site()
    torch.manual_seed(5)
    T = 10
    ids = torch.randint(0, 60, (1, T))
    acts = np.zeros((1, T, 7), np.float32)
    acts[0, 3, 0] = 2.5   # "injected" input tokens 3 and 6
    acts[0, 6, 2] = 3.0

    fm = harness.forward_metrics(m, ids, acts, loudness_scale=1.0)
    logits = fm["logits"]                     # [1, T, V] fp32
    ptc = fm["per_token_ce"]                  # [1, T], pos 0 NaN

    # (a) harness convention: per_token_ce[t] = -log softmax(logits[t-1])[ids[t]]
    assert np.isnan(ptc[0, 0])
    lsm = torch.log_softmax(logits[0], dim=-1).numpy()
    for t in range(1, T):
        manual = -lsm[t - 1, int(ids[0, t])]
        assert abs(ptc[0, t] - manual) < 1e-5, (t, ptc[0, t], manual)

    # (b) causal readout: answer-position logits == the LAST row (predicts token T)
    last = causal._logits_last(fm)
    assert np.array_equal(last, logits[0, -1].numpy().astype(np.float64))

    # (c) run_evals option CE: mean over PREDICTED positions [start, end)
    ce1d = run_evals._ce1d(fm)
    start, end = 4, T
    assert abs(run_evals.option_mean_ce(ce1d, start, end)
               - float(np.mean(ptc[0, start:end].astype(np.float64)))) < 1e-9

    # (d) run_evals bucket_masks == harness.ce_report on the same acts
    valid = [t >= 1 for t in range(T)]
    masks = run_evals.bucket_masks(acts[0], valid)
    rep = harness.ce_report(ptc, acts)
    assert rep["n_injected"] == sum(masks["injected"]) == 2
    assert rep["n_after_injected"] == sum(masks["after"]) == 2
    assert rep["n_other"] == sum(masks["rest"])
    assert rep["n_overall"] == sum(valid)

    # (e) return_logits=False (val-bpb CE-only path): identical CE, no logits
    fm2 = harness.forward_metrics(m, ids, acts, loudness_scale=1.0, return_logits=False)
    assert fm2["logits"] is None
    assert np.array_equal(fm2["per_token_ce"][:, 1:], ptc[:, 1:])


# --------------------------------------------------------------------------- #
# 4. Store-vs-calendar: one canonical channel order, everywhere.
# --------------------------------------------------------------------------- #
def test_store_calendar_single_source():
    # harness loads runs/lib/probe_source.py by absolute path; None would mean the
    # concept registry never imported and every channel order is unverified — refuse.
    assert harness.ConceptProbeScoreSource is not None, \
        "harness failed to import runs/lib/probe_source.py"
    assert harness.family_concepts(FAMILY) == weekday_items.STORE_ORDER
    assert weekday_items.STORE_ORDER == sorted(weekday_items.STORE_ORDER)  # name-sorted cols
    assert sorted(weekday_items.CALENDAR_ORDER) == sorted(weekday_items.STORE_ORDER)
    # the shared driver's bindings are the weekday bank's, not a stale default
    assert causal.STORE_ORDER == weekday_items.STORE_ORDER
    assert causal.CALENDAR_ORDER == weekday_items.CALENDAR_ORDER
    # weekday_evalset's display list is CALENDAR order, capitalized
    assert weekday_evalset.WEEKDAYS == [d.capitalize() for d in weekday_items.CALENDAR_ORDER]


# --------------------------------------------------------------------------- #
# 5. attach_site: the bolted direction is the caller's array, verbatim.
# --------------------------------------------------------------------------- #
def _tiny_gpt_768(n_layer=4):
    from nanochat.gpt import GPT, GPTConfig
    cfg = GPTConfig(sequence_len=32, vocab_size=64, n_layer=n_layer, n_head=6,
                    n_kv_head=6, n_embd=768, window_pattern="L")
    torch.manual_seed(0)
    with torch.device("meta"):
        m = GPT(cfg)
    m.to_empty(device="cpu")
    torch.manual_seed(0)
    m.init_weights()
    m.eval()
    return m


def test_attach_site_carries_direction_and_calibrated_loudness():
    m = _tiny_gpt_768()
    rs = np.random.RandomState(7)
    D = rs.randn(7, 768).astype(np.float32)          # arbitrary "checkpoint" rows
    cs = rs.uniform(0.01, 0.05, 7).astype(np.float32)  # a donor arm's calibrated scale
    site = harness.attach_site(m, D, cs, after_block=3, name="weekdays")
    assert np.array_equal(site.direction.detach().numpy(), D), \
        "attach_site must copy the given direction verbatim"
    assert np.array_equal(site.channel_scale.detach().numpy(), cs), \
        "attach_site must carry the donor arm's calibrated channel_scale (dial 1.0)"
    assert site.channel_scale._never_optimize and site.threshold._never_optimize
    assert not site.direction.requires_grad
    # causal._site_params (what warms the control cache) round-trips it
    got = causal._site_params(m, "weekdays")
    assert np.array_equal(got["direction"], D) and np.array_equal(got["channel_scale"], cs)
    assert got["after_block"] == 3
    # attaching over an existing site must refuse (baseline-only control)
    try:
        harness.attach_site(m, D, cs, name="weekdays")
        raise SystemExit("attach_site must assert on an existing 'weekdays' site")
    except AssertionError:
        pass


def test_attach_site_dial_scales_donor_loudness():
    m = _tiny_gpt_768()
    D = np.random.RandomState(8).randn(7, 768).astype(np.float32)
    cs = np.full(7, 0.02, np.float32)
    site = harness.attach_site(m, D, cs, name="weekdays", dial=4.0)
    assert np.allclose(site.channel_scale.detach().numpy(), 4.0 * cs)


def test_resolve_arms_order_and_controls():
    cfgs = causal.resolve_arms("all")
    labels = [c[0] for c in cfgs]
    assert labels == ["trainable", "sphere", "orthogonal",
                      "baseline_trainable", "baseline_sphere", "baseline_orthogonal"], \
        "real arms must run FIRST so the control's direction cache is warm"
    for label, base, control in cfgs:
        assert control == label.startswith("baseline_")
        assert base in ("trainable", "sphere", "orthogonal")
    assert causal.resolve_arms("sphere,baseline_sphere") == \
        [("sphere", "sphere", False), ("baseline_sphere", "sphere", True)]


# --------------------------------------------------------------------------- #
# 6. load_model's file:->zeros rewrite (synthetic checkpoint, sphere-arm shape).
# --------------------------------------------------------------------------- #
def test_load_model_file_direction_rewrite(tmp_path=None):
    import tempfile
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.injection.sites import InjectionCfg

    model_cfg = dict(sequence_len=64, vocab_size=64, n_layer=4, n_head=2,
                     n_kv_head=2, n_embd=64, window_pattern="L")
    # a real injected meta: the trainer froze the CALIBRATED channel_scale into it
    site_cfg_saved = dict(name="weekdays", r=7, after_block=1, threshold=2.0,
                          channel_scale=[0.02] * 7, trainable_direction=False,
                          direction_init="file:runs/weekdays/DOES_NOT_EXIST.npz",
                          direction_seed=1337, optim="adamw")

    # (a) the RAW meta config cannot even build (file: init resolves + fails)
    with torch.device("meta"):
        m_src = GPT(GPTConfig(**model_cfg))
    m_src.to_empty(device="cpu")
    torch.manual_seed(0)
    m_src.init_weights()
    try:
        m_src.setup_injection_sites([dict(site_cfg_saved)])
        raise SystemExit("file:<nonexistent> direction_init must fail without the npz")
    except FileNotFoundError:
        pass

    # (b) build the "checkpoint": same config with a KNOWN nonzero direction
    m_src.setup_injection_sites([InjectionCfg(name="weekdays", r=7, after_block=1,
                                              channel_scale=[0.02] * 7,
                                              direction_init="orthonormal")])
    known = torch.arange(7 * 64, dtype=torch.float32).reshape(7, 64) / 100.0
    with torch.no_grad():
        m_src.injection_sites["weekdays"].direction.copy_(known)

    tmp = tempfile.mkdtemp()
    model_path = os.path.join(tmp, "model_002520.pt")
    torch.save(m_src.state_dict(), model_path)
    meta = {"step": 2520, "model_config": model_cfg,
            "injection_sites_config": [dict(site_cfg_saved)]}

    # (c) load through harness.load_model with _download stubbed to the synthetic
    orig = harness._download
    harness._download = lambda repo, arm, step: (model_path, json.loads(json.dumps(meta)))
    try:
        model, meta_out = harness.load_model("sphere", "cpu", "fake/repo", 2520)
    finally:
        harness._download = orig

    site = model.injection_sites["weekdays"]
    assert torch.equal(site.direction.detach(), known), \
        "loaded direction must be the CHECKPOINT's, not zeros/file"
    assert torch.equal(site.channel_scale.detach(), torch.full((7,), 0.02))
    assert site.channel_scale._never_optimize and not site.direction.requires_grad
    assert model._injection_by_block, "site must be wired into the forward loop"
    # the meta dict handed back still records the original file: provenance
    assert meta_out["injection_sites_config"][0]["direction_init"].startswith("file:")


# --------------------------------------------------------------------------- #
# 7. Committed empirical_patterns.json sanity.
# --------------------------------------------------------------------------- #
def test_empirical_patterns_artifact():
    path = os.path.join(WEEKDAY_EVAL, "empirical_patterns.json")
    assert os.path.exists(path), "run empirical_patterns.py (artifact must be committed)"
    with open(path) as f:
        ej = json.load(f)
    assert ej["store_order"] == weekday_items.STORE_ORDER
    assert ej["layer"] == 8 and ej["present_z"] == 2.0
    for day in weekday_items.STORE_ORDER:
        v = np.asarray(ej["vectors"][day], np.float64)
        assert v.shape == (7,)
        own = weekday_items.store_idx(day)
        assert v[own] >= ej["present_z"], f"{day}: own channel {v[own]} < present_z"
        assert v[own] == v.max(), f"{day}: own channel not dominant"
        others = np.delete(v, own)
        assert np.all(others < v[own]) and np.all(np.abs(others) < 2.0), \
            f"{day}: implausible cross-channel structure {v}"
        assert ej["n_active"][day] > 1000, f"{day}: too few active tokens for a stable median"


# --------------------------------------------------------------------------- #
# 8. Probe constants are locatable + shaped for GemmaScorer (no gemma/network).
# --------------------------------------------------------------------------- #
def test_probe_constants_locatable_and_shaped():
    out = harness._find_attribution_out()
    vendored = os.path.join(LIB_EVAL, "attr_out")
    assert os.path.exists(os.path.join(vendored, "probe_set_arrays.npz")), \
        "vendored attr_out must ship with the suite (superproject copy is gitignored)"
    ps = json.load(open(os.path.join(out, "probe_set.json")))
    assert 8 in ps["layers"], f"gemma layer 8 missing from probe layers {ps['layers']}"
    mb = list(ps["main_block_concepts"])
    assert [mb.index(c) for c in harness.family_concepts(FAMILY)] == list(range(47, 54)), \
        "weekday concepts not at main_block cols 47..53"
    arr = np.load(os.path.join(out, "probe_set_arrays.npz"))
    L, C = len(ps["layers"]), len(mb)
    assert arr["W"].shape[:2] == (L, C) and arr["b"].shape == (L, C)
    assert arr["nat_mean"].shape == (L, arr["W"].shape[2]) == arr["nat_std"].shape
    assert np.all(np.isfinite(arr["W"])) and np.all(arr["nat_std"] > 0)


# --------------------------------------------------------------------------- #
def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\nAll {len(tests)} consolidation tests passed.")


if __name__ == "__main__":
    main()
