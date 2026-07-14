"""Cross-suite consolidation tests for the injection-on-vs-off eval suite.

One tiny end-to-end stub shared across the three suites (harness / run_evals /
causal) pinning the science-critical invariants that a per-suite test cannot:

  1. OFF-mechanism equivalence — the four "injection off" paths (acts=None,
     forward_metrics gate_scale=0, the CORE adapter's direct model call under
     _scaled_gates(0), and the legacy acts*0 zeroing) are BIT-identical.
  2. Dose semantics — the gate sits OUTSIDE the z/rms(z) renorm, so gate_scale
     is LINEAR in injected loudness (exact at power-of-2 scales), while scaling
     the ACTS is a loudness no-op (renorm) except exact-0 (exact no-op).
  3. Position indexing — per_token_ce[t] = CE of predicting token t (pos 0 NaN)
     is the ONE convention: harness.forward_metrics, run_evals.option_mean_ce /
     bucket_masks, harness.ce_report and causal._logits_last all agree on the
     same forward of the same stub model.
  4. Store-vs-calendar — one canonical channel order (weekday_source, via the
     harness re-export); causal_items.STORE_ORDER and weekday_evalset day lists
     must match it, never re-derive it.
  5. attach_site provenance — the direction bolted onto the baseline control is
     copied VERBATIM from the array the caller passes (causal.py passes the arm
     CHECKPOINT's loaded direction), and round-trips via _extract_direction.
  6. load_model's file:->zeros rewrite — a synthetic checkpoint whose meta says
     direction_init "file:<nonexistent>" loads fine and yields the CHECKPOINT
     direction (while the raw meta config provably cannot even build).
  7. empirical_patterns.json — committed artifact sanity (store order, dominant
     own-channel >= present_z, off-diagonals subdominant).

CPU, no network, no rustbpe, no gemma. Run:
    python runs/weekdays/eval/test_consolidation.py     (or via pytest)
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import harness  # noqa: E402
import causal  # noqa: E402
import causal_items  # noqa: E402
import run_evals  # noqa: E402
import weekday_evalset  # noqa: E402
from test_harness import _tiny_gpt_with_site  # noqa: E402  (shared tiny model)


# --------------------------------------------------------------------------- #
# 1. The four OFF mechanisms are bit-identical (and ON differs).
# --------------------------------------------------------------------------- #
def test_off_mechanisms_bit_identical():
    m = _tiny_gpt_with_site()
    torch.manual_seed(2)
    ids = torch.randint(0, 60, (1, 12))
    acts = np.random.RandomState(1).randn(1, 12, 7).astype(np.float32)  # dense nonzero
    acts_t = torch.from_numpy(acts)

    vanilla = harness.forward_metrics(m, ids, acts=None)["logits"]
    gate0 = harness.forward_metrics(m, ids, acts, gate_scale=0.0)["logits"]
    with torch.inference_mode():
        # the CORE adapter's off path: direct model call under _scaled_gates(0)
        with harness._scaled_gates(m, 0.0):
            adapter_off = m(ids, acts={"weekdays": acts_t}).float().cpu()
        # the legacy acts*0 zeroing (pre-unification mechanism) — also exact
        acts_zero = m(ids, acts={"weekdays": acts_t * 0.0}).float().cpu()

    assert torch.equal(gate0, vanilla), "forward_metrics gate_scale=0 != vanilla"
    assert torch.equal(adapter_off, vanilla), "_scaled_gates(0) direct call != vanilla"
    assert torch.equal(acts_zero, vanilla), "acts*0 zeroing != vanilla"
    # gate restored after the context manager
    assert torch.equal(m.injection_sites["weekdays"].gate.detach(), torch.tensor(0.05))
    on = harness.forward_metrics(m, ids, acts, gate_scale=1.0)["logits"]
    assert not torch.equal(on, vanilla), "gate_scale=1 with dense acts must differ"


# --------------------------------------------------------------------------- #
# 2. Dose semantics: gate linear in loudness; acts scaling a renorm no-op.
# --------------------------------------------------------------------------- #
def test_gate_scale_is_linear_loudness():
    """Site math is x + gate * rms(x) * z/rms(z): the gate multiplies OUTSIDE the
    renorm. On x=0 the output IS the injected delta, so a 4x gate must give a
    bit-exactly 4x delta (power-of-2 scaling commutes with fp rounding)."""
    from nanochat.injection.sites import InjectionCfg, InjectionSite
    g = 0.0273
    s1 = InjectionSite(InjectionCfg(name="w", r=7, after_block=0, gate=g), 64)
    s4 = InjectionSite(InjectionCfg(name="w", r=7, after_block=0, gate=4 * g), 64)
    with torch.no_grad():
        s4.direction.copy_(s1.direction)
    x = torch.zeros(1, 8, 64)
    a = torch.randn(1, 8, 7, generator=torch.Generator().manual_seed(3))
    d1 = s1(x, a)   # == the injected delta exactly (x = 0)
    d4 = s4(x, a)
    assert not torch.equal(d1, x), "site with nonzero acts must inject"
    assert torch.equal(d4, 4.0 * d1), "gate is not linear in injected loudness"


def test_acts_scaling_is_renorm_noop_except_zero():
    from nanochat.injection.sites import InjectionCfg, InjectionSite
    s = InjectionSite(InjectionCfg(name="w", r=7, after_block=0, gate=0.0273), 64)
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(1, 8, 64, generator=gen)
    a = torch.randn(1, 8, 7, generator=gen)
    y = s(x, a)
    y4 = s(x, 4.0 * a)      # z -> 4z, z/rms(z) unchanged (bitwise at powers of 2)
    y0 = s(x, 0.0 * a)      # z = 0 -> rms clamp -> z_hat = 0 -> exact no-op
    assert torch.equal(y4, y), "acts scaling must be a loudness no-op (renorm)"
    assert torch.equal(y0, x), "zero acts must be an exact no-op"
    assert not torch.equal(y, x)


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

    fm = harness.forward_metrics(m, ids, acts, gate_scale=1.0)
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
    fm2 = harness.forward_metrics(m, ids, acts, gate_scale=1.0, return_logits=False)
    assert fm2["logits"] is None
    assert np.array_equal(fm2["per_token_ce"][:, 1:], ptc[:, 1:])


# --------------------------------------------------------------------------- #
# 4. Store-vs-calendar: one canonical channel order, everywhere.
# --------------------------------------------------------------------------- #
def test_store_calendar_single_source():
    # harness re-exports weekday_source.WEEKDAY_CONCEPTS (absolute-path import);
    # None would mean the fallback literal was silently in play — refuse.
    assert harness.WeekdayProbeScoreSource is not None, \
        "harness fell back to the literal WEEKDAY_CONCEPTS (weekday_source import failed)"
    assert list(harness.WEEKDAY_CONCEPTS) == causal_items.STORE_ORDER
    assert causal_items.STORE_ORDER == sorted(causal_items.STORE_ORDER)  # name-sorted store cols
    assert sorted(causal_items.CALENDAR_ORDER) == sorted(causal_items.STORE_ORDER)
    # weekday_evalset's display list is CALENDAR order, capitalized
    assert weekday_evalset.WEEKDAYS == [d.capitalize() for d in causal_items.CALENDAR_ORDER]


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


def test_attach_site_direction_verbatim_roundtrip():
    m = _tiny_gpt_768()
    D = np.random.RandomState(7).randn(7, 768).astype(np.float32)  # arbitrary "checkpoint" rows
    site = harness.attach_site(m, D)   # defaults: gate=GATE, after_block=3
    assert np.array_equal(site.direction.detach().numpy(), D), \
        "attach_site must copy the given direction verbatim"
    assert float(site.gate.detach()) == np.float32(harness.GATE)
    assert site.gate._never_optimize and not site.direction.requires_grad
    # causal._extract_direction (what warms the control cache) round-trips it
    assert np.array_equal(causal._extract_direction(m), D)
    # attaching over an existing site must refuse (baseline-only control)
    try:
        harness.attach_site(m, D)
        raise SystemExit("attach_site must assert on an existing 'weekdays' site")
    except AssertionError:
        pass


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
    site_cfg_saved = dict(name="weekdays", r=7, after_block=1, gate=0.0273,
                          trainable_direction=False,
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
                                              gate=0.0273, direction_init="orthonormal")])
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
        model, meta_out = harness.load_model("sphere", "cpu")
    finally:
        harness._download = orig

    site = model.injection_sites["weekdays"]
    assert torch.equal(site.direction.detach(), known), \
        "loaded direction must be the CHECKPOINT's, not zeros/file"
    assert float(site.gate.detach()) == np.float32(0.0273)
    assert site.gate._never_optimize and not site.direction.requires_grad
    assert model._injection_by_block, "site must be wired into the forward loop"
    # the meta dict handed back still records the original file: provenance
    assert meta_out["injection_sites_config"][0]["direction_init"].startswith("file:")


# --------------------------------------------------------------------------- #
# 7. Committed empirical_patterns.json sanity.
# --------------------------------------------------------------------------- #
def test_empirical_patterns_artifact():
    path = os.path.join(HERE, "empirical_patterns.json")
    assert os.path.exists(path), "run empirical_patterns.py (artifact must be committed)"
    with open(path) as f:
        ej = json.load(f)
    assert ej["store_order"] == causal_items.STORE_ORDER
    assert ej["layer"] == 8 and ej["present_z"] == 2.0
    for day in causal_items.STORE_ORDER:
        v = np.asarray(ej["vectors"][day], np.float64)
        assert v.shape == (7,)
        own = causal_items.store_idx(day)
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
    vendored = os.path.join(HERE, "attr_out")
    assert os.path.exists(os.path.join(vendored, "probe_set_arrays.npz")), \
        "vendored attr_out must ship with the suite (superproject copy is gitignored)"
    ps = json.load(open(os.path.join(out, "probe_set.json")))
    assert 8 in ps["layers"], f"gemma layer 8 missing from probe layers {ps['layers']}"
    mb = list(ps["main_block_concepts"])
    assert [mb.index(c) for c in harness.WEEKDAY_CONCEPTS] == list(range(47, 54)), \
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
