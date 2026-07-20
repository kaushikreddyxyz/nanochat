"""CPU tests for nanochat.injection.sites wiring into GPT and the loudness
resolution path: config defaults, direction init (incl. file:), freeze/unfreeze +
the optimizability contract, GPT integration (dormant without acts, optimizer
groups, a real step), the two-form loudness grammar, loudness.json discovery and
its loud fallback, startup-before-training calibration, and DDP bit-identity.

The site math itself (dose response, exact no-ops, the D gauge, per-event
equalization, subset L_ref, the realized-loudness check) lives in
tests/test_injection_dose.py.
"""
import os
import sys
from dataclasses import asdict

import numpy as np
import torch

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(TESTS, ".."))
sys.path.insert(0, REPO)

from nanochat.injection.sources import make_orthonormal_P  # noqa: E402
from nanochat.injection.sites import (  # noqa: E402
    InjectionCfg,
    InjectionSite,
    assert_loudness_identical_across_ranks,
    build_sites,
    calibrate_dose_gate,
    classify_loudness_spec,
    discover_loudness_json,
    donor_source_layer_concepts,
    loudness_vector_hash,
    optimizer_param_split,
    orthonormal_direction,
    reassert_optimizability,
    sites_by_block,
    validate_donor_concepts,
)

B, T, N_EMBD, R = 2, 8, 64, 14


def _cfg(**kw):
    kw.setdefault("name", "coords")
    kw.setdefault("r", R)
    kw.setdefault("channel_scale", [0.05] * kw.get("r", R))
    return InjectionCfg(**kw)


def _site(**kw):
    return InjectionSite(_cfg(**kw), kw.pop("n_embd", N_EMBD))


# --------------------------------------------------------------------------- #
# Config defaults + direction init
# --------------------------------------------------------------------------- #
def test_config_defaults_are_the_standard():
    c = InjectionCfg(name="d", r=R)
    assert c.loudness == 1.0 and c.after_block == 0 and c.threshold == 2.0
    assert c.channel_scale is None and c.trainable_direction is False
    assert c.direction_init == "orthonormal" and c.optim == "adamw"


def test_minimal_config_is_name_and_r():
    c = InjectionCfg(**{"name": "acts", "r": 4})
    assert (c.name, c.r) == ("acts", 4)


def test_orthonormal_direction_matches_v1_P():
    """orthonormal_direction == make_orthonormal_P transposed, bitwise."""
    for seed in (0, 1337):
        d = orthonormal_direction(R, N_EMBD, seed)
        p = torch.from_numpy(make_orthonormal_P(N_EMBD, R, seed))
        assert torch.equal(d, p.t().contiguous())


def test_direction_init_variants():
    assert torch.equal(_site(direction_init="zeros").direction, torch.zeros(R, N_EMBD))
    a = _site(direction_init="randn", direction_seed=7).direction
    b = _site(direction_init="randn", direction_seed=7).direction
    assert torch.equal(a, b) and not torch.equal(a, _site(direction_init="randn",
                                                          direction_seed=8).direction)
    try:
        _site(direction_init="nonsense")
        raise AssertionError("unknown direction_init accepted")
    except ValueError as e:
        assert "direction_init" in str(e)


def test_direction_init_file():
    """file:<path> loads a (r, n_embd) direction from .npy/.npz verbatim (npz key "D"
    preferred). Against the REAL committed sphere manifold: bit-exact rows, frozen by
    default, trainable on request, loud shape mismatch, .npy parity."""
    import tempfile
    npz_path = os.path.join(REPO, "runs", "weekdays", "direction_sphere.npz")
    D = np.load(npz_path)["D"]
    assert D.dtype == np.float32 and D.shape == (7, 768)

    site = InjectionSite(_cfg(name="w", r=7, channel_scale=[0.0273] * 7,
                              direction_init=f"file:{npz_path}"), 768)
    assert site.direction.requires_grad is False, "file direction must default to frozen"
    assert np.array_equal(site.direction.detach().numpy(), D), "direction != file bit-exact"

    site_t = InjectionSite(_cfg(name="w", r=7, channel_scale=[0.0273] * 7,
                                trainable_direction=True,
                                direction_init=f"file:{npz_path}"), 768)
    assert site_t.direction.requires_grad is True
    assert np.array_equal(site_t.direction.detach().numpy(), D)

    raised = False
    try:
        InjectionSite(_cfg(name="w", r=6, channel_scale=[0.0] * 6,
                           direction_init=f"file:{npz_path}"), 768)
    except AssertionError as e:
        raised = "shape" in str(e)
    assert raised, "shape mismatch must raise an AssertionError naming the shape"

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f, D)
        npy_path = f.name
    site_npy = InjectionSite(_cfg(name="w", r=7, channel_scale=[0.0273] * 7,
                                  direction_init=f"file:{npy_path}"), 768)
    assert np.array_equal(site_npy.direction.detach().numpy(), D)
    os.unlink(npy_path)


# --------------------------------------------------------------------------- #
# Module plumbing + optimizability
# --------------------------------------------------------------------------- #
def test_build_sites_and_by_block():
    sites = build_sites([
        _cfg(name="a", r=3, after_block=1, channel_scale=[0.1] * 3),
        {"name": "b", "r": 4, "after_block": 1, "channel_scale": [0.1] * 4},
        _cfg(name="c", r=5, after_block=2, channel_scale=[0.1] * 5),
    ], N_EMBD)
    assert set(sites) == {"a", "b", "c"}
    by = sites_by_block(sites)
    assert sorted(by) == [1, 2] and len(by[1]) == 2 and len(by[2]) == 1
    try:
        build_sites([_cfg(name="dup", r=3, channel_scale=[0.1] * 3),
                     _cfg(name="dup", r=3, channel_scale=[0.1] * 3)], N_EMBD)
        raise AssertionError("duplicate site name accepted")
    except ValueError as e:
        assert "duplicate" in str(e)


def test_state_dict_keys():
    sites = build_sites([_cfg(name="a", r=3, channel_scale=[0.1] * 3)], N_EMBD)
    assert set(sites.state_dict()) == {"a.channel_scale", "a.threshold", "a.direction"}


def test_optimizer_param_split_returns_trainable_directions_only():
    sites = build_sites([
        _cfg(name="frozen", r=3, channel_scale=[0.1] * 3, trainable_direction=False),
        _cfg(name="ad", r=3, channel_scale=[0.1] * 3, trainable_direction=True),
        _cfg(name="mu", r=3, channel_scale=[0.1] * 3, trainable_direction=True, optim="muon"),
    ], N_EMBD)
    adamw, muon = optimizer_param_split(sites)
    assert [id(p) for p in adamw] == [id(sites["ad"].direction)]
    assert [id(p) for p in muon] == [id(sites["mu"].direction)]
    loud = {id(s.channel_scale) for s in sites.values()} | {id(s.threshold) for s in sites.values()}
    assert not loud & {id(p) for p in adamw + muon}, "loudness params must never be optimized"


def test_freeze_unfreeze_and_reassert():
    site = _site(trainable_direction=True)
    assert site.direction.requires_grad
    site.freeze()
    assert not site.direction.requires_grad
    site.unfreeze()
    assert site.direction.requires_grad
    sites = build_sites([
        _cfg(name="fr", r=3, channel_scale=[0.1] * 3, trainable_direction=False),
        _cfg(name="tr", r=3, channel_scale=[0.1] * 3, trainable_direction=True),
    ], N_EMBD)
    sites["fr"].direction.requires_grad_(True)      # clobber, as assign=True would
    sites["tr"].direction.requires_grad_(False)
    del sites["fr"].channel_scale._never_optimize
    reassert_optimizability(sites)
    assert not sites["fr"].direction.requires_grad
    assert sites["tr"].direction.requires_grad
    assert sites["fr"].channel_scale._never_optimize is True
    assert sites["fr"].threshold._never_optimize is True


def test_cfg_round_trips_through_json_meta():
    c = _cfg(name="v", r=4, channel_scale=[0.1, 0.2, 0.3, 0.4])
    import json
    rebuilt = InjectionCfg(**json.loads(json.dumps(asdict(c))))
    assert rebuilt == c


# --------------------------------------------------------------------------- #
# GPT integration (tiny model, CPU)
# --------------------------------------------------------------------------- #
def _tiny_gpt():
    from nanochat.gpt import GPT, GPTConfig
    cfg = GPTConfig(sequence_len=64, vocab_size=64, n_layer=4, n_head=2,
                    n_kv_head=2, n_embd=64, window_pattern="L")
    torch.manual_seed(0)
    with torch.device("meta"):
        m = GPT(cfg)
    m.to_empty(device="cpu")
    torch.manual_seed(0)
    m.init_weights()
    return m


def test_gpt_sites_dormant_without_acts():
    m = _tiny_gpt()
    x = torch.randint(0, 12, (2, 16))
    y = torch.randint(0, 12, (2, 16))
    loss_vanilla = m(x, y)
    m.setup_injection_sites([_cfg(name="coords", r=R, after_block=1)])
    assert torch.equal(m(x, y), loss_vanilla), "acts=None must be bit-identical to vanilla"
    mv = _tiny_gpt()
    assert m.estimate_flops() == mv.estimate_flops()
    sp = m.num_scaling_params()
    assert sp["injection"] == sum(p.numel() for p in m.injection_sites.parameters())
    assert sp["transformer_matrices"] == mv.num_scaling_params()["transformer_matrices"]


def test_gpt_optimizer_contract_and_step():
    m = _tiny_gpt()
    sites = m.setup_injection_sites([
        _cfg(name="coords", r=R, after_block=1, trainable_direction=False),
        _cfg(name="free", r=6, after_block=2, channel_scale=[0.1] * 6,
             trainable_direction=True, optim="muon"),
    ])
    opt = m.setup_optimizer()
    in_groups = {id(p) for g in opt.param_groups for p in g["params"]}
    for s in sites.values():
        assert id(s.channel_scale) not in in_groups and id(s.threshold) not in in_groups
    assert id(sites["coords"].direction) not in in_groups        # frozen: skipped
    assert id(sites["free"].direction) in in_groups              # trainable: muon group
    inj_groups = [g for g in opt.param_groups if g.get("injection")]
    assert len(inj_groups) == 1 and inj_groups[0]["kind"] == "muon" \
        and inj_groups[0]["weight_decay"] == 0.0

    x = torch.randint(0, 12, (2, 16))
    y = torch.randint(0, 12, (2, 16))
    # above the default 2.0 threshold, so the sites actually fire
    acts = {"coords": torch.randn(2, 16, R) + 6.0, "free": torch.randn(2, 16, 6) + 6.0}
    loss_on = m(x, y, acts=acts)
    assert loss_on.item() != m(x, y).item()
    loss_on.backward()
    assert sites["coords"].channel_scale.grad is not None         # want-signal
    assert sites["coords"].direction.grad is None
    assert sites["free"].direction.grad is not None
    c0 = sites["coords"].channel_scale.detach().clone()
    d0 = sites["coords"].direction.detach().clone()
    f0 = sites["free"].direction.detach().clone()
    opt.step()
    assert torch.equal(sites["coords"].channel_scale.detach(), c0), "loudness must never be stepped"
    assert torch.equal(sites["coords"].direction.detach(), d0), "frozen direction must never move"
    assert not torch.equal(sites["free"].direction.detach(), f0), "trainable direction must move"


# --------------------------------------------------------------------------- #
# Loudness grammar + loudness.json resolution
# --------------------------------------------------------------------------- #
class _FakeDonorSource:
    """Probe-score source: a gemma layer + concept columns + a score_loc, and
    deterministic sampled activation rows."""
    def __init__(self, r=3, layer=8, concepts=None, score_loc="/tmp/nope", n_docs=64):
        self.r, self.layer, self.name = r, layer, "fake-donor"
        self.concepts = concepts or [f"c{i}" for i in range(r)]
        self.score_loc = score_loc
        self._n = n_docs

    def sample_activation_rows(self, k, seed):
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(min(k, self._n)):
            m = int(rng.integers(20, 40))
            z = np.zeros((m, self.r), np.float32)
            for c in range(self.r):
                fire = rng.random(m) < (0.2 * (c + 1))
                z[fire, c] = 2.0 + (c + 1) * 1.0     # above the 2.0 relu knee
            rows.append(z)
        pooled = np.concatenate(rows)
        return pooled, len(rows), pooled.shape[0]


def _fake_loudness(concepts, layer=8):
    L, K = str(layer), len(concepts)
    return {
        "version": 1, "concepts": list(concepts),
        "ridge": {"active_loudness": {L: {"p50": [0.02 * (i + 1) for i in range(K)]}}},
        "subspace_total": {"ridge": {L: {"p50": 0.05, "p90": 0.09, "p95": 0.12, "p99": 0.2}}},
    }


def test_loudness_spec_grammar():
    assert classify_loudness_spec(1.0) == ("dial", 1.0)
    assert classify_loudness_spec("1.0") == ("dial", 1.0)
    assert classify_loudness_spec(0) == ("dial", 0.0)
    assert classify_loudness_spec("abs:0.03") == ("abs", 0.03)
    assert classify_loudness_spec("abs0.03") == ("abs", 0.03)
    for bad, msg in ((-1.0, ">= 0"), ("abs:", "needs a number"), ("junk", "could not convert"),
                     ([0.1, 0.2], "per-channel vector")):
        try:
            classify_loudness_spec(bad)
            raise AssertionError(f"accepted {bad!r}")
        except ValueError as e:
            assert msg in str(e), (bad, str(e))


def test_source_layer_requirement():
    class _NoLayer:                              # FnSource-like: nothing to key on
        name, r = "fn", 4
    try:
        donor_source_layer_concepts(_NoLayer())
        raise AssertionError("accepted a layer-less source")
    except ValueError as e:
        assert "probe-score source" in str(e) and "abs:" in str(e)


def test_abs_loudness_needs_no_loudness_json_or_gemma_identity():
    """The escape hatch: a source with no gemma layer/concepts still calibrates."""
    class _NoIdentity:
        name, r = "tabular", 3

        def sample_activation_rows(self, k, seed):
            rng = np.random.default_rng(seed)
            pooled = (np.abs(rng.normal(size=(800, 3))) + 2.5).astype(np.float32)
            return pooled, 32, pooled.shape[0]

    scale, meta = calibrate_dose_gate(_NoIdentity(), None, orthonormal_direction(3, N_EMBD),
                                      abs_target=0.04, threshold=2.0, k=32, seed=0,
                                      min_docs=8, log=lambda *_: None)
    assert meta["mode"] == "abs" and meta["target_median"] == 0.04
    assert meta["L_ref"] is None and meta["layer"] is None and meta["donor_loudness"] is None
    assert meta["injected_loudness"]["p50"] == 0.04 or abs(
        meta["injected_loudness"]["p50"] - 0.04) < 1e-9
    assert len(scale) == 3 and all(v > 0 for v in scale)


def test_loudness_discovery_logs_loudly():
    logs = []
    log = logs.append
    _, desc = discover_loudness_json(_FakeDonorSource(), "/some/override", log,
                                     load_fn=lambda loc: {"src": loc})
    assert desc == "override:/some/override" and any("--loudness-json" in x for x in logs)

    logs.clear()
    src = _FakeDonorSource(score_loc="the-store")
    _, desc = discover_loudness_json(src, None, log, load_fn=lambda loc: {"loc": loc})
    assert desc == "store:the-store" and any("source store root" in x for x in logs)

    logs.clear()

    def _load(loc):
        if loc == "the-store":
            raise FileNotFoundError("no loudness.json here")
        return {"fallback": loc}
    _, desc = discover_loudness_json(src, None, log, load_fn=_load,
                                     fallback_repo="kaushikreddyxyz/climbmix-scored")
    assert desc == "fallback:kaushikreddyxyz/climbmix-scored"
    assert any("FALLBACK" in x for x in logs) and any("!!!!" in x for x in logs)

    try:
        discover_loudness_json(src, None, log, load_fn=lambda loc: (_ for _ in ()).throw(
            FileNotFoundError(f"no loudness.json at {loc}")))
        raise AssertionError("missing artifact must be a hard error, not a silent fallback")
    except RuntimeError as e:
        assert "abs:" in str(e) and "--loudness-json" in str(e)


def test_live_source_calibrates_before_any_training_batch():
    calls = []

    class _FakeLive:
        name, r, layer = "live", 3, 8
        concepts = ["c0", "c1", "c2"]
        score_loc = "/tmp/none"

        def sample_activation_rows(self, k, seed):
            calls.append("calibrate")           # the live scorer runs here (startup)
            rng = np.random.default_rng(seed)
            pooled = (np.abs(rng.normal(size=(600, self.r))) + 2.5).astype(np.float32)
            return pooled, 32, pooled.shape[0]

        def draw_training_batch(self):
            calls.append("train")

    src = _FakeLive()
    calibrate_dose_gate(src, _fake_loudness(src.concepts), orthonormal_direction(3, N_EMBD),
                        dial=1.0, k=16, seed=0, min_docs=8, log=lambda *_: None)
    src.draw_training_batch()
    assert calls == ["calibrate", "train"], f"calibration must precede any training batch: {calls}"


def test_ddp_identity_of_the_calibrated_scale():
    loud = _fake_loudness(_FakeDonorSource(r=4).concepts)
    d = orthonormal_direction(4, N_EMBD)

    def rank():
        return calibrate_dose_gate(_FakeDonorSource(r=4), loud, d, dial=1.0, k=32, seed=3,
                                   min_docs=8, log=lambda *_: None)[0]

    ranks = [rank() for _ in range(4)]
    hashes = [loudness_vector_hash(v) for v in ranks]
    assert len(set(hashes)) == 1, "deterministic calibration must give identical hashes"
    assert_loudness_identical_across_ranks(ranks[0], "v", lambda *a: None,
                                           all_gather_hash=lambda h: hashes)
    try:
        assert_loudness_identical_across_ranks(ranks[0], "v", lambda *a: None,
                                               all_gather_hash=lambda h: [h, "deadbeef"])
        raise AssertionError("rank disagreement not caught")
    except RuntimeError as e:
        assert "disagree" in str(e)


def test_concepts_are_indexed_by_name_not_position():
    # Real sites are SUBSETS of the 54, so concepts match BY NAME; position is
    # irrelevant by construction (a stronger permutation guarantee than the old
    # exact-order check). An UNKNOWN name is still a hard refusal.
    assert list(validate_donor_concepts(["a", "b"], ["a", "b"])) == [0, 1]
    assert list(validate_donor_concepts(["a", "b"], ["b", "a"])) == [1, 0]
    assert list(validate_donor_concepts(["a", "b", "c", "d"], ["d", "b"])) == [3, 1]
    for loud_c, src_c, msg in ((["a", "b"], ["a", "zzz"], "zzz"), (["a", "a"], ["a"], "twice")):
        try:
            validate_donor_concepts(loud_c, src_c)
            raise AssertionError(f"accepted {src_c}")
        except ValueError as e:
            assert msg in str(e)


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        fn()
        print(f"{name}: OK")
    print("\nALL CHECKS PASSED")
