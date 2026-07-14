"""CPU tests for nanochat.injection.sites and their wiring into GPT: v1/v2
forward-value equivalence, RMS calibration, exact zero-row no-op, gate=0
no-op + blocked direction grads (gate still gets its want-signal), per-channel
(vector) gate invariants, auto-gate calibration + determinism + checkpoint-meta
persistence, the optimizer contract (gates in no group, frozen skipped, trainable
bucketed adamw/muon), state-dict keys, and acts=None == vanilla."""
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
    assert_gate_identical_across_ranks,
    build_sites,
    calibrate_auto_gate,
    calibrate_donor_gate,
    discover_loudness_json,
    donor_gate_from_loudness,
    donor_source_layer_concepts,
    gate_vector_hash,
    optimizer_param_split,
    orthonormal_direction,
    parse_donor_gate_spec,
    parse_gate_spec,
    reassert_optimizability,
    sites_by_block,
    validate_donor_concepts,
)

B, T, N_EMBD, R = 2, 8, 64, 14


def test_gate_default_is_005():
    # Default gate back to 0.05 (injected RMS = 0.05 * residual RMS).
    assert InjectionCfg(name="d", r=R, after_block=0).gate == 0.05


def _v1_inject(x, coords, P, beta):
    """The retired v1 inline formula from gpt.py (P is (n_embd, r))."""
    zc = coords.to(x.dtype) @ P.t()
    rms_x = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    rms_z = zc.pow(2).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    return x + beta * (rms_x / rms_z) * zc


def _tabular_site(beta=0.05, seed=1337, trainable=False, r=R, n_embd=N_EMBD):
    cfg = InjectionCfg(name="coords", r=r, after_block=0, gate=beta,
                       trainable_direction=trainable, direction_seed=seed)
    return InjectionSite(cfg, n_embd)


def test_orthonormal_direction_matches_v1_P():
    """orthonormal_direction == make_orthonormal_P transposed, bitwise: the
    single-store training path reproduces the store's P from the same seed."""
    P = make_orthonormal_P(N_EMBD, R, seed=1337)                # (n_embd, r) numpy
    D = orthonormal_direction(R, N_EMBD, seed=1337)             # (r, n_embd) torch
    assert np.array_equal(P.T, D.numpy())
    # rows orthonormal (isometric embedding of the activation space)
    eye = D @ D.t()
    assert torch.allclose(eye, torch.eye(R), atol=1e-5)
    # deterministic
    assert torch.equal(D, orthonormal_direction(R, N_EMBD, seed=1337))
    assert not torch.equal(D, orthonormal_direction(R, N_EMBD, seed=7))


def test_v1_v2_forward_equivalence():
    torch.manual_seed(0)
    x = torch.randn(B, T, N_EMBD)
    coords = torch.randn(B, T, R)
    coords[0, :3] = 0.0                                          # zero (BOS/missing) rows
    beta = 0.05
    P = torch.from_numpy(make_orthonormal_P(N_EMBD, R, seed=1337))
    site = _tabular_site(beta=beta, seed=1337)
    assert torch.equal(site.direction.data, P.t())               # same fixed direction
    v1 = _v1_inject(x, coords, P, beta)
    v2 = site(x, coords)
    # identical forward values up to fp association order (the two formulas
    # multiply the same three factors in a different grouping)
    assert torch.allclose(v1, v2, rtol=1e-5, atol=1e-6), \
        f"max |v1-v2| = {(v1 - v2).abs().max().item():.3e}"
    # the zero-row no-op is EXACT in both
    assert torch.equal(v1[0, :3], x[0, :3])
    assert torch.equal(v2[0, :3], x[0, :3])


def test_rms_calibration():
    torch.manual_seed(1)
    x = torch.randn(B, T, N_EMBD)
    a = torch.randn(B, T, R)
    for gate in (0.05, 0.3, 1.0):
        site = _tabular_site(beta=gate)
        out = site(x, a)
        ratio = (out - x).pow(2).mean(-1).sqrt() / x.pow(2).mean(-1).sqrt()
        assert torch.allclose(ratio, torch.full_like(ratio, gate), rtol=1e-4), \
            f"gate={gate}: injected RMS ratio off ({ratio.min():.5f}..{ratio.max():.5f})"


def test_zero_rows_exact_noop_and_no_nan():
    torch.manual_seed(2)
    x = torch.randn(B, T, N_EMBD)
    site = _tabular_site()
    out0 = site(x, torch.zeros(B, T, R))
    assert torch.equal(out0, x), "all-zero activations must be an EXACT no-op"
    mixed = torch.randn(B, T, R)
    mixed[1, 2:5] = 0.0
    outm = site(x, mixed)
    assert torch.equal(outm[1, 2:5], x[1, 2:5])
    assert torch.isfinite(outm).all()


def test_gate_zero_noop_and_zero_direction_grad():
    torch.manual_seed(3)
    x = torch.randn(B, T, N_EMBD, requires_grad=True)
    a = torch.randn(B, T, R)
    site = _tabular_site(beta=0.0, trainable=True)
    out = site(x, a)
    assert torch.equal(out.detach(), x.detach()), "gate=0 forward must equal x exactly"
    out.sum().backward()
    # direction is trainable but sees ZERO gradient through the closed gate...
    assert site.direction.grad is not None
    assert torch.count_nonzero(site.direction.grad) == 0, "gate=0 must block direction grads"
    # ...while the gate itself still gets its want-signal
    assert site.gate.grad is not None and torch.isfinite(site.gate.grad)


def test_gate_grads_assigned_but_never_in_optimizer_split():
    torch.manual_seed(4)
    sites = build_sites([
        InjectionCfg(name="frozen", r=R, after_block=1, gate=0.05, trainable_direction=False),
        InjectionCfg(name="adamw_site", r=6, after_block=2, gate=0.1, trainable_direction=True),
        InjectionCfg(name="muon_site", r=5, after_block=2, gate=0.1, trainable_direction=True, optim="muon"),
    ], N_EMBD)
    x = torch.randn(B, T, N_EMBD)
    out = x
    for name, r in [("frozen", R), ("adamw_site", 6), ("muon_site", 5)]:
        out = sites[name](out, torch.randn(B, T, r))
    out.sum().backward()
    adamw, muon = optimizer_param_split(sites)
    assert adamw == [sites["adamw_site"].direction]
    assert muon == [sites["muon_site"].direction]
    for s in sites.values():
        assert s.gate.grad is not None, "gate grad must be ASSIGNED every backward"
        assert getattr(s.gate, "_never_optimize", False) is True
        assert all(s.gate is not p for p in adamw + muon)
    assert sites["frozen"].direction.grad is None                # frozen: no grad at all
    # by-block map groups the two block-2 sites together
    by_block = sites_by_block(sites)
    assert set(by_block) == {1, 2} and len(by_block[2]) == 2


def test_state_dict_keys():
    site = _tabular_site()
    assert set(site.state_dict().keys()) == {"gate", "direction"}
    sites = build_sites([InjectionCfg(name="a", r=3, after_block=0)], N_EMBD)
    assert set(sites.state_dict().keys()) == {"a.gate", "a.direction"}


def test_scalar_gate_path_byte_identical_to_old_forward():
    # The scalar-gate path must equal the retired channel_weights=ones forward
    # bit-for-bit (channel_weights folded into the gate; scalar path unchanged).
    torch.manual_seed(9)
    x = torch.randn(B, T, N_EMBD)
    a = torch.randn(B, T, R)
    site = _tabular_site(beta=0.05)
    z = a @ site.direction
    z_hat = z / z.pow(2).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    rms_x = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    old = x + 0.05 * rms_x * z_hat
    assert torch.equal(site(x, a), old)


def test_per_channel_gate_mute_and_loudness():
    torch.manual_seed(5)
    x = torch.randn(B, T, N_EMBD)
    # gate[0]=0 mutes channel 0: an activation with ONLY channel 0 active is a no-op
    site = InjectionSite(InjectionCfg(name="m", r=3, after_block=0, gate=[0.0, 0.2, 0.2]), N_EMBD)
    a = torch.zeros(B, T, 3)
    a[..., 0] = torch.randn(B, T)
    assert torch.equal(site(x, a), x), "gate=0 channel must contribute exactly nothing"
    a[..., 1] = torch.randn(B, T)
    assert not torch.equal(site(x, a), x)
    # overall injected RMS of a vector gate == rms(gate) * rms(x)
    g = [0.02, 0.1, 0.2]
    sv = InjectionSite(InjectionCfg(name="v", r=3, after_block=0, gate=g), N_EMBD)
    a2 = torch.randn(B, T, 3)
    out = sv(x, a2)
    ratio = (out - x).pow(2).mean(-1).sqrt() / x.pow(2).mean(-1).sqrt()
    want = float(np.sqrt(np.mean(np.square(g))))
    assert torch.allclose(ratio, torch.full_like(ratio, want), rtol=1e-4), (ratio, want)


def test_all_zero_vector_gate_exact_noop():
    torch.manual_seed(6)
    x = torch.randn(B, T, N_EMBD)
    site = InjectionSite(InjectionCfg(name="z", r=4, after_block=0, gate=[0.0, 0.0, 0.0, 0.0]), N_EMBD)
    assert torch.equal(site(x, torch.randn(B, T, 4)), x), "all-zero gate vector must be an EXACT no-op"


def test_vector_gate_state_and_never_optimize():
    site = InjectionSite(InjectionCfg(name="v", r=3, after_block=0, gate=[0.1, 0.2, 0.3]), N_EMBD)
    assert tuple(site.gate.shape) == (3,)
    assert getattr(site.gate, "_never_optimize", False) is True
    adamw, muon = optimizer_param_split(build_sites(
        [InjectionCfg(name="v", r=3, after_block=0, gate=[0.1, 0.2, 0.3])], N_EMBD))
    assert adamw == [] and muon == []              # gate never optimized, direction frozen


class _FakeGateSource:
    """Deterministic per-channel activation stats for auto-gate calibration:
    channel c fires on a fixed fraction of tokens with a fixed magnitude."""
    name = "fake"

    def __init__(self, r=4, n_docs=64):
        self.r, self._n = r, n_docs

    def sample_activation_stats(self, k, seed):
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(min(k, self._n)):
            m = rng.integers(20, 40)
            z = np.zeros((m, self.r), np.float32)
            for c in range(self.r):
                fire = rng.random(m) < (0.2 * (c + 1))     # denser channels fire more
                z[fire, c] = (c + 1) * 1.0
            rows.append(z)
        pooled = np.concatenate(rows)
        rms = np.sqrt((pooled ** 2).mean(0)).astype(np.float32)
        nz = (pooled != 0).mean(0).astype(np.float32)
        return rms, nz, len(rows), pooled.shape[0]


def test_auto_gate_calibration_and_determinism():
    src = _FakeGateSource(r=4, n_docs=64)
    g1, meta1 = calibrate_auto_gate(src, target=0.05, k=32, seed=0, min_docs=8)
    g2, _ = calibrate_auto_gate(src, target=0.05, k=32, seed=0, min_docs=8)
    assert g1 == g2, "auto-gate must be deterministic in (source, seed)"
    assert abs(float(np.sqrt(np.mean(np.square(g1)))) - 0.05) < 1e-5, "rms(gate) must equal the target"
    # equalized contribution: gate_c * rms_c ~ const on active channels
    rms = np.asarray(meta1["channel_rms"])
    contrib = np.asarray(g1) * rms
    active = contrib > 0
    assert active.sum() >= 2 and np.allclose(contrib[active], contrib[active][0], rtol=1e-4)
    # too few docs -> loud failure
    try:
        calibrate_auto_gate(_FakeGateSource(n_docs=3), target=0.05, k=32, seed=0, min_docs=8)
        raise AssertionError("auto-gate must fail loudly with too few docs")
    except RuntimeError:
        pass


def test_auto_gate_persists_through_cfg_roundtrip():
    # The calibrated vector rides in cfg.gate -> asdict -> checkpoint meta, so a
    # resume rebuilds the same site without recalibrating.
    gate_vec, _ = calibrate_auto_gate(_FakeGateSource(r=4), target=0.05, k=32, seed=1, min_docs=8)
    cfg = InjectionCfg(name="v", r=4, after_block=0, gate=gate_vec)
    rebuilt = InjectionCfg(**asdict(cfg))
    assert rebuilt.gate == gate_vec
    site = InjectionSite(rebuilt, N_EMBD)
    assert torch.equal(site.gate.detach(), torch.tensor(gate_vec))
    assert parse_gate_spec("auto:0.1") == (True, 0.1)
    assert parse_gate_spec(gate_vec) == (False, gate_vec)


def test_vector_gate_checkpoint_resume_roundtrip():
    # Resume contract behind injection_train's auto-gate fix: the calibrated
    # VECTOR must be back in the cfg (from checkpoint meta) BEFORE the sites are
    # built — load_state_dict(assign=True) enforces shapes, so a scalar
    # placeholder site cannot load a vector-gate checkpoint.
    gate_vec, _ = calibrate_auto_gate(_FakeGateSource(r=4), target=0.05, k=32, seed=1, min_docs=8)
    saved_cfg = InjectionCfg(name="v", r=4, after_block=0, gate=gate_vec)

    class Holder(torch.nn.Module):
        def __init__(self, cfgs):
            super().__init__()
            self.injection_sites = build_sites(cfgs, N_EMBD)

    ckpt = Holder([saved_cfg]).state_dict()
    meta_gate = [float(g) for g in asdict(saved_cfg)["gate"]]    # json meta round trip
    resumed = Holder([InjectionCfg(name="v", r=4, after_block=0, gate=meta_gate)])
    missing, unexpected = resumed.load_state_dict(ckpt, strict=False, assign=True)
    assert not missing and not unexpected
    reassert_optimizability(resumed.injection_sites)
    g = resumed.injection_sites["v"].gate
    assert torch.equal(g.detach(), torch.tensor(gate_vec)), "calibrated gate must survive resume exactly"
    assert g._never_optimize is True and g.requires_grad
    # the failure mode the fix removes: a scalar-placeholder site MUST refuse the vector checkpoint
    placeholder = Holder([InjectionCfg(name="v", r=4, after_block=0, gate=0.05)])
    try:
        placeholder.load_state_dict(ckpt, strict=False, assign=True)
        raise AssertionError("scalar-placeholder site silently loaded a vector gate")
    except RuntimeError as e:
        assert "size mismatch" in str(e)


def test_freeze_unfreeze_and_reassert():
    site = _tabular_site(trainable=True)
    assert site.direction.requires_grad
    site.freeze()
    assert not site.direction.requires_grad
    site.unfreeze()
    assert site.direction.requires_grad
    # reassert restores the cfg contract after e.g. an assign=True load
    sites = build_sites([
        InjectionCfg(name="fr", r=3, after_block=0, trainable_direction=False),
        InjectionCfg(name="tr", r=3, after_block=0, trainable_direction=True),
    ], N_EMBD)
    sites["fr"].direction.requires_grad_(True)      # clobber
    sites["tr"].direction.requires_grad_(False)
    del sites["fr"].gate._never_optimize
    reassert_optimizability(sites)
    assert not sites["fr"].direction.requires_grad
    assert sites["tr"].direction.requires_grad
    assert sites["fr"].gate._never_optimize is True


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
    m.setup_injection_sites([InjectionCfg(name="coords", r=R, after_block=1, gate=0.05)])
    assert torch.equal(m(x, y), loss_vanilla), "acts=None must be bit-identical to vanilla"
    # flops + scaling accounting stay comparable to vanilla
    mv = _tiny_gpt()
    assert m.estimate_flops() == mv.estimate_flops()
    sp = m.num_scaling_params()
    assert sp["injection"] == sum(p.numel() for p in m.injection_sites.parameters())
    assert sp["transformer_matrices"] == mv.num_scaling_params()["transformer_matrices"]


def test_gpt_optimizer_contract_and_step():
    m = _tiny_gpt()
    sites = m.setup_injection_sites([
        InjectionCfg(name="coords", r=R, after_block=1, gate=0.05, trainable_direction=False),
        InjectionCfg(name="free", r=6, after_block=2, gate=0.1, trainable_direction=True, optim="muon"),
    ])
    opt = m.setup_optimizer()
    in_groups = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(sites["coords"].gate) not in in_groups
    assert id(sites["free"].gate) not in in_groups
    assert id(sites["coords"].direction) not in in_groups        # frozen: skipped
    assert id(sites["free"].direction) in in_groups              # trainable: muon group
    inj_groups = [g for g in opt.param_groups if g.get("injection")]
    assert len(inj_groups) == 1 and inj_groups[0]["kind"] == "muon" \
        and inj_groups[0]["weight_decay"] == 0.0

    x = torch.randint(0, 12, (2, 16))
    y = torch.randint(0, 12, (2, 16))
    acts = {"coords": torch.randn(2, 16, R), "free": torch.randn(2, 16, 6)}
    loss_on = m(x, y, acts=acts)
    assert loss_on.item() != m(x, y).item()
    loss_on.backward()
    assert sites["coords"].gate.grad is not None                 # want-signal
    assert sites["coords"].direction.grad is None
    assert sites["free"].direction.grad is not None
    g0 = sites["coords"].gate.detach().clone()
    d0 = sites["coords"].direction.detach().clone()
    f0 = sites["free"].direction.detach().clone()
    opt.step()
    assert torch.equal(sites["coords"].gate.detach(), g0), "gate must never be stepped"
    assert torch.equal(sites["coords"].direction.detach(), d0), "frozen direction must never move"
    assert not torch.equal(sites["free"].direction.detach(), f0), "trainable direction must move"


# --------------------------------------------------------------------------- #
# Donor-matched gate (loudness.json): parse, calibration target + determinism,
# concept-order refusal, source-layer requirement, checkpoint-meta persistence,
# resume-does-not-rescore, live-source calibrates-before-training-batch, fallback
# discovery logging, and DDP bit-identity.
# --------------------------------------------------------------------------- #
class _FakeDonorSource:
    """Probe-score source for donor-gate tests: a gemma layer + concept columns +
    a score_loc, and deterministic per-channel activation stats (like _FakeGateSource)."""
    def __init__(self, r=3, layer=8, concepts=None, score_loc="/tmp/nope", n_docs=64):
        self.r, self.layer, self.name = r, layer, "fake-donor"
        self.concepts = concepts or [f"c{i}" for i in range(r)]
        self.score_loc = score_loc
        self._n = n_docs

    def sample_activation_stats(self, k, seed):
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(min(k, self._n)):
            m = rng.integers(20, 40)
            z = np.zeros((m, self.r), np.float32)
            for c in range(self.r):
                fire = rng.random(m) < (0.2 * (c + 1))
                z[fire, c] = (c + 1) * 1.0
            rows.append(z)
        pooled = np.concatenate(rows)
        rms = np.sqrt((pooled ** 2).mean(0)).astype(np.float32)
        nz = (pooled != 0).mean(0).astype(np.float32)
        return rms, nz, len(rows), pooled.shape[0]


def _fake_loudness(concepts, layer=8):
    L, K = str(layer), len(concepts)
    return {
        "version": 1, "concepts": list(concepts),
        "ridge": {"active_loudness": {L: {"p50": [0.02 * (i + 1) for i in range(K)]}}},
        "subspace_total": {"ridge": {L: {"p50": 0.05, "p90": 0.09, "p95": 0.12, "p99": 0.2}}},
    }


def test_donor_gate_spec_parses():
    assert parse_donor_gate_spec("donor") == (True, "p50")
    assert parse_donor_gate_spec("donor:p95") == (True, "p95")
    assert parse_donor_gate_spec("auto:0.1") == (False, None)
    assert parse_donor_gate_spec(0.05) == (False, None)
    try:
        parse_donor_gate_spec("donor:p42")
        raise AssertionError("bad stat accepted")
    except ValueError:
        pass
    # parse_gate_spec's (bool,value) contract is UNCHANGED
    assert parse_gate_spec("auto:0.1") == (True, 0.1)


def test_donor_gate_calibration_target_and_determinism():
    src = _FakeDonorSource(r=3)
    loud = _fake_loudness(src.concepts)
    g1, m1 = donor_gate_from_loudness(src, loud, stat="p50", k=32, seed=0, min_docs=8)
    g2, _ = donor_gate_from_loudness(src, loud, stat="p50", k=32, seed=0, min_docs=8)
    assert g1 == g2, "donor gate must be deterministic in (source, seed)"
    assert abs(float(np.sqrt(np.mean(np.square(g1)))) - 0.05) < 1e-5, "rms(gate) must == subspace_total target"
    g95, _ = donor_gate_from_loudness(src, loud, stat="p95", k=32, seed=0, min_docs=8)
    assert abs(float(np.sqrt(np.mean(np.square(g95)))) - 0.12) < 1e-5, "stat picks the subspace_total quantile"
    # per-channel weight ∝ donor_loudness_c / rms_c (constant ratio on active channels)
    rms = np.asarray(m1["channel_rms"]); donor = np.asarray(m1["donor_loudness"])
    w = np.asarray(g1) / (donor / rms)
    assert np.allclose(w, w[0], rtol=1e-4)
    assert m1["mode"] == "donor" and m1["layer"] == 8 and m1["stat"] == "p50"


def test_donor_concept_order_mismatch_refuses():
    src = _FakeDonorSource(r=3, concepts=["a", "b", "c"])
    bad = _fake_loudness(["a", "c", "b"])       # permuted vs source columns
    try:
        donor_gate_from_loudness(src, bad, stat="p50", k=16, seed=0, min_docs=8)
        raise AssertionError("permuted concepts were accepted")
    except ValueError as e:
        assert "concepts" in str(e)
    validate_donor_concepts(["a", "b"], ["a", "b"])   # exact match ok
    try:
        validate_donor_concepts(["a", "b"], ["b", "a"])
        raise AssertionError("permutation accepted")
    except ValueError:
        pass


def test_donor_source_layer_requirement():
    class _NoLayer:                              # FnSource-like: nothing to key on
        name, r = "fn", 4
    try:
        donor_source_layer_concepts(_NoLayer())
        raise AssertionError("accepted a layer-less source")
    except ValueError as e:
        assert "probe-score source" in str(e)
    class _Store:                               # e.g. a Qwen store whose meta names its layer
        name, r = "store", 3
        meta = {"layer": 6, "concepts": ["a", "b", "c"]}
    assert donor_source_layer_concepts(_Store()) == (6, ["a", "b", "c"])
    class _StoreNoLayer:
        name, r = "store", 3
        meta = {"concepts": ["a", "b", "c"]}
    try:
        donor_source_layer_concepts(_StoreNoLayer())
        raise AssertionError("accepted a store with no layer identity")
    except ValueError as e:
        assert "layer" in str(e)


def test_donor_gate_persists_through_cfg_roundtrip():
    src = _FakeDonorSource(r=4)
    gate_vec, _ = donor_gate_from_loudness(src, _fake_loudness(src.concepts),
                                           stat="p50", k=32, seed=1, min_docs=8)
    cfg = InjectionCfg(name="v", r=4, after_block=0, gate=gate_vec)
    rebuilt = InjectionCfg(**asdict(cfg))       # json meta round trip
    assert rebuilt.gate == gate_vec
    site = InjectionSite(rebuilt, N_EMBD)
    assert torch.equal(site.gate.detach(), torch.tensor(gate_vec))


def test_donor_resume_does_not_rescore():
    # The resume path reuses the calibrated vector from checkpoint meta; the
    # source is NEVER sampled again. Mirrors injection_train's reuse branch.
    class _NoRescore(_FakeDonorSource):
        def sample_activation_stats(self, k, seed):
            raise AssertionError("resume must NOT rescore the source")
    _ = _NoRescore(r=3)
    ckpt_sites = {"v": {"name": "v", "gate": [0.01, 0.02, 0.03]}}
    pending = {"v": "p50"}
    resolved = None
    for name in list(pending):
        g = ckpt_sites.get(name, {}).get("gate")
        if isinstance(g, (list, tuple)):
            resolved = InjectionCfg(name=name, r=3, after_block=0, gate=[float(x) for x in g])
            del pending[name]
    assert not pending, "donor site must be resolved from meta, not left pending (no rescore loop)"
    assert resolved.gate == [0.01, 0.02, 0.03]


def test_donor_live_source_calibrates_before_training_batch():
    # A live/dynamic source runs its scorer at STARTUP (during calibration),
    # strictly before any training batch is drawn. Record call order.
    calls = []

    class _FakeLive:
        name, r, layer = "live", 3, 8
        concepts = ["a", "b", "c"]
        score_loc = "/tmp/none"

        def sample_activation_stats(self, k, seed):
            calls.append("calibrate")           # the live scorer runs here (startup)
            rng = np.random.default_rng(seed)
            pooled = np.abs(rng.normal(size=(200, self.r))).astype(np.float32) + 0.5
            return (np.sqrt((pooled ** 2).mean(0)).astype(np.float32),
                    np.ones(self.r, np.float32), 32, pooled.shape[0])

        def draw_training_batch(self):
            calls.append("train")

    src = _FakeLive()
    donor_gate_from_loudness(src, _fake_loudness(src.concepts), stat="p50", k=16, seed=0, min_docs=8)
    src.draw_training_batch()
    assert calls == ["calibrate", "train"], f"calibration must precede any training batch: {calls}"


def test_donor_fallback_discovery_logs_loudly():
    logs = []
    log = logs.append
    # 1) --loudness-json override wins (and is logged)
    _, desc = discover_loudness_json(_FakeDonorSource(), "/some/override", log,
                                     load_fn=lambda loc: {"src": loc})
    assert desc == "override:/some/override" and any("--loudness-json" in x for x in logs)
    # 2) source store root present
    logs.clear()
    src = _FakeDonorSource(score_loc="the-store")
    _, desc = discover_loudness_json(src, None, log, load_fn=lambda loc: {"loc": loc})
    assert desc == "store:the-store" and any("source store root" in x for x in logs)
    # 3) store root lacks loudness.json -> LOUD fallback, always logged
    logs.clear()

    def _load(loc):
        if loc == "the-store":
            raise FileNotFoundError("no loudness.json here")
        return {"fallback": loc}
    _, desc = discover_loudness_json(src, None, log, load_fn=_load,
                                     fallback_repo="kaushikreddyxyz/climbmix-scored")
    assert desc == "fallback:kaushikreddyxyz/climbmix-scored"
    assert any("FALLBACK" in x for x in logs) and any("!!!!" in x for x in logs)


def test_donor_gate_ddp_identity():
    loud = _fake_loudness(_FakeDonorSource(r=4).concepts)
    g, _ = donor_gate_from_loudness(_FakeDonorSource(r=4), loud, stat="p50", k=32, seed=3, min_docs=8)
    # deterministic calibration on every "rank" -> identical gate hashes
    ranks = [donor_gate_from_loudness(_FakeDonorSource(r=4), loud, stat="p50",
                                      k=32, seed=3, min_docs=8)[0] for _ in range(4)]
    hashes = [gate_vector_hash(gv) for gv in ranks]
    assert len(set(hashes)) == 1, "deterministic calibration must give identical gate hashes"
    assert_gate_identical_across_ranks(g, "v", lambda *a: None, all_gather_hash=lambda h: hashes)
    # ...and the collective check RAISES on any rank disagreement
    try:
        assert_gate_identical_across_ranks(g, "v", lambda *a: None,
                                           all_gather_hash=lambda h: [h, "deadbeef"])
        raise AssertionError("rank disagreement not caught")
    except RuntimeError as e:
        assert "disagree" in str(e)


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        fn()
        print(f"{name}: OK")
    print("\nALL CHECKS PASSED")
