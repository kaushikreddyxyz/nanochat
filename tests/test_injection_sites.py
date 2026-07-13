"""CPU tests for nanochat.oracle.injections (v2 injection sites) and their
wiring into GPT.

Pins the invariants the injection design depends on:
  * v1/v2 forward-value equivalence: the retired inline v1 math
    ``x + beta*(rms_x/rms_z)*zc`` (fixed orthonormal P) equals the
    InjectionSite forward ``x + gate*rms_x.detach()*z/rms_z`` with
    direction = P.T and gate = beta -- identical forward values; the detach
    only changes gradients, deliberately.
  * RMS calibration: injected per-token RMS == gate * per-token RMS(x).
  * zero activation rows are an EXACT no-op (bitwise).
  * gate=0 is an exact forward no-op AND blocks all gradient to the direction,
    while the gate itself still receives a gradient (loggable want-signal).
  * optimizer contract: gates never appear in the split; frozen directions
    excluded; trainable directions bucketed adamw/muon per cfg.
  * state_dict carries exactly {gate, direction, channel_weights} per site.
  * GPT integration: acts=None forward is bit-identical to a vanilla model,
    setup_optimizer covers/excludes the right params, optimizer steps leave
    gates + frozen directions untouched.
"""
import os
import sys

import numpy as np
import torch

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(TESTS, ".."))
sys.path.insert(0, REPO)

from nanochat.oracle.coords_store import make_orthonormal_P  # noqa: E402
from nanochat.oracle.injections import (  # noqa: E402
    InjectionCfg,
    InjectionSite,
    build_sites,
    optimizer_param_split,
    orthonormal_direction,
    reassert_optimizability,
    sites_by_block,
)

B, T, N_EMBD, R = 2, 8, 64, 14


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
    legacy --inject-coords path reproduces the store's P from the same seed."""
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
    assert set(site.state_dict().keys()) == {"gate", "direction", "channel_weights"}
    sites = build_sites([InjectionCfg(name="a", r=3, after_block=0)], N_EMBD)
    assert set(sites.state_dict().keys()) == {"a.gate", "a.direction", "a.channel_weights"}


def test_channel_weights_mute():
    torch.manual_seed(5)
    x = torch.randn(B, T, N_EMBD)
    cfg = InjectionCfg(name="m", r=3, after_block=0, gate=0.2,
                       channel_weights=[0.0, 1.0, 1.0])
    site = InjectionSite(cfg, N_EMBD)
    a = torch.zeros(B, T, 3)
    a[..., 0] = torch.randn(B, T)                # only the muted channel is active
    assert torch.equal(site(x, a), x), "muted channel must contribute exactly nothing"
    a[..., 1] = torch.randn(B, T)
    assert not torch.equal(site(x, a), x)


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


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        fn()
        print(f"{name}: OK")
    print("\nALL CHECKS PASSED")
