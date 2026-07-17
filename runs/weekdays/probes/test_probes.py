"""CPU tests for the weekday probe experiment. NO network / rustbpe / gemma.
Plain asserts (run: `python runs/weekdays/probes/test_probes.py`).

Covers:
  1. pack_batches: every doc row covered exactly once, BOS layout, masks, caps;
     padded-batch features == solo-doc features (causal no-contamination).
  2. ridge + R^2 moment algebra == direct dense computation (synthetic).
  3. capture equivalence: gate_scale=0 site-hook capture == vanilla block-hook
     capture bitwise (the OFF stream is the true no-injection stream).
  4. end-to-end planted recovery on a tiny GPT: the ON probe's raw-space
     readout rows match the site's direction rows (argmax by |cos|, high cos);
     the OFF probe has ~no signal (random-init model, targets independent).
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "runs", "weekdays", "eval"))

import harness  # noqa: E402
import probe_lib  # noqa: E402
from probe_lib import Moments, pack_batches, predict, r2_from_moments, run_condition, solve_ridge  # noqa: E402


def _tiny_gpt(with_site, gate=0.5, seed=0, n_embd=64):
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.injection.sites import InjectionCfg
    cfg = GPTConfig(sequence_len=128, vocab_size=64, n_layer=4, n_head=2,
                    n_kv_head=2, n_embd=n_embd, window_pattern="L")
    torch.manual_seed(seed)
    with torch.device("meta"):
        m = GPT(cfg)
    m.to_empty(device="cpu")
    torch.manual_seed(seed)
    m.init_weights()
    m.eval()
    if with_site:
        m.setup_injection_sites([InjectionCfg(name="weekdays", r=7, after_block=1,
                                              gate=gate, direction_seed=7)])
    return m


def _synth_docs(n_docs=48, seed=0, r=7, p_active=0.2):
    """Random-token docs with sparse single-channel positive acts (z ~ 2..4),
    mimicking the thresholded weekday source."""
    rng = np.random.RandomState(seed)
    docs = []
    for i in range(n_docs):
        n = rng.randint(20, 90)
        ids = rng.randint(0, 60, n).astype(np.int32)
        acts = np.zeros((n, r), np.float32)
        hot = rng.rand(n) < p_active
        acts[hot, rng.randint(0, r, hot.sum())] = rng.uniform(2.0, 4.0, hot.sum())
        docs.append({"ids": ids, "acts": acts, "doc_idx": i})
    return docs


# --------------------------------------------------------------------------- #
def test_pack_batches():
    docs = _synth_docs(17, seed=1)
    seen = {}
    for ids, acts, mask, doc_idxs in pack_batches(docs, bos_id=63, max_rows=4,
                                                  max_tokens=400):
        B, T = ids.shape
        assert B <= 4 and (B == 1 or B * T <= 400)
        assert (ids[:, 0] == 63).all() and not mask[:, 0].any()
        for b, di in enumerate(doc_idxs):
            n = len(docs[di]["ids"])
            assert (ids[b, 1:1 + n].numpy() == docs[di]["ids"]).all()
            assert np.allclose(acts[b, 1:1 + n].numpy(), docs[di]["acts"])
            assert mask[b, 1:1 + n].all() and not mask[b, 1 + n:].any()
            assert di not in seen
            seen[di] = True
    assert len(seen) == len(docs), "every doc emitted exactly once"

    # padded-batch features == solo forward (causality: pad never contaminates)
    m = _tiny_gpt(with_site=True)
    cap = probe_lib.ResidCapture(m)
    batch = next(pack_batches(docs[:3], bos_id=63, max_rows=3, max_tokens=10 ** 6))
    ids, acts, mask, doc_idxs = batch
    harness.forward_metrics(m, ids, acts=acts, gate_scale=1.0, return_logits=False)
    Xb = cap.pop()
    for b, di in enumerate(doc_idxs):
        n = len(docs[di]["ids"])
        solo_ids = torch.cat([torch.tensor([63]), torch.from_numpy(docs[di]["ids"].astype(np.int64))])
        solo_acts = np.concatenate([np.zeros((1, 7), np.float32), docs[di]["acts"]])
        harness.forward_metrics(m, solo_ids, acts=solo_acts[None], gate_scale=1.0,
                                return_logits=False)
        Xs = cap.pop()
        assert torch.allclose(Xb[b, :1 + n], Xs[0, :1 + n], atol=1e-5), \
            f"padded-batch features drifted from solo forward (doc {di})"
    cap.close()
    print("ok: pack_batches (+ causal padding equivalence)")


# --------------------------------------------------------------------------- #
def test_ridge_algebra():
    rng = np.random.RandomState(0)
    d, k, n_tr, n_te = 24, 5, 4000, 1200
    Wtrue = rng.randn(d, k)
    Xtr = rng.randn(n_tr, d) * rng.uniform(0.5, 2.0, d) + rng.randn(d)
    Xte = rng.randn(n_te, d) * Xtr.std(0) + Xtr.mean(0)
    Ytr = Xtr @ Wtrue + rng.randn(n_tr, k) * 3.0
    Yte = Xte @ Wtrue + rng.randn(n_te, k) * 3.0

    mtr, mte = Moments(d, k), Moments(d, k)
    for X, Y, m in ((Xtr, Ytr, mtr), (Xte, Yte, mte)):
        for i in range(0, len(X), 517):    # uneven chunks: accumulation order
            m.add(torch.tensor(X[i:i + 517], dtype=torch.float32),
                  torch.tensor(Y[i:i + 517], dtype=torch.float32))

    for lam in (1e-4, 1e-2, 1.0):
        probe = solve_ridge(mtr, lam)
        # direct standardized ridge on the raw rows
        mu, sig = Xtr.mean(0), Xtr.std(0)
        Z = (Xtr - mu) / sig
        Wd = np.linalg.solve(Z.T @ Z / n_tr + lam * np.eye(d),
                             Z.T @ (Ytr - Ytr.mean(0)) / n_tr)
        assert np.allclose(probe["W"], Wd, atol=1e-4), f"ridge weights differ (lam={lam})"
        # moment R^2 == direct R^2 on the test rows
        pred = predict(probe, Xte)
        r2_direct = 1 - ((Yte - pred) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)
        r2_mom = r2_from_moments(probe, mte)
        assert np.allclose(r2_mom, r2_direct, atol=1e-6), f"R2 algebra broke (lam={lam})"
    print("ok: ridge + R2 moment algebra == direct")


# --------------------------------------------------------------------------- #
def test_capture_off_equals_vanilla():
    m = _tiny_gpt(with_site=True)
    docs = _synth_docs(6, seed=2)
    batch = next(pack_batches(docs, bos_id=63, max_rows=6, max_tokens=10 ** 6))
    ids, acts, _, _ = batch

    cap = probe_lib.ResidCapture(m)
    assert cap.uses_site
    harness.forward_metrics(m, ids, acts=acts, gate_scale=0.0, return_logits=False)
    x_off = cap.pop()
    cap.close()

    # forces the block hook, at the tiny model's site block (after_block=1)
    cap_blk = probe_lib.ResidCapture(m, after_block=1, site_name="nope-no-site")
    assert not cap_blk.uses_site
    harness.forward_metrics(m, ids, acts=None, return_logits=False)
    x_vanilla = cap_blk.pop()
    cap_blk.close()
    assert torch.equal(x_off, x_vanilla), \
        "gate_scale=0 site capture must be bitwise == vanilla block capture"
    print("ok: OFF capture == vanilla block capture (bitwise)")


# --------------------------------------------------------------------------- #
def test_planted_direction_recovery():
    m = _tiny_gpt(with_site=True, gate=0.5)
    D = m.injection_sites["weekdays"].direction.detach().numpy()   # [7, 64]
    docs = _synth_docs(60, seed=3)

    res = {}
    for name, gs in (("on", 1.0), ("off", 0.0)):
        res[name] = run_condition(m, docs, bos_id=63, gate_scale=gs, use_acts=True,
                                  device="cpu", forward_metrics=harness.forward_metrics,
                                  log=lambda s: None)
    on, off = res["on"]["pops"]["act"], res["off"]["pops"]["act"]

    # moment R^2 crosschecks the direct row-buffer R^2
    assert np.allclose(on["r2_heldout"], on["r2_direct_te_act"], atol=5e-3), \
        "moment R2 != direct R2 on captured rows"

    # ON probe reads the planted code; OFF probe has nothing to read
    assert on["r2_heldout_mean"] > 0.5, f"ON act-R2 too low: {on['r2_heldout_mean']}"
    assert off["r2_heldout_mean"] < 0.15, f"OFF act-R2 unexpectedly high: {off['r2_heldout_mean']}"
    assert on["argmax_acc_te_act"] > 0.9, f"ON argmax acc: {on['argmax_acc_te_act']}"

    Du = D / np.linalg.norm(D, axis=1, keepdims=True)

    # ENCODING estimate (raw cross-covariance, all rows) recovers D sharply:
    # the site writes channel c along D_c, so Cxy[:, c] must point along D_c.
    on_all, off_all = res["on"]["pops"]["all"], res["off"]["pops"]["all"]
    E = on_all["Cxy"].T                                              # [7, d]
    Eu = E / np.linalg.norm(E, axis=1, keepdims=True)
    Ce = Eu @ Du.T
    assert (Ce.argmax(1) == np.arange(7)).all(), f"encoding day mismatch:\n{np.round(Ce, 2)}"
    assert np.diag(Ce).min() > 0.85, f"encoding matched cos low: {np.round(np.diag(Ce), 3)}"
    Eoff = off_all["Cxy"].T
    off_enc = np.diag((Eoff / np.linalg.norm(Eoff, axis=1, keepdims=True)) @ Du.T)
    assert np.abs(off_enc).mean() < 0.3, f"OFF encoding aligned with D?! {off_enc}"

    # DECODING readout V is covariance-whitened => only a relative alignment
    # claim: matched cos clearly above the off-diagonal background.
    Vu = on["V"] / np.linalg.norm(on["V"], axis=1, keepdims=True)
    C = Vu @ Du.T
    diag = np.diag(C)
    offd = np.abs(C[~np.eye(7, dtype=bool)])
    assert diag.mean() > max(2.0 * offd.mean(), 0.05), \
        f"decoder diag {diag.mean():.3f} not above background {offd.mean():.3f}"
    print(f"ok: planted recovery — ON act-R2 {on['r2_heldout_mean']:.3f}, "
          f"enc matched cos min {np.diag(Ce).min():.3f}, dec diag mean {diag.mean():.3f} "
          f"(bg {offd.mean():.3f}); OFF act-R2 {off['r2_heldout_mean']:.3f}")


if __name__ == "__main__":
    test_pack_batches()
    test_ridge_algebra()
    test_capture_off_equals_vanilla()
    test_planted_direction_recovery()
    print("ALL PROBE TESTS PASSED")
