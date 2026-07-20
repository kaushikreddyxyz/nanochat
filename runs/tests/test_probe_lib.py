"""probe_lib tests: moment-accumulated ridge / R^2 against a brute-force reference,
doc batching layout, and the parameterized ProbeSpec. CPU-only, no model."""
import numpy as np
import pytest
import torch

import probe_lib
from probe_lib import Moments, ProbeSpec, RowBuffer, pack_batches, predict, r2_from_moments, solve_ridge


def _moments(X, Y):
    m = Moments(X.shape[1], Y.shape[1])
    m.add(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(Y.astype(np.float32)))
    return m


def _brute_ridge(X, Y, lam, eps=1e-8):
    """Same estimator, computed directly from the samples instead of moments."""
    n = X.shape[0]
    mu = X.mean(0)
    ybar = Y.mean(0)
    sigma = np.sqrt(np.maximum((X ** 2).mean(0) - mu ** 2, eps))
    Z = (X - mu) / sigma
    Sxx = Z.T @ Z / n
    Sxy = Z.T @ (Y - ybar) / n
    W = np.linalg.solve(Sxx + lam * np.eye(X.shape[1]), Sxy)
    return W, mu, sigma, ybar


@pytest.mark.parametrize("lam", [1e-5, 1e-2, 1.0])
def test_solve_ridge_matches_brute_force(lam):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 9)) * rng.uniform(0.5, 3.0, 9) + rng.normal(size=9)
    Y = X @ rng.normal(size=(9, 4)) + 0.1 * rng.normal(size=(400, 4))
    probe = solve_ridge(_moments(X, Y), lam)
    W, mu, sigma, ybar = _brute_ridge(X, Y, lam)
    # Moments accumulates fp32 batch products, so agreement is fp32-limited.
    assert np.allclose(probe["W"], W, atol=1e-4)
    assert np.allclose(probe["mu"], mu, atol=1e-5)
    assert np.allclose(probe["sigma"], sigma, atol=1e-5)
    assert np.allclose(probe["ybar"], ybar, atol=1e-5)


def test_readout_rows_are_raw_space_v_equals_w_over_sigma():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 6)) * 2.0 + 1.0
    Y = rng.normal(size=(200, 3))
    probe = solve_ridge(_moments(X, Y), 1e-3)
    assert probe["V"].shape == (3, 6)
    assert np.allclose(probe["V"], (probe["W"] / probe["sigma"][:, None]).T)
    # V is the raw-space readout: (x - mu) @ V[c] + ybar[c] == the prediction
    pred = predict(probe, X)
    assert np.allclose(pred, (X - probe["mu"]) @ probe["V"].T + probe["ybar"])


def test_r2_from_moments_matches_direct_computation():
    rng = np.random.default_rng(2)
    Xtr = rng.normal(size=(300, 5))
    Ytr = Xtr @ rng.normal(size=(5, 3)) + 0.3 * rng.normal(size=(300, 3))
    Xte = rng.normal(size=(120, 5))
    Yte = Xte @ rng.normal(size=(5, 3)) + 0.3 * rng.normal(size=(120, 3))
    probe = solve_ridge(_moments(Xtr, Ytr), 1e-2)
    got = r2_from_moments(probe, _moments(Xte, Yte))
    pred = predict(probe, Xte)
    direct = 1 - ((Yte - pred) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)
    assert np.allclose(got, direct, atol=1e-8)


def test_moments_accumulate_incrementally():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 4)).astype(np.float32)
    Y = rng.normal(size=(50, 2)).astype(np.float32)
    whole = _moments(X, Y)
    parts = Moments(4, 2)
    for lo, hi in ((0, 17), (17, 34), (34, 50)):
        parts.add(torch.from_numpy(X[lo:hi]), torch.from_numpy(Y[lo:hi]))
    parts.add(torch.zeros(0, 4), torch.zeros(0, 2))       # empty batch is a no-op
    assert parts.n == whole.n == 50
    for k in ("sx", "sy", "xtx", "xty", "yty"):
        assert np.allclose(getattr(parts, k), getattr(whole, k), atol=1e-4)


def test_pack_batches_layout_and_mask():
    rng = np.random.default_rng(4)
    docs = [{"ids": rng.integers(1, 100, n), "acts": rng.normal(size=(n, 4)).astype(np.float32),
             "doc_idx": i} for i, n in enumerate([5, 12, 3, 9])]
    seen = {}
    for ids, acts, mask, idxs in pack_batches(docs, bos_id=7, max_rows=2, max_tokens=10 ** 6):
        assert ids.shape == mask.shape and acts.shape[:2] == ids.shape
        for b, di in enumerate(idxs):
            n = len(docs[di]["ids"])
            assert ids[b, 0].item() == 7                      # BOS first
            assert mask[b, 0].item() is False or not mask[b, 0]
            assert mask[b].sum().item() == n                  # body positions only
            assert np.array_equal(ids[b, 1:1 + n].numpy(), docs[di]["ids"])
            assert np.allclose(acts[b, 1:1 + n].numpy(), docs[di]["acts"])
            assert not acts[b, 1 + n:].any()                  # pad acts stay zero
            seen[di] = True
    assert sorted(seen) == [0, 1, 2, 3]


def test_pack_batches_respects_row_cap():
    docs = [{"ids": np.arange(4), "acts": np.zeros((4, 2), np.float32), "doc_idx": i}
            for i in range(5)]
    sizes = [len(idxs) for *_, idxs in pack_batches(docs, 1, max_rows=2, max_tokens=10 ** 6)]
    assert max(sizes) <= 2 and sum(sizes) == 5


def test_row_buffer_is_sized_from_the_spec_not_a_module_constant():
    buf = RowBuffer(d=11, k=3, cap=4)
    X, Y = buf.arrays()
    assert X.shape == (0, 11) and Y.shape == (0, 3)
    buf.add(torch.ones(6, 11), torch.ones(6, 3))
    X, Y = buf.arrays()
    assert X.shape == (4, 11) and Y.shape == (4, 3)          # capped
    buf.add(torch.ones(2, 11), torch.ones(2, 3))
    assert buf.arrays()[0].shape == (4, 11)                  # full -> no-op


@pytest.mark.parametrize("family,after_block,r", [("weekdays", 3, 7), ("seasons", 0, 4)])
def test_probe_spec_from_family(family, after_block, r):
    spec = ProbeSpec.from_family(family, after_block=after_block, d_model=768)
    assert spec.site_name == family and spec.after_block == after_block
    assert spec.r == r and spec.d_model == 768
    assert spec.lambda_grid == probe_lib.LAMBDA_GRID
    import concepts
    assert list(spec.concepts) == list(concepts.get_family(family).store_order)
