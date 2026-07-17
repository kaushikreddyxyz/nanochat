"""Core library for the weekday PROBE experiment: train linear (ridge) probes on
each arm's residual stream AT THE INJECTION LAYER (post-block-3, post-site),
with injection ON vs OFF, and compare probe readout directions to the injection
direction matrix D.

Probe targets = the SAME aligned+thresholded gemma weekday z-scores training
injected (WeekdayProbeScoreSource output; store channel order friday..wednesday
= climbmix cols 47..53). Features = the 768-d residual entering block 4:
  * arms WITH a site: forward with acts at gate_scale 1.0 (ON) or 0.0 (OFF)
    through the pinned harness path, capture the SITE's forward output
    (gate 0 => site output == block output, verified in test_probes.py);
  * baseline (no site): capture transformer.h[AFTER_BLOCK]'s output.

Two probe populations per condition, both moment-accumulated (no activation
dumps): "all" rows (every non-BOS non-pad token) and "act" rows (acts row
nonzero — the tokens the injection actually fires on). Ridge is solved in
standardized feature space from exact fp64 moments; heldout R^2 is computed
from moments (closed form) and cross-checked on a raw row buffer that also
gives day-argmax accuracy. Raw-space readout direction rows V = W / sigma
(the same recipe as the gemma probes: build_manifold.py's V = W / nat_std).

Everything here is CPU-testable; no network, no tokenizer, no HF.
"""
import numpy as np
import torch

AFTER_BLOCK = 3
SITE_NAME = "weekdays"
D_MODEL = 768
R = 7
LAMBDA_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)  # on correlation-scaled Sxx (diag ~ 1)
STORE = ["friday", "monday", "saturday", "sunday", "thursday", "tuesday", "wednesday"]


# --------------------------------------------------------------------------- #
# residual capture at the injection point
# --------------------------------------------------------------------------- #
class ResidCapture:
    """Forward hook capturing the post-block-3 (+site) residual, fp32.

    Arms with a 'weekdays' site hook the SITE module (its output is the stream
    the next block sees, injection included; at gate_scale 0 it equals the block
    output exactly). Site-less models hook transformer.h[after_block]. NOTE: the
    site only fires when the forward receives acts — pass acts for every gated
    condition (the pinned on-vs-off invariant does this anyway)."""

    def __init__(self, model, after_block=AFTER_BLOCK, site_name=SITE_NAME):
        sites = getattr(model, "injection_sites", None)
        self.uses_site = sites is not None and site_name in sites
        module = sites[site_name] if self.uses_site else model.transformer.h[after_block]
        self.buf = None
        self._h = module.register_forward_hook(self._store)

    def _store(self, _m, _inp, out):
        self.buf = out.detach().float()

    def pop(self):
        x, self.buf = self.buf, None
        assert x is not None, "ResidCapture: no forward ran (site not fired? acts missing?)"
        return x

    def close(self):
        self._h.remove()


# --------------------------------------------------------------------------- #
# doc batching (pad-to-max; causal model => trailing pad never contaminates
# earlier positions; pad rows excluded via mask)
# --------------------------------------------------------------------------- #
def pack_batches(docs, bos_id, max_rows=32, max_tokens=65536):
    """docs: list of dicts {ids: int array [n], acts: float32 [n, r], doc_idx}.
    Yields (ids [B,T] long, acts [B,T,r] fp32, mask [B,T] bool, doc_idxs).
    Row layout: [BOS] + body; mask True on body positions (t>=1, < 1+n).
    Sorted by length desc so pad waste stays small."""
    order = sorted(range(len(docs)), key=lambda i: -len(docs[i]["ids"]))
    batch = []

    def emit(idxs):
        T = 1 + max(len(docs[i]["ids"]) for i in idxs)
        B = len(idxs)
        r = docs[idxs[0]]["acts"].shape[1]
        ids = np.full((B, T), bos_id, np.int64)
        acts = np.zeros((B, T, r), np.float32)
        mask = np.zeros((B, T), bool)
        for b, i in enumerate(idxs):
            n = len(docs[i]["ids"])
            ids[b, 1:1 + n] = docs[i]["ids"]
            acts[b, 1:1 + n] = docs[i]["acts"]
            mask[b, 1:1 + n] = True
        return (torch.from_numpy(ids), torch.from_numpy(acts),
                torch.from_numpy(mask), [docs[i]["doc_idx"] for i in idxs])

    for i in order:
        cand = batch + [i]
        T = 1 + max(len(docs[j]["ids"]) for j in cand)
        if batch and (len(cand) > max_rows or len(cand) * T > max_tokens):
            yield emit(batch)
            batch = [i]
        else:
            batch = cand
    if batch:
        yield emit(batch)


# --------------------------------------------------------------------------- #
# exact fp64 moment accumulation
# --------------------------------------------------------------------------- #
class Moments:
    """Sufficient statistics for ridge + R^2: n, Sx, Sy, XtX, XtY, y^2 (per
    channel). Batch-local products in fp32 on-device, accumulated in fp64."""

    def __init__(self, d=D_MODEL, k=R):
        self.n = 0
        self.sx = np.zeros(d, np.float64)
        self.sy = np.zeros(k, np.float64)
        self.xtx = np.zeros((d, d), np.float64)
        self.xty = np.zeros((d, k), np.float64)
        self.yty = np.zeros(k, np.float64)

    def add(self, X, Y):
        """X [n,d] fp32 tensor (any device), Y [n,k] fp32 tensor."""
        if X.shape[0] == 0:
            return
        self.n += X.shape[0]
        self.sx += X.sum(0).double().cpu().numpy()
        self.sy += Y.sum(0).double().cpu().numpy()
        self.xtx += (X.t() @ X).double().cpu().numpy()
        self.xty += (X.t() @ Y).double().cpu().numpy()
        self.yty += (Y * Y).sum(0).double().cpu().numpy()


def solve_ridge(m: Moments, lam: float, eps=1e-8):
    """Ridge in standardized-feature space from train moments.
    Returns dict: W [d,k] std-space weights, mu/sigma [d], ybar [k],
    V [k,d] raw-space readout rows (V[c] = W[:,c]/sigma — gemma-probe recipe)."""
    n = m.n
    assert n > 0
    mu = m.sx / n
    ybar = m.sy / n
    var = np.maximum(np.diag(m.xtx) / n - mu ** 2, eps)
    sigma = np.sqrt(var)
    cxx = m.xtx / n - np.outer(mu, mu)
    sxx = cxx / np.outer(sigma, sigma)                    # correlation-scaled
    sxy = (m.xty / n - np.outer(mu, ybar)) / sigma[:, None]
    W = np.linalg.solve(sxx + lam * np.eye(len(mu)), sxy)
    V = (W / sigma[:, None]).T                            # [k, d] raw-space rows
    return {"W": W, "mu": mu, "sigma": sigma, "ybar": ybar, "V": V, "lam": lam}


def r2_from_moments(probe, m: Moments):
    """Per-channel heldout R^2 of yhat = ((x-mu)/sigma) @ W + ybar against the
    TEST moments m (closed form; exact). Baseline = test-set channel mean."""
    W, mu, sigma, ybar = probe["W"], probe["mu"], probe["sigma"], probe["ybar"]
    n = m.n
    assert n > 0
    zy = (m.xty - np.outer(mu, m.sy)) / sigma[:, None]            # sum z y^T [d,k]
    t = (m.sx - n * mu) / sigma                                   # sum z [d]
    zzz = (m.xtx - np.outer(m.sx, mu) - np.outer(mu, m.sx)
           + n * np.outer(mu, mu)) / np.outer(sigma, sigma)       # sum z z^T
    cross = (W * zy).sum(0) + ybar * m.sy                         # sum y yhat [k]
    quad = (W * (zzz @ W)).sum(0) + 2 * ybar * (W.T @ t) + n * ybar ** 2
    sse = m.yty - 2 * cross + quad
    sst = m.yty - m.sy ** 2 / n
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sst > 0, 1.0 - sse / sst, np.nan)


def predict(probe, X):
    """[n,d] -> [n,k] fp64 predictions."""
    return ((np.asarray(X, np.float64) - probe["mu"]) / probe["sigma"]) @ probe["W"] + probe["ybar"]


# --------------------------------------------------------------------------- #
# condition runner: forward the cached docs through a model, accumulate, solve
# --------------------------------------------------------------------------- #
class RowBuffer:
    """Capped raw-row store (heldout ACTIVE rows) for argmax accuracy + a direct
    R^2 crosscheck of the moment algebra. fp16 features to bound memory."""

    def __init__(self, cap=30000, d=D_MODEL, k=R):
        self.cap = cap
        self.X, self.Y = [], []
        self.n = 0

    def add(self, X, Y):
        take = min(self.cap - self.n, X.shape[0])
        if take <= 0:
            return
        self.X.append(X[:take].half().cpu().numpy())
        self.Y.append(Y[:take].cpu().numpy())
        self.n += take

    def arrays(self):
        if not self.X:
            return np.zeros((0, D_MODEL), np.float16), np.zeros((0, R), np.float32)
        return np.concatenate(self.X), np.concatenate(self.Y)


def run_condition(model, docs, bos_id, *, gate_scale, use_acts, device,
                  forward_metrics, heldout_every=10, max_rows=32,
                  max_tokens=65536, lambda_grid=LAMBDA_GRID, log=print):
    """Forward all docs once at this gate setting, accumulate probe moments,
    solve the ridge grid, pick lambda by heldout R^2 (mean over channels, per
    population). docs: {ids, acts, doc_idx}; doc_idx % heldout_every ==
    heldout_every-1 -> test split. use_acts=False => vanilla forward (baseline,
    no site). Returns a result dict (json+npz-able)."""
    cap = ResidCapture(model)
    mom = None                       # lazily sized from the first captured batch
    buf_te_act = RowBuffer()
    ce_sum, ce_n = 0.0, 0

    try:
        for ids, acts, mask, doc_idxs in pack_batches(docs, bos_id, max_rows, max_tokens):
            fm = forward_metrics(model, ids.to(device),
                                 acts=(acts if use_acts else None),
                                 gate_scale=gate_scale, return_logits=False)
            X = cap.pop()                                  # [B,T,d] fp32 on device
            if mom is None:
                d, k = X.shape[-1], acts.shape[-1]
                mom = {(p, s): Moments(d, k) for p in ("all", "act") for s in ("tr", "te")}
            is_te = torch.tensor([(i % heldout_every) == heldout_every - 1
                                  for i in doc_idxs])
            act_mask = mask & (acts != 0).any(-1)
            acts_dev = acts.to(X.device)
            for split, smask in (("tr", mask & ~is_te[:, None]),
                                 ("te", mask & is_te[:, None])):
                sm = smask.to(X.device)
                mom[("all", split)].add(X[sm], acts_dev[sm])
                am = (act_mask & smask).to(X.device)
                mom[("act", split)].add(X[am], acts_dev[am])
                if split == "te":
                    buf_te_act.add(X[am], acts_dev[am])
            ptc = np.asarray(fm["per_token_ce"], np.float64)
            valid = mask.numpy() & ~np.isnan(ptc)
            ce_sum += float(ptc[valid].sum()); ce_n += int(valid.sum())
    finally:
        cap.close()

    out = {"gate_scale": float(gate_scale), "use_acts": bool(use_acts),
           "hooked_site": bool(cap.uses_site),
           "ce_mean": ce_sum / max(ce_n, 1), "n_ce_tokens": ce_n,
           "lambda_grid": list(lambda_grid), "pops": {}}
    Xte, Yte = buf_te_act.arrays()
    for pop in ("all", "act"):
        tr, te = mom[(pop, "tr")], mom[(pop, "te")]
        grid = []
        for lam in lambda_grid:
            probe = solve_ridge(tr, lam)
            r2 = r2_from_moments(probe, te)
            grid.append((lam, probe, r2))
        lam_best, probe, r2 = max(grid, key=lambda g: np.nanmean(g[2]))
        # Encoding-direction estimate: raw cross-covariance stream x target.
        # The ridge readout V is COVARIANCE-WHITENED (optimal decoder != signal
        # direction); Cxy[:, c] estimates where channel c's signal actually
        # LIVES in the stream — the sharp geometric comparison against D.
        cxy = tr.xty / tr.n - np.outer(tr.sx / tr.n, tr.sy / tr.n)
        rec = {"n_train": tr.n, "n_test": te.n, "lambda": lam_best,
               "Cxy": cxy,
               "r2_heldout": r2.tolist(),
               "r2_heldout_mean": float(np.nanmean(r2)),
               "r2_grid": {str(l): float(np.nanmean(r)) for l, _, r in grid},
               "W": probe["W"], "V": probe["V"], "mu": probe["mu"],
               "sigma": probe["sigma"], "ybar": probe["ybar"]}
        if Xte.shape[0] > 0:
            pred = predict(probe, Xte.astype(np.float32))
            rec["argmax_acc_te_act"] = float(
                (pred.argmax(1) == Yte.argmax(1)).mean())
            ssr = ((Yte - pred) ** 2).sum(0)
            sst = ((Yte - Yte.mean(0)) ** 2).sum(0)
            rec["r2_direct_te_act"] = (1 - ssr / np.maximum(sst, 1e-12)).tolist()
        out["pops"][pop] = rec
        log(f"    [{pop}] n_tr={tr.n} n_te={te.n} lam={lam_best:g} "
            f"R2(mean)={rec['r2_heldout_mean']:.4f}"
            + (f" argmax_acc={rec.get('argmax_acc_te_act', float('nan')):.4f}"
               if "argmax_acc_te_act" in rec else ""))
    return out


def save_condition(path_npz, result):
    """Split a run_condition result into an npz (arrays) + a small dict (json)."""
    arrays = {}
    js = {k: v for k, v in result.items() if k != "pops"}
    js["pops"] = {}
    array_keys = ("W", "V", "mu", "sigma", "ybar", "Cxy")
    for pop, rec in result["pops"].items():
        js["pops"][pop] = {k: v for k, v in rec.items() if k not in array_keys}
        for k in array_keys:
            arrays[f"{pop}_{k}"] = np.asarray(rec[k], np.float64)
    np.savez_compressed(path_npz, **arrays)
    return js
