"""Injection health metrics: catch the SILENT failures of a multi-hour injected run.

The characteristic failure mode of this stack is a run that completes looking fine
while the injection quietly stopped mattering — a channel goes dead, tokenizer drift
zeroes whole documents, alignment starts pooling differently and the realized dose
drifts away from what was calibrated at startup. Loss barely moves for any of these.
So we measure, every ``--injection-log-every`` steps, what the site actually did.

Cost model (why this is affordable):

* Everything is a reduction over tensors already in memory — the ``(B, T, r)`` acts
  the site consumed and the site's own parameters. No extra forward pass.
* Realized loudness needs NO ``x``. The site injects ``rms(x) * (u @ D_hat)``, so
  ``rms(injected)/rms(x) == rms(u @ D_hat)`` exactly — the ratio the calibration
  targets is independent of the residual stream. And ``rms(u @ D_hat)^2 ==
  (u G u^T)/n_embd`` with ``G = D_hat D_hat^T``, so the per-token loudness costs
  ``O(B*T*r^2)`` instead of materializing a ``(B*T, n_embd)`` projection. The same
  ``G`` is the Gram matrix Tier 2 wants, computed once and used twice.
* Accumulation is local and lock-free; the only collective is ONE ``all_reduce`` of a
  single packed vector, at log time. No per-step collective, no ``.item()`` in the
  hot path (nothing forces a CPU-GPU sync between log steps).

A metrics bug must never kill a multi-hour job: every entry point is guarded, and the
first failure disables collection for the rest of the run with one loud warning.
"""
import math

import numpy as np
import torch

# Loudness is logged relative to the calibration target, so the histogram lives in
# log2(realized/target): +-4 covers a 16x miss either way at ~9% resolution, which is
# far finer than the 25% the startup check calls a deviation.
HIST_LO, HIST_HI, HIST_BINS = -4.0, 4.0, 64


def saturation_bounds(src, r):
    """(lo, hi) per-channel z-score values of the int8 quantization ceiling, or None
    when the source is not a quantized probe-score store. An activation sitting AT the
    ceiling was clipped upstream: its true score was larger and the dose is understated.
    Derived from the store's own frozen constants rather than assumed to be 4 sigma."""
    try:
        zero, scale = np.asarray(src._zero, np.float64), np.asarray(src._scale, np.float64)
        mean, std = np.asarray(src._mean, np.float64), np.asarray(src._std, np.float64)
    except AttributeError:
        return None
    if not all(v.shape == (r,) for v in (zero, scale, mean, std)) or not np.all(std > 0):
        return None
    a = (127.0 * scale + zero - mean) / std
    b = (-128.0 * scale + zero - mean) / std
    return np.minimum(a, b), np.maximum(a, b)


class _SiteAcc:
    """Per-site device-side accumulators for one log window. float64 because a window
    is tens of millions of tokens — float32 counts stop being exact past 2^24."""

    def __init__(self, r, device):
        z = lambda n: torch.zeros(n, dtype=torch.float64, device=device)
        self.scalars = z(8)      # tok, rowfire, zero_rows, cofire, loud_sum, nonfinite_a, nonfinite_l, sat
        self.hist = z(HIST_BINS + 2)   # log2(loudness/target), with under/overflow bins
        self.fire = z(r)         # per-channel firing count
        self.aeff = z(r)         # per-channel sum of relu(a - z0)
        self.gcs = z(r)          # per-channel sum of the channel_scale gradient
        self.loss = z(4)         # fired_sum, fired_n, unfired_sum, unfired_n
        self.grad = z(3)         # n_grad_steps, sum |grad D|, sum |grad threshold|
        self.src = z(4)          # host-side source counters (ok, miss_shard, miss_row, drift)

    def parts(self):
        return (self.scalars, self.hist, self.fire, self.aeff, self.gcs,
                self.loss, self.grad, self.src)

    def zero_(self):
        for t in self.parts():
            t.zero_()


class InjectionMetrics:
    """Accumulate injection health over a window of steps; emit wandb keys at flush.

    ``sites``   : name -> InjectionSite (the live modules; parameters are read fresh).
    ``sources`` : name -> ActivationSource, for alignment bookkeeping (optional).
    ``targets`` : name -> calibrated target median injected loudness.
    ``reduce_fn``: ``f(1-D float64 tensor) -> summed-across-ranks tensor``. None = single
    process. Injected rather than calling ``dist`` directly so the DDP path is testable.
    """

    def __init__(self, sites, sources=None, targets=None, concepts=None, device=None,
                 log_every=50, loss_split=False, reduce_fn=None, log=print):
        self.sites = dict(sites)
        self.sources = dict(sources or {})
        self.targets = dict(targets or {})
        self.log_every = int(log_every)
        self.loss_split = bool(loss_split)
        self.reduce_fn = reduce_fn
        self.log = log
        # `active` is fixed at construction and drives WHEN the collective runs, so every
        # rank enters flush() on the same steps. `enabled` is runtime health and may go
        # False on one rank only — if that also skipped the all_reduce the other ranks
        # would hang, which is strictly worse than the metrics bug it guards against.
        self.active = self.log_every > 0 and bool(self.sites)
        self.enabled = self.active
        self.device = device or next(iter(self.sites.values())).direction.device
        self.acc, self.names, self.init_dhat, self.sat, self._src_prev = {}, {}, {}, {}, {}
        for name, site in self.sites.items():
            r = int(site.cfg.r)
            self.acc[name] = _SiteAcc(r, self.device)
            supplied = (concepts or {}).get(name) or getattr(self.sources.get(name), "concepts", None)
            self.names[name] = [str(c).replace("/", "_") for c in (supplied or [f"ch{i}" for i in range(r)])]
            if len(self.names[name]) != r:
                self.names[name] = [f"ch{i}" for i in range(r)]
            with torch.no_grad():
                d = site.direction.detach().float()
                self.init_dhat[name] = (d / _rms_rows(d)).clone()
            b = saturation_bounds(self.sources.get(name), r)
            self.sat[name] = None if b is None else (
                torch.as_tensor(b[0], dtype=torch.float32, device=self.device),
                torch.as_tensor(b[1], dtype=torch.float32, device=self.device))
            self._src_prev[name] = self._src_counts(name)

    # -- introspection for the startup banner -------------------------------- #
    def enabled_metrics(self):
        """Human-readable list of what will actually be logged, given what the sites and
        sources expose. Printed at startup so an operator knows what to expect in wandb."""
        if not self.enabled:
            return []
        m = ["realized loudness (p50/mean vs target)", "per-channel firing rate + mean a_eff|firing",
             "dead-channel count", "row firing rate + co-fire width", "exact-zero row rate",
             "D row RMS / Gram off-diagonal / cosine drift from init", "gradient norms (direction, loudness)",
             "non-finite counts"]
        if any(hasattr(s, "stats") for s in self.sources.values()):
            m.append("alignment ok/drift/miss rates")
        if any(hasattr(s, "pool_depth_stats") for s in self.sources.values()):
            m.append("pooling depth p50/p90/p99 (cumulative)")
        if any(v is not None for v in self.sat.values()):
            m.append("int8 saturation rate")
        m.append("loss split injected-vs-not" if self.loss_split else "loss split OFF (--injection-loss-split)")
        return m

    def should_log(self, step):
        """Deliberately independent of ``enabled``: flush() carries a DDP collective."""
        return self.active and step % self.log_every == 0

    # -- accumulation -------------------------------------------------------- #
    def observe(self, acts, per_token_loss=None, valid=None):
        """One micro-batch. ``acts``: name -> (B, T, r). ``per_token_loss``: flat
        (B*T,) unreduced loss aligned to the acts' INPUT positions; ``valid``: flat
        bool mask of non-ignored targets. Both None unless --injection-loss-split."""
        if not self.enabled:
            return
        try:
            self._observe(acts, per_token_loss, valid)
        except Exception as e:  # noqa: BLE001 — a metrics bug must not kill the run
            self._disable("activation accumulation", e)

    def observe_grads(self):
        """Call AFTER backward and BEFORE optimizer.step()/zero_grad, so every gradient
        read is still rank-local (the optimizer all-reduces in step) and this accumulator's
        cross-rank mean is uniformly correct."""
        if not self.enabled:
            return
        try:
            self._observe_grads()
        except Exception as e:  # noqa: BLE001
            self._disable("gradient accumulation", e)

    @torch.no_grad()
    def _observe(self, acts, per_token_loss, valid):
        for name, site in self.sites.items():
            a = acts.get(name) if hasattr(acts, "get") else None
            if a is None:
                continue
            acc = self.acc[name]
            r = int(site.cfg.r)
            a2 = a.detach().reshape(-1, r).float()
            n = a2.shape[0]
            finite = torch.isfinite(a2)
            nonfinite = (~finite).sum()
            # An EXACT-zero row is the fingerprint of a failed lookup (tokenizer drift,
            # missing doc) or a nanochat token no gemma token covered — the loader maps
            # both to exact zeros with no noise. Counted on the tensor the model actually
            # saw, so it is independent of the source's own bookkeeping. Measured BEFORE
            # scrubbing: a non-finite row is a different alarm, not a failed lookup.
            zero_rows = (~(a2 != 0).any(1)).sum()
            a2 = torch.where(finite, a2, torch.zeros((), device=a2.device))

            thr = site.threshold.detach().float()
            cs = site.channel_scale.detach().float()
            aeff = (a2 - thr).clamp_min(0)
            firing = aeff > 0
            fire_c = firing.sum(0)
            n_on = firing.sum(1)                       # channels live per token
            rowfire = n_on > 0
            n_rowfire = rowfire.sum()
            d = site.direction.detach().float()
            d_hat = d / _rms_rows(d)
            g = d_hat @ d_hat.t()
            u = aeff * cs
            n_embd = d.shape[1]
            # rms(u @ D_hat)^2 == (u G u^T)/n_embd: the realized loudness as a fraction of
            # rms(x), without materializing the (B*T, n_embd) projection.
            loud = ((u @ g) * u).sum(1).clamp_min(0).div(n_embd).sqrt()
            loud_finite = torch.isfinite(loud)
            loud = torch.where(loud_finite, loud, torch.zeros((), device=loud.device))
            fired = loud > 0

            target = float(self.targets.get(name) or 0.0)
            ref = target if target > 0 else 1.0
            lr = torch.log2(loud[fired].clamp_min(1e-12) / ref)
            edges = torch.linspace(HIST_LO, HIST_HI, HIST_BINS + 1, device=loud.device, dtype=lr.dtype)
            idx = torch.bucketize(lr, edges)
            acc.hist.scatter_add_(0, idx.to(torch.int64),
                                  torch.ones_like(idx, dtype=torch.float64))

            sat = torch.zeros((), device=a2.device)
            if self.sat[name] is not None:
                lo, hi = self.sat[name]
                sat = ((a2 >= hi) | (a2 <= lo)).sum()

            f64 = lambda t: torch.as_tensor(t, device=a2.device).to(torch.float64)
            acc.scalars += torch.stack([f64(float(n)), f64(n_rowfire), f64(zero_rows),
                                        f64(n_on.sum()), f64(loud.sum()), f64(nonfinite),
                                        f64((~loud_finite).sum()), f64(sat)])
            acc.fire += fire_c.to(torch.float64)
            acc.aeff += aeff.sum(0).to(torch.float64)

            if per_token_loss is not None and valid is not None:
                l = per_token_loss.detach().reshape(-1).float()
                v = valid.reshape(-1)
                f, nf = rowfire & v, (~rowfire) & v
                acc.loss += torch.stack([(l * f).sum(), f.sum(), (l * nf).sum(), nf.sum()]).to(torch.float64)

    @torch.no_grad()
    def _observe_grads(self):
        for name, site in self.sites.items():
            acc = self.acc[name]
            dg = site.direction.grad
            cg = site.channel_scale.grad
            tg = site.threshold.grad
            acc.grad += torch.stack([
                torch.ones((), device=self.device, dtype=torch.float64),
                (dg.detach().float().norm() if dg is not None else torch.zeros((), device=self.device)).to(torch.float64),
                (tg.detach().float().abs().mean() if tg is not None else torch.zeros((), device=self.device)).to(torch.float64)])
            if cg is not None:
                acc.gcs += cg.detach().float().to(torch.float64)

    # -- flush --------------------------------------------------------------- #
    def flush(self, step):
        """One all_reduce, then wandb keys. Resets the window. Returns {} on failure or
        when nothing was observed — but the collective still runs, so a rank that lost
        its metrics never desynchronizes the ones that still have theirs."""
        if not self.active:
            return {}
        try:
            packed, parts = self._pack()
        except Exception as e:  # noqa: BLE001
            self._disable("metric packing", e)
            packed, parts = self._empty_packet()
        if self.reduce_fn is not None:
            packed = self.reduce_fn(packed)
        try:
            out = {} if not self.enabled else self._unpack(packed, parts)
        except Exception as e:  # noqa: BLE001
            self._disable("flush", e)
            out = {}
        for acc in self.acc.values():
            acc.zero_()
        return out

    def _empty_packet(self):
        parts = [t for n in self.sites for t in self.acc[n].parts()]
        return torch.zeros(sum(t.numel() for t in parts), dtype=torch.float64, device=self.device), parts

    def _pack(self):
        for name in self.sites:
            cur = self._src_counts(name)
            if cur is not None:
                prev = self._src_prev[name] or [0.0] * 4
                self.acc[name].src += torch.as_tensor([c - p for c, p in zip(cur, prev)],
                                                      dtype=torch.float64, device=self.device)
                self._src_prev[name] = cur
        parts = [t for n in self.sites for t in self.acc[n].parts()]
        return torch.cat([t.reshape(-1) for t in parts]), parts

    def _unpack(self, packed, parts):
        order = list(self.sites)
        vals, off = [], 0
        for t in parts:
            vals.append(packed[off:off + t.numel()])
            off += t.numel()
        out = {}
        per_site = len(self.acc[order[0]].parts())
        for i, name in enumerate(order):
            out.update(self._site_keys(name, vals[i * per_site:(i + 1) * per_site]))
        return out

    def _site_keys(self, name, v):
        scalars, hist, fire, aeff, gcs, loss, grad, src = (x.detach().cpu().numpy() for x in v)
        p = f"inj/{name}/"
        tok, rowfire, zero_rows, cofire, loud_sum, nf_a, nf_l, sat = (float(x) for x in scalars)
        names = self.names[name]
        r = len(names)
        out = {}
        if tok <= 0:
            return out
        target = float(self.targets.get(name) or 0.0)

        # -- Tier 1: is the injection still doing what it was calibrated to do? --
        n_fired = float(hist.sum())
        if n_fired > 0:
            mean_loud = loud_sum / n_fired
            p50 = _hist_p50(hist, target if target > 0 else 1.0)
            out[p + "loudness_mean"] = mean_loud
            out[p + "loudness_p50"] = p50
            if target > 0:
                out[p + "loudness_ratio_p50"] = p50 / target      # 1.0 == as calibrated
                out[p + "loudness_ratio_mean"] = mean_loud / target
        if target > 0:
            out[p + "loudness_target"] = target
        out[p + "row_fire_rate"] = rowfire / tok
        out[p + "zero_row_rate"] = zero_rows / tok
        out[p + "cofire_mean"] = cofire / rowfire if rowfire > 0 else 0.0
        rates = fire / tok
        for i, c in enumerate(names):
            out[p + "fire_rate/" + c] = float(rates[i])
            out[p + "aeff_mean/" + c] = float(aeff[i] / fire[i]) if fire[i] > 0 else 0.0
        out[p + "fire_rate_min"] = float(rates.min())
        out[p + "fire_rate_max"] = float(rates.max())
        # The alarm to set a wandb alert on: a channel that fired at calibration and
        # stopped is invisible in p50 loudness (which medians across channels).
        out[p + "dead_channels"] = float((fire == 0).sum())

        ok, miss_shard, miss_row, drift = (float(x) for x in src)
        looked = ok + miss_shard + miss_row + drift
        if looked > 0:
            out[p + "align_ok_rate"] = ok / looked
            out[p + "align_drift_rate"] = drift / looked      # tokenizer drift -> exact zeros
            out[p + "align_miss_rate"] = (miss_shard + miss_row) / looked
            out[p + "align_docs"] = looked
        fired_sum, fired_n, unfired_sum, unfired_n = (float(x) for x in loss)
        if fired_n > 0 and unfired_n > 0:
            lf, lu = fired_sum / fired_n, unfired_sum / unfired_n
            out[p + "loss_injected"] = lf
            out[p + "loss_uninjected"] = lu
            out[p + "loss_gap"] = lf - lu     # <0: the model is exploiting the signal

        # -- Tier 2: geometry (the trainable-direction loopholes) --
        site = self.sites[name]
        with torch.no_grad():
            d = site.direction.detach().float()
            row_rms = _rms_rows(d).reshape(-1)
            d_hat = d / row_rms.reshape(-1, 1).clamp_min(1e-4)
            # /n_embd so the Gram reads as cosines (unit-RMS rows have squared L2 norm
            # n_embd, not 1): diagonal 1, off-diagonal 1 == two rows fully parallel.
            g = (d_hat @ d_hat.t() / d.shape[1]).cpu().numpy()
            cos = torch.nn.functional.cosine_similarity(d_hat, self.init_dhat[name], dim=1).cpu().numpy()
        rr = row_rms.cpu().numpy()
        out[p + "d_row_rms_min"] = float(rr.min())
        out[p + "d_row_rms_max"] = float(rr.max())
        out[p + "d_row_rms_mean"] = float(rr.mean())
        if r > 1:
            offdiag = np.abs(g[~np.eye(r, dtype=bool)])
            # Rows rotating parallel inflate co-firing loudness and collapse concept
            # separation while leaving every per-row gauge invariant untouched.
            out[p + "gram_offdiag_absmax"] = float(offdiag.max())
            out[p + "gram_offdiag_absmean"] = float(offdiag.mean())
        out[p + "dir_cos_init_min"] = float(cos.min())
        out[p + "dir_cos_init_mean"] = float(cos.mean())

        n_gsteps, dnorm, tabs = (float(x) for x in grad)
        if n_gsteps > 0:
            out[p + "grad_norm_direction"] = dnorm / n_gsteps
            out[p + "grad_threshold_absmean"] = tabs / n_gsteps
            gm = gcs / n_gsteps
            for c, gv in zip(names, gm):
                out[p + "grad_loudness/" + c] = float(gv)
            out[p + "grad_loudness_mean"] = float(gm.mean())
            out[p + "grad_loudness_absmax"] = float(np.abs(gm).max())
            # The loudness params are never stepped, but autograd still assigns them a
            # gradient. dL/d(scale) < 0 means a louder injection would lower the loss:
            # sign-flipped and projected onto the scale vector, positive == "wants louder".
            w = np.asarray(site.channel_scale.detach().float().cpu().numpy(), np.float64)
            nrm = float(np.linalg.norm(w))
            out[p + "want_louder"] = float(-(gm * w).sum() / nrm) if nrm > 0 else 0.0

        # -- Tier 3: context --
        out[p + "nonfinite_acts"] = nf_a
        out[p + "nonfinite_injected"] = nf_l
        if self.sat[name] is not None:
            out[p + "saturation_rate"] = sat / (tok * r)
        depth = self._pool_depth(name)
        if depth:
            for q in ("p50", "p90", "p99"):
                out[p + "pool_depth_" + q] = float(depth[q])
            out[p + "pool_depth_mean"] = float(depth["mean"])
            out[p + "no_coverage_rate"] = float(depth["zero_coverage_rate"])
        return out

    # -- helpers ------------------------------------------------------------- #
    def _src_counts(self, name):
        src = self.sources.get(name)
        try:
            s = src.stats()
        except Exception:  # noqa: BLE001 — sources without bookkeeping simply contribute nothing
            return None
        return [float(s.get(k, 0)) for k in ("ok", "miss_shard", "miss_row", "drift")]

    def _pool_depth(self, name):
        """Cumulative (whole-run) pooling depth from the source's own histogram — it is a
        property of the tokenizer pair, not of the window, and reusing it costs nothing."""
        src = self.sources.get(name)
        try:
            return src.pool_depth_stats()
        except Exception:  # noqa: BLE001
            return None

    def _disable(self, where, exc):
        self.enabled = False
        self.log("!" * 80)
        self.log(f"[inj-metrics] DISABLED for the rest of the run: {where} raised "
                 f"{type(exc).__name__}: {exc}. Training continues WITHOUT injection health metrics.")
        self.log("!" * 80)


def _rms_rows(d):
    """The site's own row RMS, clamp and all — metrics must measure the same D_hat the
    forward builds, not a differently-regularized one."""
    return d.pow(2).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()


def _hist_p50(hist, ref):
    """Median of the log2(loudness/ref) histogram, back in loudness units. Bin 0 is the
    underflow (< ref/16) and the last bin the overflow; both saturate at their edge."""
    total = hist.sum()
    if total <= 0:
        return 0.0
    k = int(np.searchsorted(np.cumsum(hist), total / 2.0))
    k = min(k, HIST_BINS + 1)
    width = (HIST_HI - HIST_LO) / HIST_BINS
    if k == 0:
        return float(ref * 2.0 ** HIST_LO)
    if k == HIST_BINS + 1:
        return float(ref * 2.0 ** HIST_HI)
    return float(ref * 2.0 ** (HIST_LO + (k - 0.5) * width))


def default_reduce_fn(dist, device):
    """all_reduce(SUM) of the packed metric vector — the only collective this module adds,
    and only on log steps."""
    def _reduce(t):
        t = t.to(device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t
    return _reduce
