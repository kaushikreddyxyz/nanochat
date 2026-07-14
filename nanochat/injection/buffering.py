"""Video-player buffering primitives (pure, no threads / torch / DDP): the small
decision functions the ride-along loader and injection_train share so they stay
unit-testable in isolation.

- ``BufferState``: the prefill/running/rebuffer hysteresis state machine over a
  token-denominated buffer depth. dry < rebuffer <= prefill (hysteresis: refill
  past the dry mark all the way to rebuffer before resuming, so it can't thrash).
- ``coordinate_rebuffer``: DDP all-or-none pause decision (any rank low => all
  rebuffer), factored out so the collective logic tests without real DDP.
- ``duty_cycle_forecast`` / ``rebuffer_progress``: the forecast and progress/ETA
  line math.
- ``size_prefetch_streams``: pick download streams + window from the shard
  download time vs the time one shard lasts in training.

The threaded buffer (``BufferControl``) and the packing loop live in
``activation_dataloader.py``; this module is the logic they defer to.
"""
import math

PREFILL, RUNNING, REBUFFER = "prefill", "running", "rebuffer"


class BufferState:
    """Hysteresis over a buffer depth in tokens. Starts in PREFILL (consumer
    blocked) until depth reaches ``prefill``; RUNNING until depth falls below
    ``dry`` (buffer ran dry); then REBUFFER (consumer blocked) until depth climbs
    back to ``rebuffer`` — deliberately past ``dry`` so a source hovering at the
    dry mark does not flip every step."""

    def __init__(self, prefill, rebuffer, dry):
        if not (0 <= dry < rebuffer <= prefill):
            raise ValueError(f"need 0 <= dry < rebuffer <= prefill, got "
                             f"dry={dry} rebuffer={rebuffer} prefill={prefill}")
        self.prefill, self.rebuffer, self.dry = int(prefill), int(rebuffer), int(dry)
        self.phase = PREFILL

    def observe(self, depth):
        """Fold one depth reading into the phase and return it."""
        if self.phase == PREFILL:
            if depth >= self.prefill:
                self.phase = RUNNING
        elif self.phase == RUNNING:
            if depth < self.dry:
                self.phase = REBUFFER
        elif self.phase == REBUFFER:
            if depth >= self.rebuffer:
                self.phase = RUNNING
        return self.phase

    @property
    def paused(self):
        """Consumer must not draw a batch while prefilling or rebuffering."""
        return self.phase != RUNNING


def coordinate_rebuffer(local_low, world_size, allreduce_max):
    """All-or-none rebuffer across DDP ranks: if ANY rank's buffer is low, every
    rank rebuffers (independent per-rank stalls amplify at allreduce, so pause
    together). ``allreduce_max(int) -> int`` mirrors dist.all_reduce(MAX); it is
    only called when world_size > 1 (single-rank short-circuits to the local
    flag), which keeps this pure-testable with a simulated collective."""
    if world_size <= 1:
        return bool(local_low)
    return allreduce_max(1 if local_low else 0) > 0


def duty_cycle_forecast(production_tok_s, consumption_tok_s):
    """(ratio, duty_cycle, one-line message). duty_cycle = min(1, prod/cons): if
    the source out-produces consumption the GPU never waits (~100%); below 1 the
    GPU idles the shortfall. Message matches the format injection_train prints
    after prefill."""
    cons = max(float(consumption_tok_s), 1e-9)
    ratio = float(production_tok_s) / cons
    duty = min(1.0, ratio)
    msg = (f"activation production ~{ratio:.1f}x consumption — "
           f"forecast duty cycle ~{duty * 100:.0f}%")
    if ratio < 1.0:
        msg += ", consider more --lookup-workers / download streams"
    elif ratio < 1.5:
        # production is pre-packing, consumption post-packing: best-fit cropping
        # discards ~35% of produced tokens, so <1.5x actually starves.
        msg += " (MARGINAL: best-fit packing crops ~35% of produced tokens)"
    return ratio, duty, msg


def rebuffer_progress(depth_tokens, target_tokens, tok_per_s, activity=None):
    """(percent, eta_seconds, one-line message) for a buffering/rebuffer bar.
    ``depth_tokens`` is the current buffer depth toward ``target_tokens``; ETA from
    the measured production rate (None if unknown). ``activity`` is the current
    work string (e.g. "downloading scores_00071.npy / aligning")."""
    target = max(int(target_tokens), 1)
    frac = min(max(depth_tokens / target, 0.0), 1.0)
    remaining = max(target - depth_tokens, 0)
    eta = (remaining / tok_per_s) if tok_per_s and tok_per_s > 0 else None
    eta_str = f"~{eta:.0f}s" if eta is not None else "~?s"
    tail = f" — {activity}" if activity else ""
    msg = f"buffering {frac * 100:.0f}% ({eta_str}){tail}"
    return frac * 100.0, eta, msg


def shard_cover_seconds(tokens_per_shard, per_rank_consumption_tok_s, world_size):
    """Wall seconds one score shard lasts in training. DDP ranks stride row
    groups WITHIN each shard (``_document_batches``), so every rank crosses
    shard boundaries at the GLOBAL consumption pace — and every rank downloads
    every shard: cover = tokens_per_shard / (per_rank_consumption * world)."""
    world_cons = float(per_rank_consumption_tok_s) * max(int(world_size), 1)
    return float(tokens_per_shard) / max(world_cons, 1e-9)


def size_prefetch_streams(shard_bytes, download_bytes_per_s, cover_seconds,
                          *, min_streams=1, max_streams=8, min_ahead=2):
    """Pick (streams, ahead, message) for the rolling prefetcher from the shard
    download time vs the time one shard lasts in training. One shard takes
    ``t_dl = shard_bytes / download_bytes_per_s`` to fetch and covers
    ``cover_seconds`` of training; we need ~ceil(t_dl / cover) streams to fetch
    the next shard(s) in time, and a window one deeper. Clamped to
    [min_streams, max_streams]; the message logs the arithmetic."""
    bw = max(float(download_bytes_per_s), 1.0)
    cover = max(float(cover_seconds), 1e-9)
    t_dl = float(shard_bytes) / bw
    need = t_dl / cover
    streams = min(max(int(math.ceil(need)), min_streams), max_streams)
    ahead = max(min_ahead, int(math.ceil(need)) + 1)
    msg = (f"prefetch sizing: {shard_bytes / 1e9:.1f}GB / {bw / 1e6:.0f}MB/s = "
           f"{t_dl:.0f}s download vs {cover:.0f}s/shard training "
           f"(~{need:.1f}x) => {streams} streams, ahead={ahead}")
    return streams, ahead, msg
