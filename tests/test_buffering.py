"""CPU tests for the video-player buffering layer (nanochat.injection.buffering
pure logic + nanochat.injection.activation_dataloader.BufferControl threading),
no network / GPU / real tokenizer:

  [A] BufferState hysteresis: PREFILL -> RUNNING -> REBUFFER -> RUNNING, and it
      stays paused until the buffer climbs past dry all the way to rebuffer.
  [B] coordinate_rebuffer: DDP all-or-none across 1-3 simulated ranks incl. the
      only-one-rank-low case (no real DDP).
  [C] duty_cycle_forecast: ratio -> duty-cycle line, with the low-throughput hint.
  [D] rebuffer_progress: percent + ETA + message math (incl. unknown rate).
  [E] size_prefetch_streams: streams/window from download-vs-cover arithmetic.
  [F] BufferControl: prefill fills to target BEFORE the first pull; a dry buffer
      is detected and do_rebuffer refills to rebuffer_tokens before resuming.
  [G] the buffered loader packs byte-identically to the synchronous loader.

Standalone: `python tests/test_buffering.py`.
"""
import os
import sys
import threading
import time

import numpy as np

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(TESTS, ".."))
sys.path.insert(0, REPO)

from nanochat.injection.buffering import (  # noqa: E402
    BufferState, PREFILL, RUNNING, REBUFFER,
    coordinate_rebuffer, duty_cycle_forecast, rebuffer_progress, size_prefetch_streams,
)
from nanochat.injection.activation_dataloader import BufferControl, _Chunk  # noqa: E402

fails = []


def check(cond, msg):
    print(f"  {'OK ' if cond else 'FAIL'}  {msg}")
    if not cond:
        fails.append(msg)


# --------------------------------------------------------------------------- #
print("\n[A] BufferState hysteresis")
st = BufferState(prefill=100, rebuffer=50, dry=20)
check(st.phase == PREFILL and st.paused, "starts in PREFILL, paused")
check(st.observe(40) == PREFILL, "still prefilling below prefill target")
check(st.observe(100) == RUNNING and not st.paused, "reaching prefill -> RUNNING (consume)")
check(st.observe(60) == RUNNING, "draining above dry stays RUNNING")
check(st.observe(19) == REBUFFER and st.paused, "falling below dry -> REBUFFER (pause)")
check(st.observe(30) == REBUFFER and st.paused, "past dry but below rebuffer: STILL paused (hysteresis)")
check(st.observe(50) == RUNNING and not st.paused, "reaching rebuffer -> RUNNING (resume)")
try:
    BufferState(prefill=10, rebuffer=20, dry=5)          # rebuffer > prefill
    check(False, "invalid ordering must raise")
except ValueError:
    check(True, "rejects rebuffer > prefill (needs dry < rebuffer <= prefill)")


# --------------------------------------------------------------------------- #
print("\n[B] coordinate_rebuffer (DDP all-or-none, simulated ranks)")


def sim_allreduce(flags):
    """all_reduce(MAX): every rank contributes and receives max over all ranks."""
    return lambda _x: max(flags)


for flags in ([1, 0, 0], [0, 1, 0], [1, 1, 1]):
    decisions = [coordinate_rebuffer(bool(flags[r]), len(flags), sim_allreduce(flags))
                 for r in range(len(flags))]
    check(all(decisions), f"any rank low -> ALL rebuffer together (flags={flags} -> {decisions})")
check(not any(coordinate_rebuffer(False, 3, sim_allreduce([0, 0, 0])) for _ in range(3)),
      "no rank low -> nobody rebuffers")
check(coordinate_rebuffer(True, 1, lambda x: (_ for _ in ()).throw(AssertionError("no collective"))),
      "world_size=1 short-circuits to the local flag (no collective call)")
check(not coordinate_rebuffer(False, 1, lambda x: 99), "world_size=1, not low -> no rebuffer")


# --------------------------------------------------------------------------- #
print("\n[C] duty_cycle_forecast")
ratio, duty, msg = duty_cycle_forecast(2400, 1000)
check(abs(ratio - 2.4) < 1e-6 and duty == 1.0, f"prod>cons -> duty capped at 100% ({msg})")
check("~2.4x" in msg and "~100%" in msg and "consider" not in msg, f"healthy forecast line: {msg}")
ratio, duty, msg = duty_cycle_forecast(400, 1000)
check(abs(ratio - 0.4) < 1e-6 and abs(duty - 0.4) < 1e-6, f"prod<cons -> duty ~= ratio ({msg})")
check("~0.4x" in msg and "~40%" in msg and "consider more" in msg, f"starving forecast has the hint: {msg}")


# --------------------------------------------------------------------------- #
print("\n[D] rebuffer_progress (percent + ETA)")
pct, eta, msg = rebuffer_progress(42, 100, 58.0 / 35.0, activity="downloading scores_00071.npy / aligning")
check(abs(pct - 42.0) < 1e-6, f"percent = depth/target ({pct})")
check(eta is not None and abs(eta - 35.0) < 0.5, f"ETA = remaining/rate ({eta:.1f}s)")
check(msg.startswith("buffering 42%") and "~35s" in msg and "scores_00071.npy" in msg,
      f"message format matches the spec ({msg})")
pct, eta, msg = rebuffer_progress(10, 100, 0.0)
check(eta is None and "~?s" in msg, f"unknown rate -> ETA unknown ({msg})")
pct, eta, msg = rebuffer_progress(500, 100, 10.0)
check(pct == 100.0 and eta == 0.0, "depth past target clamps to 100% / 0s")


# --------------------------------------------------------------------------- #
print("\n[E] size_prefetch_streams (download vs cover arithmetic)")
streams, ahead, msg = size_prefetch_streams(8.7e9, 100e6, 50.0)     # 87s dl / 50s cover ~1.7x
check(streams == 2 and ahead == 3, f"100MB/s NIC -> 2 streams, ahead 3 ({msg})")
streams, ahead, _ = size_prefetch_streams(8.7e9, 45e6, 50.0)        # 193s dl / 50s ~3.9x
check(streams == 4 and ahead == 5, f"slow 45MB/s NIC -> 4 streams, ahead 5 (got {streams},{ahead})")
streams, ahead, _ = size_prefetch_streams(8.7e9, 1e9, 50.0)         # 8.7s dl / 50s ~0.17x
check(streams == 1 and ahead == 2, f"fast NIC -> single stream, min window (got {streams},{ahead})")
streams, ahead, _ = size_prefetch_streams(8.7e9, 10e6, 50.0, max_streams=6)  # 870s dl -> clamp
check(streams == 6, f"needy sizing clamps to max_streams (got {streams})")


# --------------------------------------------------------------------------- #
print("\n[F] BufferControl: prefill to target, dry detection, coordinated rebuffer")


class GatedProducer:
    """Fake _ChunkProducer: emits fixed-token chunks; a cleared gate stalls it
    (simulates a source falling behind so the buffer can be drained dry)."""

    def __init__(self, chunk_tokens, stats):
        self.chunk_tokens = chunk_tokens
        self.stats = stats
        self.gate = threading.Event()
        self.gate.set()

    def next_chunk(self):
        self.gate.wait()
        if self.stats is not None:
            self.stats["produced_tokens"] = self.stats.get("produced_tokens", 0) + self.chunk_tokens
            self.stats["produce_seconds"] = self.stats.get("produce_seconds", 0.0) + 1e-4
        docs = [(np.zeros(self.chunk_tokens + 1, np.int64), {})]   # BOS + body
        return _Chunk(docs, (0, 0, 1), self.chunk_tokens)


stats = {}
prod = GatedProducer(chunk_tokens=10, stats=stats)
ctrl = BufferControl(prod, prefill_tokens=100, rebuffer_tokens=50, dry_tokens=20,
                     max_tokens=200, stats=stats).start()
ctrl.wait_prefill()
check(ctrl.buffer_tokens() >= 100, f"prefill filled to >= prefill_tokens BEFORE first pull ({ctrl.buffer_tokens()}t)")
check(ctrl.produce_rate() > 0, f"production rate measured during prefill ({ctrl.produce_rate():,.0f} tok/s)")

prod.gate.clear()                       # source stalls: buffer can now only drain
time.sleep(0.05)                        # let any in-flight chunk settle
drained = 0
while not ctrl.buffer_low() and drained < 10000:
    ctrl.pull()
    drained += 1
check(ctrl.buffer_low() and ctrl.buffer_tokens() < 20, f"buffer went dry after draining ({ctrl.buffer_tokens()}t < 20)")

prod.gate.set()                         # source recovers
ctrl.do_rebuffer()
check(ctrl.buffer_tokens() >= 50, f"do_rebuffer refilled to >= rebuffer_tokens before resuming ({ctrl.buffer_tokens()}t)")
check(stats.get("buffer_tokens") == ctrl.buffer_tokens(), "stats.buffer_tokens tracks the live depth")
ctrl.stop()


# --------------------------------------------------------------------------- #
print("\n[G] buffered loader packs byte-identically to the synchronous loader")
import nanochat.injection.activation_dataloader as _adl  # noqa: E402


class FakeTok:
    def get_bos_token_id(self):
        return 0

    def encode(self, texts, prepend=None, num_threads=8):
        out = []
        for t in texts:
            n = 3 + (int(t.split("-")[1]) % 6)             # deterministic per-doc length
            ids = [int(t.split("-")[1]) * 100 + j + 1 for j in range(n)]
            out.append(([prepend] + ids) if prepend is not None else ids)
        return out


class FakeSrc:
    r = 2
    name = "s"
    positional = False

    def lookup(self, text, n):
        v = float(int(text.split("-")[1]) % 7)
        return np.full((n, self.r), v, np.float32), int(int(text.split("-")[1]))

    def add_noise(self, z, key):
        return z


N_CORPUS, TBS = 40, 4


def fake_document_batches(split, resume, tbs):
    def gen():
        epoch = 1
        while True:
            for s in range(0, N_CORPUS, tbs):
                yield [f"doc-{i}" for i in range(s, min(s + tbs, N_CORPUS))], (0, 0, epoch)
            epoch += 1
    return gen()


_orig_db = _adl._document_batches
_adl._document_batches = fake_document_batches
try:
    B, T = 2, 16
    sync = _adl.acts_data_loader_with_state(FakeTok(), {"s": FakeSrc()}, B, T, split="train",
                                            device="cpu", buffer_size=8, tokenizer_batch_size=TBS)
    ctrl2, buffered = _adl.acts_data_loader_buffered(
        FakeTok(), {"s": FakeSrc()}, B, T, split="train", device="cpu", buffer_size=8,
        tokenizer_batch_size=TBS, prefill_tokens=200, rebuffer_tokens=100, dry_tokens=32, max_tokens=600)
    ctrl2.wait_prefill()
    identical = True
    for _ in range(20):
        xa, ya, aa, sa = next(sync)
        xb, yb, ab, sb = next(buffered)
        identical = identical and np.array_equal(xa.numpy(), xb.numpy()) \
            and np.array_equal(ya.numpy(), yb.numpy()) \
            and np.array_equal(aa["s"].numpy(), ab["s"].numpy()) and sa == sb
    check(identical, "buffered loader yields identical tokens/acts/state as the synchronous loader")
    ctrl2.stop()
finally:
    _adl._document_batches = _orig_db


print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
if __name__ == "__main__":
    sys.exit(1 if fails else 0)


def test_buffering():
    assert not fails, f"{len(fails)} failures: {fails}"
