"""CPU test for the rolling shard prefetcher (nanochat.injection.prefetch) with a
FAKE fetcher (no network): the background window stages ahead of consumption,
deletes consumed shards behind (keep_behind), stays bounded (never bulk), evicts
via the on_delete hook, blocks + fires on_wait when the window is behind, and is a
no-op passthrough for a local dir. Standalone: `python tests/test_shard_prefetch.py`.
"""
import os
import sys
import threading
import time

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(TESTS, ".."))
sys.path.insert(0, REPO)

from nanochat.injection.prefetch import ShardPrefetcher, repo_for_factory  # noqa: E402

fails = []


def check(cond, msg):
    print(f"  {'OK ' if cond else 'FAIL'}  {msg}")
    if not cond:
        fails.append(msg)


print("\n[A] local passthrough (fetch_fn=None) is a no-op")
p0 = ShardPrefetcher([0, 1, 2], fetch_fn=None).start()
check(not p0.enabled and p0.ensure(1) is None, "no-op prefetcher: ensure() returns immediately, no thread")

print("\n[B] rolling window: stage ahead, delete behind, stay bounded, evict via hook")
N = 12
order = list(range(N))
lock = threading.Lock()
fetched, deleted, evicted = [], [], []


def fetch(sid):
    with lock:
        fetched.append(sid)
    time.sleep(0.01)                 # simulate download latency
    return f"/stage/{sid}"


def delete(sid):
    with lock:
        deleted.append(sid)


def on_delete(sid):
    with lock:
        evicted.append(sid)


AHEAD, KEEP = 2, 1
p = ShardPrefetcher(order, fetch, delete_fn=delete, ahead=AHEAD, keep_behind=KEEP,
                    on_delete=on_delete).start()
max_staged = 0
for sid in order:
    res = p.ensure(sid)
    check(res == f"/stage/{sid}", f"ensure({sid}) staged") if sid == 0 else None
    if res != f"/stage/{sid}":
        fails.append(f"ensure({sid}) returned {res!r}")
    time.sleep(0.03)                 # let the background worker catch up (stage-ahead + delete-behind)
    max_staged = max(max_staged, len(p.stats()["staged"]))
time.sleep(0.2)
p.stop()

with lock:
    fetched_snapshot = list(fetched)
    deleted_snapshot = set(deleted)
    evicted_snapshot = set(evicted)
check(sorted(set(fetched_snapshot)) == order, f"every shard fetched (got {sorted(set(fetched_snapshot))})")
check(len(fetched_snapshot) == len(set(fetched_snapshot)), f"no shard fetched twice (idempotent): {fetched_snapshot}")
check(max_staged <= AHEAD + KEEP + 2, f"staged window stayed bounded (max {max_staged} <= {AHEAD + KEEP + 2}, not bulk {N})")
# after consuming through the last shard, everything up to frontier-keep_behind should be deleted
expected_deleted = set(range(0, N - 1 - KEEP))
check(expected_deleted <= deleted_snapshot,
      f"consumed shards deleted behind the window (expected superset of {sorted(expected_deleted)}, got {sorted(deleted_snapshot)})")
# on_delete fires just before delete_fn, so deleted ⊆ evicted at any snapshot
check(deleted_snapshot and deleted_snapshot <= evicted_snapshot,
      f"on_delete (memmap-evict hook) fired before every delete (evicted={sorted(evicted_snapshot)})")
check(len(p.stats()["staged"]) <= KEEP + 2, f"final staged set is small ({p.stats()['staged']})")

print("\n[C] behind window -> ensure blocks + fires on_wait (starvation hook)")
waited = []
gate = threading.Event()


def slow_fetch(sid):
    gate.wait(2.0)                   # worker is stuck until we release the gate
    return f"/stage/{sid}"


def on_wait(sid):
    waited.append(sid)


p2 = ShardPrefetcher([0, 1, 2, 3], slow_fetch, ahead=2, keep_behind=1, on_wait=on_wait).start()
t0 = time.time()
th = threading.Thread(target=lambda: (time.sleep(0.05), gate.set()))
th.start()
res = p2.ensure(0)                   # not staged yet (worker gated) -> on_wait + inline fetch
th.join()
p2.stop()
check(res == "/stage/0" and 0 in waited, f"ensure blocked, fired on_wait, then returned (waited={waited})")
check(time.time() - t0 >= 0.04, "ensure actually waited on the gated fetch (not a spurious pass)")

print("\n[E] transient fetch failure: real error surfaces, worker survives and retries")
attempts = []


def flaky_fetch(sid):
    attempts.append(sid)
    if len(attempts) <= 2:
        raise IOError(f"transient network error on shard {sid}")
    return f"/stage/{sid}"


p3 = ShardPrefetcher([0, 1, 2, 3], flaky_fetch, ahead=2, keep_behind=1).start()
try:
    r0 = p3.ensure(0)                # first attempts fail: inline fetch must see the REAL error
    inline_err = None
except Exception as e:               # noqa: BLE001
    inline_err = e
    r0 = None
check(r0 == "/stage/0" or isinstance(inline_err, IOError),
      f"failed fetch raises the real error (not UnboundLocalError): {type(inline_err).__name__ if inline_err else r0}")
deadline = time.time() + 8.0
ok_rest = False
while time.time() < deadline:        # worker must survive the failures and stage the rest
    try:
        if p3.ensure(1) == "/stage/1" and p3.ensure(2) == "/stage/2":
            ok_rest = True
            break
    except IOError:                  # a remaining transient attempt — keep retrying
        pass
    time.sleep(0.05)
p3.stop()
check(ok_rest, f"worker survived transient fetch failures and kept staging (attempts={attempts})")

print("\n[D] repo_for_factory: count-based shard->repo assignment")
rf = repo_for_factory(["r0", "r1", "r2"], 25)
check(rf(0) == "r0" and rf(24) == "r0" and rf(25) == "r1" and rf(74) == "r2" and rf(999) == "r2",
      "sid // per_repo picks the repo (clamped to last)")

print("\n[F] multi-stream: concurrent fetches bounded by streams, ordering preserved, no dup")
STREAMS = 3
NF = 16
mlock = threading.Lock()
live = [0]
peak = [0]
mfetched = []


def mfetch(sid):
    with mlock:
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        mfetched.append(sid)
    time.sleep(0.02)                  # hold the stream so siblings overlap
    with mlock:
        live[0] -= 1
    return f"/stage/{sid}"


pm = ShardPrefetcher(list(range(NF)), mfetch, ahead=STREAMS + 2, keep_behind=1, streams=STREAMS).start()
order_ok = True
for sid in range(NF):
    if pm.ensure(sid) != f"/stage/{sid}":     # ensure returns THIS shard's staged path (order preserved)
        order_ok = False
    time.sleep(0.01)
time.sleep(0.1)
pm.stop()
check(order_ok, "ensure returned the correct staged path for every shard in consumption order")
# worker fetches are bounded by streams; ensure adds at most ONE inline catch-up
# fetch (single consumer thread) when the window is briefly behind at startup.
check(peak[0] <= STREAMS + 1, f"concurrent fetches bounded by streams+1 inline (peak {peak[0]} <= {STREAMS + 1})")
check(peak[0] >= 2, f"streams actually ran concurrently (peak {peak[0]} >= 2, not serialized)")
check(len(mfetched) == len(set(mfetched)), f"no shard fetched twice across streams ({sorted(mfetched)})")
check(set(range(NF)) <= set(mfetched) | set(range(pm.stats()['frontier'] + 1)),
      "all consumed shards were staged")
check(pm.stats()["fetches"] > 0 and pm.stats()["mean_fetch_seconds"] > 0,
      f"fetch timing recorded for bandwidth sizing ({pm.stats()['fetches']} fetches, "
      f"{pm.stats()['mean_fetch_seconds']}s mean)")

print("\n[G] set_streams grows the worker pool (auto-sizing after the prefill measurement)")
ps = ShardPrefetcher(list(range(8)), lambda s: f"/stage/{s}", ahead=4, keep_behind=1, streams=1).start()
before = ps.stats()["streams"]
ps.set_streams(4)
after = ps.stats()["streams"]
ps.ensure(0)
time.sleep(0.05)
ps.stop()
check(before == 1 and after == 4, f"set_streams raised the stream count {before} -> {after}")

print("\n[H] widening `ahead` on a live prefetcher is honored (auto-sizing grows the window)")
ph = ShardPrefetcher(list(range(10)), lambda s: (time.sleep(0.005), f"/stage/{s}")[1],
                     ahead=1, keep_behind=1, streams=1).start()
ph.ensure(0)
time.sleep(0.2)
narrow = max(ph.stats()["staged"], default=-1)
ph.ahead = 4                                      # what injection_train's auto-sizing does
ph.set_streams(2)
time.sleep(0.6)                                   # workers re-window (idle wait <= 0.5s)
wide = max(ph.stats()["staged"], default=-1)
ph.stop()
check(narrow <= 1, f"ahead=1 window stayed at frontier+1 before widening (max staged {narrow})")
check(wide >= 3, f"raising .ahead mid-run widened the staged window (max staged {wide} >= 3)")

print("\n[I] a BaseException in a worker releases the reservation (ensure never hangs on it)")
batt = []
block = threading.Lock()


def base_exc_fetch(sid):
    with block:
        batt.append(sid)
        if sid == 1 and batt.count(1) == 1:
            raise SystemExit("worker killed mid-fetch")   # NOT an Exception subclass
    return f"/stage/{sid}"


pb = ShardPrefetcher([0, 1, 2, 3], base_exc_fetch, ahead=2, keep_behind=1, streams=1).start()
check(pb.ensure(0) == "/stage/0", "shard 0 staged normally")
deadline = time.time() + 5.0
while time.time() < deadline and 1 not in batt:   # wait for the worker to attempt (and die on) shard 1
    time.sleep(0.01)
res_box = []
t_ens = threading.Thread(target=lambda: res_box.append(pb.ensure(1)), daemon=True)
t_ens.start()
t_ens.join(timeout=5.0)
pb.stop()
check(not t_ens.is_alive() and res_box == ["/stage/1"],
      f"ensure(1) recovered inline after the worker died (reservation released, got {res_box})")

print("\n[J] cross-rank delete coordination: min-frontier files gate deletion; local"
      "\n    memmap eviction still fires on the OWN frontier; stale/foreign files ignored")
import tempfile  # noqa: E402

with tempfile.TemporaryDirectory() as coord:
    def _mk_fetch(tag):
        def _fetch(sid):
            with open(os.path.join(coord, f"f_{sid}"), "w") as fh:
                fh.write(tag)              # idempotent rewrite, like hf re-download
            return coord
        return _fetch

    def _delete(sid):
        try:
            os.remove(os.path.join(coord, f"f_{sid}"))
        except OSError:                    # double-delete across ranks: benign no-op
            pass

    def _exists(sid):
        return os.path.exists(os.path.join(coord, f"f_{sid}"))

    # stale files from a previous crashed run: OWN ranks (overwritten to -1 at
    # init) and a FOREIGN rank 7 (world_size=2 -> must be ignored by the min).
    for r, v in ((0, "999"), (1, "999"), (7, "999")):
        with open(os.path.join(coord, f".frontier_r{r}"), "w") as fh:
            fh.write(v)

    evictedA, evictedB = [], []
    ORDER = list(range(6))
    pA = ShardPrefetcher(ORDER, _mk_fetch("A"), delete_fn=_delete, ahead=2, keep_behind=1,
                         on_delete=evictedA.append, streams=1,
                         coord_dir=coord, rank=0, world_size=2)
    pB = ShardPrefetcher(ORDER, _mk_fetch("B"), delete_fn=_delete, ahead=2, keep_behind=1,
                         on_delete=evictedB.append, streams=1,
                         coord_dir=coord, rank=1, world_size=2)
    with open(os.path.join(coord, ".frontier_r0")) as fh:
        r0_init = fh.read().strip()
    with open(os.path.join(coord, ".frontier_r1")) as fh:
        r1_init = fh.read().strip()
    check(r0_init == "-1" and r1_init == "-1",
          f"init resets OWN stale frontier files to -1 (r0={r0_init}, r1={r1_init})")
    pA.start(); pB.start()

    for sid in range(5):                   # rank 0 races ahead: frontier 4
        pA.ensure(sid)
    pB.ensure(0)                           # rank 1 lags: frontier 0
    time.sleep(1.2)                        # workers re-window (idle wait <= 0.5s)
    check(_exists(0) and _exists(1),
          "lagging rank pins shards: nothing behind the GLOBAL min frontier deleted "
          f"(f_0={_exists(0)}, f_1={_exists(1)}; foreign .frontier_r7=999 ignored)")
    check(set(evictedA) >= {0, 1, 2},
          f"fast rank's OWN memmap eviction fired on its LOCAL frontier (evictedA={sorted(set(evictedA))})")
    check(not evictedB,
          f"lagging rank evicted nothing yet (evictedB={sorted(set(evictedB))})")

    for sid in range(1, 5):                # rank 1 catches up: global min -> 4
        pB.ensure(sid)
    deadline = time.time() + 4.0
    while time.time() < deadline and (_exists(0) or _exists(1) or _exists(2)):
        time.sleep(0.05)
    gone = not (_exists(0) or _exists(1) or _exists(2))
    check(gone, "once every rank passes a shard (min frontier), its files are deleted "
          f"(f_0={_exists(0)}, f_1={_exists(1)}, f_2={_exists(2)})")
    check(_exists(3) or _exists(4), "shards within keep_behind of the frontier survive")
    pA.stop(); pB.stop()

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
if __name__ == "__main__":
    sys.exit(1 if fails else 0)


def test_shard_prefetch():
    assert not fails, f"{len(fails)} failures: {fails}"
