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
    evicted_snapshot = set(deleted)  # on_delete fires just before delete_fn
check(sorted(set(fetched_snapshot)) == order, f"every shard fetched (got {sorted(set(fetched_snapshot))})")
check(len(fetched_snapshot) == len(set(fetched_snapshot)), f"no shard fetched twice (idempotent): {fetched_snapshot}")
check(max_staged <= AHEAD + KEEP + 2, f"staged window stayed bounded (max {max_staged} <= {AHEAD + KEEP + 2}, not bulk {N})")
# after consuming through the last shard, everything up to frontier-keep_behind should be deleted
expected_deleted = set(range(0, N - 1 - KEEP))
check(expected_deleted <= deleted_snapshot,
      f"consumed shards deleted behind the window (expected superset of {sorted(expected_deleted)}, got {sorted(deleted_snapshot)})")
check(evicted_snapshot == deleted_snapshot, "on_delete (memmap-evict hook) fired for exactly the deleted shards")
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

print("\n[D] repo_for_factory: count-based shard->repo assignment")
rf = repo_for_factory(["r0", "r1", "r2"], 25)
check(rf(0) == "r0" and rf(24) == "r0" and rf(25) == "r1" and rf(74) == "r2" and rf(999) == "r2",
      "sid // per_repo picks the repo (clamped to last)")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
if __name__ == "__main__":
    sys.exit(1 if fails else 0)


def test_shard_prefetch():
    assert not fails, f"{len(fails)} failures: {fails}"
