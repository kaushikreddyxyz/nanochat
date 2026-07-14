"""Rolling shard prefetcher (Amendment 3): download score/parquet shards in
background threads WHILE training runs, keeping a small window ahead and deleting
consumed shards behind so disk stays minimal — never a bulk pre-download, never a
hard block at a shard boundary (the window absorbs download latency; if the next
shard still isn't ready the consumer blocks and the Amendment-2 starvation
machinery fires via ``on_wait``).

Consumption is monotonic (the ride-along loader enumerates shards in corpus
order), so the prefetcher tracks a single frontier position. ``streams`` worker
threads stage shards ``ahead`` beyond the frontier and delete shards more than
``keep_behind`` positions behind it; ``ensure(sid)`` blocks until ``sid`` is
staged (falling back to an inline fetch if the workers are behind) and advances
the frontier. Fetches are idempotent and race-free — a shard is reserved in
``_inflight`` under the lock before its fetch starts, so sibling streams and the
inline ``ensure`` fallback never double-download it. An 8.7GB shard covers ~50s
of full-speed training but takes ~45-210s to fetch depending on NIC, so multiple
streams are what keep the window ahead (size with ``buffering.size_prefetch_streams``).

Local dirs = no-op passthrough (``fetch_fn=None``): ``ensure`` returns
immediately, no threads, no deletes. The real HF fetcher must set
``HF_HUB_DISABLE_XET=1`` (xet stalls on pods). CPU-tested with a fake fetcher.

Multi-rank (DDP) staging: every rank runs its own prefetcher over the SAME
staging dir, but ranks' producers skew by a shard or two — a rank-local delete
frontier lets a fast rank remove files a slow rank is still reading
(FileNotFoundError / partial-read races, observed at 8xH100 prefill). With
``coord_dir``/``rank``/``world_size`` set, each rank persists its frontier to a
tiny ``.frontier_r{rank}`` file (atomic ``os.replace``) and deletion keys off
the MINIMUM frontier across all ranks; a missing/garbled frontier file reads as
-1 and conservatively blocks deletion. Fetches were already cross-process safe
(hf_hub_download file locks); only deletion needed coordination.
"""
import os
import threading
import time

SCORE_FILES = ("scores_{sid:05d}.npy", "docs_{sid:05d}.jsonl")


def repo_for_factory(repos, per_repo):
    """(count-based shard->repo assignment) sid -> repos[sid // per_repo]."""
    repos = list(repos)
    return lambda sid: repos[min(int(sid) // int(per_repo), len(repos) - 1)]


def make_hf_score_fetcher(staging_dir, repo_for, *, files=SCORE_FILES,
                          climbmix_dir=None, climbmix_repo="karpathy/climbmix-400b-shuffle",
                          climbmix_file="shard_{sid:05d}.parquet"):
    """fetch_fn(sid): download the per-shard score files (+ optional ClimbMix
    parquet) into ``staging_dir`` with HF_HUB_DISABLE_XET=1 (xet stalls on pods)."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.makedirs(staging_dir, exist_ok=True)
    if climbmix_dir:
        os.makedirs(climbmix_dir, exist_ok=True)

    def _download_verified(repo, name, dest_dir):
        """hf_hub_download that GUARANTEES the destination file exists on return.
        A crashed run can leave staging with a stale .cache metadata entry whose
        destination file was deleted; combined with 8 ranks' concurrent
        downloads of the same file, hf_hub_download has been observed to return
        without materializing the destination (rank saw ensure() succeed, then
        FileNotFoundError on np.load). Verify + force a real re-download."""
        from huggingface_hub import hf_hub_download
        dest = os.path.join(dest_dir, name)
        hf_hub_download(repo, name, repo_type="dataset", local_dir=dest_dir)
        if not os.path.exists(dest):
            hf_hub_download(repo, name, repo_type="dataset", local_dir=dest_dir,
                            force_download=True)
        if not os.path.exists(dest):
            raise FileNotFoundError(
                f"hf_hub_download returned without materializing {dest} "
                f"(stale staging .cache metadata?)")
        return dest

    def fetch(sid):
        for f in files:
            _download_verified(repo_for(sid), f.format(sid=sid), staging_dir)
        if climbmix_dir:
            _download_verified(climbmix_repo, climbmix_file.format(sid=sid), climbmix_dir)
        return staging_dir

    return fetch


def make_local_deleter(staging_dir, *, files=SCORE_FILES, climbmix_dir=None,
                       climbmix_file="shard_{sid:05d}.parquet"):
    """delete_fn(sid): remove the staged files for ``sid`` (rolling cleanup)."""
    def delete(sid):
        names = [os.path.join(staging_dir, f.format(sid=sid)) for f in files]
        if climbmix_dir:
            names.append(os.path.join(climbmix_dir, climbmix_file.format(sid=sid)))
        for p in names:
            try:
                os.remove(p)
            except OSError:
                pass

    return delete


class ShardPrefetcher:
    def __init__(self, shard_ids, fetch_fn, *, delete_fn=None, ahead=2, keep_behind=1,
                 on_wait=None, on_delete=None, streams=1,
                 coord_dir=None, rank=0, world_size=1):
        self.order = list(shard_ids)
        self._pos = {sid: i for i, sid in enumerate(self.order)}
        self.fetch_fn = fetch_fn
        self.delete_fn = delete_fn
        self.ahead = int(ahead)
        self.keep_behind = int(keep_behind)
        self.on_wait = on_wait
        self.on_delete = on_delete
        self.streams = max(1, int(streams))
        self._staged = {}          # sid -> fetch_fn result
        self._inflight = set()
        self._frontier = -1        # highest consumed position
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._threads = []
        self._stop = False
        self._waits = 0
        self._wait_seconds = 0.0
        self._deleted = 0
        self._fetch_count = 0      # completed fetches (bandwidth numerator)
        self._fetch_seconds = 0.0
        self._evicted_local = set()   # sids whose LOCAL memmap eviction already fired
        # -- cross-rank delete coordination (shared staging dir; see module doc) --
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._coord_dir = coord_dir if (coord_dir and self._world_size > 1
                                        and self.enabled) else None
        self._wlock = threading.Lock()
        self._written_frontier = None
        if self._coord_dir:
            os.makedirs(self._coord_dir, exist_ok=True)
            self._write_frontier(-1)   # reset OUR file (stale values from a
            # previous crashed run would otherwise unblock deletion early;
            # callers should barrier across ranks between __init__ and start())

    # -- frontier files: .frontier_r{rank} in the shared staging dir ----------
    def _frontier_path(self, r):
        return os.path.join(self._coord_dir, f".frontier_r{r}")

    def _write_frontier(self, val):
        tmp = self._frontier_path(self._rank) + f".tmp{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(str(int(val)))
        os.replace(tmp, self._frontier_path(self._rank))   # atomic
        self._written_frontier = int(val)

    def _publish_frontier(self):
        """Persist our frontier if it advanced (monotonic; safe from any thread)."""
        if not self._coord_dir:
            return
        with self._wlock:
            with self._cv:
                fr = self._frontier
            if self._written_frontier is not None and fr <= self._written_frontier:
                return
            self._write_frontier(fr)

    def _global_frontier(self):
        """MIN of every rank's persisted frontier. Missing/garbled file -> -1
        (that rank hasn't reported yet: conservatively block deletion)."""
        lo = None
        for r in range(self._world_size):
            try:
                with open(self._frontier_path(r)) as f:
                    v = int(f.read().strip() or "-1")
            except (OSError, ValueError):
                v = -1
            lo = v if lo is None else min(lo, v)
        return -1 if lo is None else lo

    @property
    def enabled(self):
        return self.fetch_fn is not None

    def start(self):
        if self.enabled:
            self._spawn(self.streams)
        return self

    def _spawn(self, target):
        """Bring the live worker count up to ``target`` (idempotent; only grows —
        auto-sizing raises streams after the prefill bandwidth measurement)."""
        with self._cv:
            self.streams = max(self.streams, int(target))
            need = self.streams - len(self._threads)
            base = len(self._threads)
            for k in range(max(need, 0)):
                th = threading.Thread(target=self._run, name=f"shard-prefetch-{base + k}", daemon=True)
                self._threads.append(th)
                th.start()

    def set_streams(self, target):
        if self.enabled:
            self._spawn(target)

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    # -- idempotent, race-free fetch (worker and ensure share this) --
    def _fetch(self, sid):
        with self._cv:
            while sid in self._inflight:      # someone else is fetching it
                self._cv.wait()
            if sid in self._staged:
                return self._staged[sid]
            self._inflight.add(sid)
        try:
            res = self.fetch_fn(sid)
        except BaseException:                 # failed fetch: never stage, re-raise as-is
            with self._cv:
                self._inflight.discard(sid)
                self._cv.notify_all()
            raise
        with self._cv:
            self._inflight.discard(sid)
            if not self._stop:
                self._staged[sid] = res
            self._cv.notify_all()
        return res

    def ensure(self, sid):
        """Block until ``sid`` is staged; advance the frontier. No-op passthrough
        (local dir) returns immediately. Blocks + fires ``on_wait`` only if the
        background window has not reached ``sid`` yet."""
        if not self.enabled:
            return None
        _MISS = object()
        with self._cv:
            self._frontier = max(self._frontier, self._pos.get(sid, self._frontier))
            self._cv.notify_all()          # let the worker re-window / delete behind
            hit = self._staged.get(sid, _MISS)
        self._publish_frontier()           # file IO outside the cv lock
        if hit is not _MISS:
            return hit
        if self.on_wait is not None:       # window did not cover this shard: starvation
            self.on_wait(sid)
        t0 = time.time()
        res = self._fetch(sid)             # inline fallback so the consumer never deadlocks
        with self._cv:
            self._waits += 1
            self._wait_seconds += time.time() - t0
        return res

    def _run(self):
        # One stream worker. Reserves the nearest un-staged in-window shard under
        # the lock (so sibling streams pick DIFFERENT shards, never the same one),
        # fetches it outside the lock, then re-windows.
        while True:
            # Deletion frontier: with cross-rank coordination this is the MIN of
            # all ranks' persisted frontiers (never delete what any rank still
            # needs); read OUTSIDE the cv lock (small file IO). min() with the
            # local frontier below stays conservative if our own persist lags.
            gf = self._global_frontier() if self._coord_dir else None
            with self._cv:
                while not self._stop and self._frontier < 0:
                    self._cv.wait()
                if self._stop:
                    return
                frontier = self._frontier
                del_frontier = frontier if gf is None else min(gf, frontier)
                window = [self.order[i] for i in range(max(frontier, 0),
                                                       min(frontier + self.ahead + 1, len(self.order)))]
                todo = [s for s in window if s not in self._staged and s not in self._inflight]
                # File deletion keys off del_frontier (global min across ranks);
                # the rank's OWN memmap-cache eviction (on_delete) keys off its
                # LOCAL frontier — a fast rank must not pin passed shards' mmaps
                # while it waits for the slowest rank to move on.
                drop = [s for s in list(self._staged)
                        if self._pos.get(s, 0) < del_frontier - self.keep_behind and s not in self._inflight]
                evict_local = [s for s in list(self._staged)
                               if self._pos.get(s, 0) < frontier - self.keep_behind
                               and s not in self._evicted_local and s not in self._inflight]
                self._evicted_local.update(evict_local)
                sid = todo[0] if todo else None
                if sid is not None:
                    self._inflight.add(sid)    # reserve before releasing the lock
            if self.on_delete is not None:
                for d in evict_local:
                    self.on_delete(d)          # idempotent (pop with default); may re-fire in _evict
            for d in drop:
                self._evict(d)
            if sid is None:
                with self._cv:
                    if not self._stop and self._frontier == frontier:
                        self._cv.wait(timeout=0.5)
                continue
            t0 = time.time()
            try:
                res = self.fetch_fn(sid)       # nearest-ahead first, then the loop re-windows
            except BaseException as e:         # transient (network): drop the reservation, back off;
                with self._cv:                 # a persistent failure surfaces via ensure's inline fetch
                    self._inflight.discard(sid)
                    self._cv.notify_all()
                    if not self._stop and isinstance(e, Exception):
                        self._cv.wait(timeout=2.0)
                if not isinstance(e, Exception):
                    raise                      # non-Exception kills the worker, but never holds the reservation (ensure would hang on it)
                continue
            with self._cv:
                self._inflight.discard(sid)
                if not self._stop:
                    self._staged[sid] = res
                    self._fetch_count += 1
                    self._fetch_seconds += time.time() - t0
                self._cv.notify_all()

    def _evict(self, sid):
        with self._cv:
            if sid not in self._staged or sid in self._inflight:
                return
            self._staged.pop(sid, None)
            self._evicted_local.discard(sid)   # bookkeeping stays bounded
        if self.on_delete is not None:
            self.on_delete(sid)
        if self.delete_fn is not None:
            self.delete_fn(sid)   # concurrent ranks double-delete: deleter must
            #                       treat a missing file as a no-op (OSError pass)
        with self._cv:
            self._deleted += 1

    def stats(self):
        with self._cv:
            mean_fetch = (self._fetch_seconds / self._fetch_count) if self._fetch_count else 0.0
            return {"staged": sorted(self._staged), "frontier": self._frontier,
                    "deleted": self._deleted, "waits": self._waits,
                    "wait_seconds": round(self._wait_seconds, 3),
                    "streams": self.streams, "fetches": self._fetch_count,
                    "mean_fetch_seconds": round(mean_fetch, 3)}
