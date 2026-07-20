"""CPU tests for the runtime probe-score sources (nanochat.injection.sources)
and their positional ride-along join — nothing precomputed offline. Synthesizes
a fake climbmix-scored shard triplet (columns/quant/corpus_stats json,
scores_<sid>.npy, docs_<sid>.jsonl) plus fake nanochat/gemma tokenizers, and
drives the REAL code path:

  [A] RuntimeProbeScoreSource.lookup_by_row alignment: dequant+standardize +
      prefix align gemma->nanochat, with hand-checkable multi-gemma->one-nano and
      one-gemma->multi-nano cases, unmapped->exact-zero, and drift/miss->None.
  [B] positional join == content-hash join (the (shard,row) keying is exact).
  [C] the ride-along loader's (shard,row) cursor drives lookup_by_row correctly.
  [D] LiveProbeScoreSource with a stub score_fn; gate='auto' unsupported.
  [E] single-worker alignment throughput (docs/s), honestly extrapolated.

Standalone: `python tests/test_runtime_probe_source.py`.
"""
import json
import os
import re
import sys
import tempfile
import time
import types

import numpy as np

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(TESTS, ".."))
sys.path.insert(0, REPO)

from nanochat.injection.sources import (RuntimeProbeScoreSource, LiveProbeScoreSource,  # noqa: E402
                                        doc_hash)
from nanochat.injection.align import gemma_to_qwen_map, nanochat_char_offsets  # noqa: E402

fails = []


def check(cond, msg):
    print(f"  {'OK ' if cond else 'FAIL'}  {msg}")
    if not cond:
        fails.append(msg)


# --------------------------------------------------------------------------- #
# fake tokenizers (fully controllable spans)
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"\S+|\s+")


class FakeNanoEnc:
    """Word-granularity byte tokenizer with an id->bytes map (so
    nanochat_char_offsets can reconstruct spans from per-token byte lengths)."""

    def __init__(self):
        self._map = {}

    def encode_ordinary(self, text):
        ids = []
        for k, m in enumerate(_WORD.finditer(text)):
            t = m.group(0).encode("utf-8")
            i = (int.from_bytes(t[:4].ljust(4, b"\0"), "big") + 7919 * k) % 60000 + 1
            while i in self._map and self._map[i] != t:
                i = (i + 1) % 60000 + 1
            self._map[i] = t
            ids.append(i)
        return ids

    def decode_single_token_bytes(self, i):
        return self._map[i]


def char_gemma_encode(text):
    """Char-granularity gemma stand-in that skips whitespace (so leading spaces
    have no anchor -> unmapped nanochat tokens)."""
    ids, offs = [], []
    for j, ch in enumerate(text):
        if ch.isspace():
            continue
        ids.append((ord(ch) % 240) + 1)
        offs.append((j, j + 1))
    return ids, offs


# --------------------------------------------------------------------------- #
# synthesize a store (K=4 concepts, layers [6,8,14]); scale=1/zero=0/mean=0/std=1
# so standardized z == the stored int8 value, per gemma row.
# --------------------------------------------------------------------------- #
K = 4
LAYERS = [6, 8, 14]
CONCEPTS = ["a", "b", "c", "d"]
DOCS = ["ab cd", " x", "hello world foo bar", "single", "one two three four five"]

STORE = tempfile.mkdtemp(prefix="runtime_probe_store_")
PARQ = tempfile.mkdtemp(prefix="runtime_probe_parquet_")
json.dump({"concepts": CONCEPTS, "layers": LAYERS, "families": {c: "fam" for c in CONCEPTS}},
          open(os.path.join(STORE, "columns.json"), "w"))
json.dump({"zero": [[0.0] * K for _ in LAYERS], "scale": [[1.0] * K for _ in LAYERS]},
          open(os.path.join(STORE, "quant.json"), "w"))
json.dump({"mean": [[0.0] * K for _ in LAYERS], "std": [[1.0] * K for _ in LAYERS]},
          open(os.path.join(STORE, "corpus_stats.json"), "w"))

_gemma_counts = [len(char_gemma_encode(t)[0]) for t in DOCS]
_total = sum(_gemma_counts)
scores = np.zeros((_total, 3, K), np.int8)         # axis1: 0=L6,1=L8,2=L14
for gi in range(_total):
    scores[gi, 1, :] = (gi % 60) + 1               # L8 row value encodes the global gemma index
np.save(os.path.join(STORE, "scores_00000.npy"), scores)
_off = 0
with open(os.path.join(STORE, "docs_00000.jsonl"), "w") as f:
    for di, (t, n) in enumerate(zip(DOCS, _gemma_counts)):
        f.write(json.dumps({"doc": di, "start": _off, "n": n}) + "\n")
        _off += n

enc = FakeNanoEnc()
# Amendment 1: MEAN over covering gemma tokens is the DEFAULT; 'last' is opt-in.
SRC = RuntimeProbeScoreSource(STORE, shards=[0], layer=8, nano_enc=enc,
                              gemma_encode=char_gemma_encode, noise_sigma=0.0)
SRC_LAST = RuntimeProbeScoreSource(STORE, shards=[0], layer=8, nano_enc=FakeNanoEnc(),
                                   gemma_encode=char_gemma_encode, noise_sigma=0.0,
                                   align_policy="last")
check(SRC.align_policy == "mean", "default align_policy is 'mean'")


def _expected(text, policy="mean"):
    """Reference: each nano token's covering set = gemma tokens whose CHAR SPAN
    OVERLAPS it; 'mean' averages them, 'last' takes the rightmost; no overlap -> 0."""
    nano_ids = enc.encode_ordinary(text)
    nano_off = np.asarray(nanochat_char_offsets(enc, nano_ids, text), np.int64)
    g_ids, g_off = char_gemma_encode(text)
    gstart = sum(_gemma_counts[:DOCS.index(text)])
    zg = scores[gstart:gstart + len(g_ids), 1, :].astype(np.float32)  # (n_gemma, K)
    g_off = np.asarray(g_off, np.int64).reshape(-1, 2)
    out = np.zeros((len(nano_ids), K), np.float32)
    for i, (ns, ne) in enumerate(nano_off):
        if ne <= ns:
            continue
        cov = [gi for gi, (gs, ge) in enumerate(g_off) if ge > gs and gs < ne and ge > ns]
        if not cov:
            continue
        out[i] = zg[cov[-1]] if policy == "last" else zg[cov].mean(0)
    return out, len(nano_ids)


# --------------------------------------------------------------------------- #
print("\n[A] lookup_by_row overlap alignment + standardization (mean default + last)")
for di, text in enumerate(DOCS):
    exp_m, n_nano = _expected(text, "mean")
    got_m, key = SRC.lookup_by_row(0, di, text, n_nano)
    check(got_m is not None and np.allclose(got_m, exp_m, atol=1e-6), f"doc {di!r} MEAN rows match reference")
    exp_l, _ = _expected(text, "last")
    got_l, _ = SRC_LAST.lookup_by_row(0, di, text, n_nano)
    check(got_l is not None and np.array_equal(got_l, exp_l), f"doc {di!r} LAST rows match reference")
    check(key == int(doc_hash(text)), f"doc {di!r} noise key == content hash")

# hand-checked: "ab cd" nano ["ab"," ","cd"] -> gemma chars a,b,c,d (z=1,2,3,4).
# overlap sets: "ab"@(0,2) covers {a,b}; " "@(2,3) covers NONE (whitespace char, gemma
# dropped it) -> zero; "cd"@(3,5) covers {c,d}.
gm, _ = SRC.lookup_by_row(0, 0, "ab cd", 3)
check(np.allclose(gm, np.array([[1.5] * K, [0] * K, [3.5] * K], np.float32)),
      "MEAN: 'ab'->mean(a,b)=1.5, ' '->0 (no overlap), 'cd'->mean(c,d)=3.5")
gl, _ = SRC_LAST.lookup_by_row(0, 0, "ab cd", 3)
check(np.array_equal(gl, np.array([[2] * K, [0] * K, [4] * K], np.float32)),
      "LAST: 'ab'->b=2, ' '->0, 'cd'->d=4 (rightmost overlapping gemma)")
# " x" nano [" ","x"]; leading space overlaps NO gemma -> exact zero row (both policies)
gx, _ = SRC.lookup_by_row(0, 1, " x", 2)
check(np.array_equal(gx[0], np.zeros(K, np.float32)) and not np.array_equal(gx[1], np.zeros(K, np.float32)),
      "unmapped nanochat token (no overlapping gemma) -> EXACT zero row")

# nested broadcast: ONE big gemma token spanning 3 nano tokens -> all three inherit it
# (both policies). Uses a dedicated single-token gemma stand-in.
def one_gemma_encode(text):
    return [1], [(0, len(text))]           # the whole doc is one gemma token
class WordNano2:                            # 3 nano tokens over "XYZABC": "XY","ZA","BC"
    _map = {1: b"XY", 2: b"ZA", 3: b"BC"}
    def encode_ordinary(self, t): return [1, 2, 3]
    def decode_single_token_bytes(self, i): return self._map[i]
STORE1 = tempfile.mkdtemp(prefix="runtime_probe_store1_")
json.dump({"concepts": CONCEPTS, "layers": LAYERS, "families": {c: "fam" for c in CONCEPTS}},
          open(os.path.join(STORE1, "columns.json"), "w"))
json.dump({"zero": [[0.0] * K for _ in LAYERS], "scale": [[1.0] * K for _ in LAYERS]},
          open(os.path.join(STORE1, "quant.json"), "w"))
json.dump({"mean": [[0.0] * K for _ in LAYERS], "std": [[1.0] * K for _ in LAYERS]},
          open(os.path.join(STORE1, "corpus_stats.json"), "w"))
sc1 = np.zeros((1, 3, K), np.int8); sc1[0, 1, :] = 7   # single gemma token, z=7
np.save(os.path.join(STORE1, "scores_00000.npy"), sc1)
open(os.path.join(STORE1, "docs_00000.jsonl"), "w").write(json.dumps({"doc": 0, "start": 0, "n": 1}) + "\n")
for pol in ("mean", "last"):
    s1 = RuntimeProbeScoreSource(STORE1, shards=[0], layer=8, nano_enc=WordNano2(),
                                 gemma_encode=one_gemma_encode, noise_sigma=0.0, align_policy=pol)
    g1, _ = s1.lookup_by_row(0, 0, "XYZABC", 3)
    check(np.array_equal(g1, np.full((3, K), 7, np.float32)),
          f"nested: one gemma -> 3 nano tokens all get its score ({pol})")

# drift / miss contracts
check(SRC.lookup_by_row(0, 0, "ab cd", 99)[0] is None, "nano token-count drift -> None")
check(SRC.lookup_by_row(0, 999, "ab cd", 3)[0] is None, "row out of range -> None")
check(SRC.lookup_by_row(7, 0, "ab cd", 3)[0] is None, "shard not configured -> None")
st = SRC.stats()
check(st["miss_row"] >= 1 and st["miss_shard"] >= 1 and st["drift"] >= 1, f"fallbacks counted: {st}")


# --------------------------------------------------------------------------- #
print("\n[B] positional join == content-hash join")
SRC_HASH = RuntimeProbeScoreSource(STORE, shards=[0], layer=8, nano_enc=FakeNanoEnc(),
                                   gemma_encode=char_gemma_encode, noise_sigma=0.0,
                                   climbmix_dir=None)
# build the hash index directly from our known docs (avoids needing a parquet walk here)
SRC_HASH._index = {int(doc_hash(t)): (0, sum(_gemma_counts[:i]), _gemma_counts[i])
                   for i, t in enumerate(DOCS)}
same = True
for di, text in enumerate(DOCS):
    _, n_nano = _expected(text)
    a = SRC.lookup_by_row(0, di, text, n_nano)[0]
    b = SRC_HASH.lookup(text, n_nano)[0]
    same = same and np.array_equal(a, b)
check(same, "row i (positional) and hash(parquet row i) yield identical activations")


# --------------------------------------------------------------------------- #
print("\n[C] ride-along loader (shard,row) cursor drives lookup_by_row")
import pyarrow as pa           # noqa: E402
import pyarrow.parquet as pq   # noqa: E402

N_CORPUS = 24
DOC_LENS = (np.random.default_rng(3).integers(3, 20, size=N_CORPUS)).tolist()
corpus_text = [f"doc-{i}-len-{DOC_LENS[i]}" for i in range(N_CORPUS)]
train_pq = os.path.join(PARQ, "shard_00000.parquet")
val_pq = os.path.join(PARQ, "shard_00001.parquet")
pq.write_table(pa.table({"text": corpus_text}), train_pq, row_group_size=10_000)  # 1 row group
pq.write_table(pa.table({"text": ["v"]}), val_pq)

BOS = 0


class StubPositionalSource:
    """Positional source whose activation for (shard,row) encodes the row itself
    (row+1 in every channel), so the loader's cursor is directly checkable."""
    positional = True
    r = 3
    name = "stub"

    def lookup_by_row(self, sid, row, text, n_tokens):
        return np.full((n_tokens, self.r), row + 1, np.float32), int(doc_hash(text))

    def add_noise(self, z, key):
        return z


class FakeTok:
    def get_bos_token_id(self):
        return BOS

    def encode(self, texts, prepend=None, append=None, num_threads=8):
        out = []
        for t in texts:
            i = int(t.split("-")[1])
            ids = [i * 100 + j + 1 for j in range(DOC_LENS[i])]  # token value -> (doc, pos)
            out.append(([prepend] + ids) if prepend is not None else ids)
        return out


def fake_document_batches(split, resume, tbs):
    # enumerate the train parquet in row order, 1 row group, chunked by tbs;
    # epoch increments per pass (mirrors _document_batches re-reading rg 0)
    def gen():
        epoch = 1
        while True:
            for s in range(0, N_CORPUS, tbs):
                yield corpus_text[s:s + tbs], (0, 0, epoch)
            epoch += 1
    return gen()


import nanochat.dataset as _ds                              # noqa: E402
import nanochat.injection.activation_dataloader as _adl     # noqa: E402
_ds.list_parquet_files = lambda *a, **k: [train_pq, val_pq]
_adl._document_batches = fake_document_batches
from nanochat.injection.activation_dataloader import acts_data_loader_with_state  # noqa: E402

B, T, NB = 2, 24, 30
it = acts_data_loader_with_state(FakeTok(), {"stub": StubPositionalSource()}, B, T,
                                 split="train", device="cpu", buffer_size=6)
mismatches = bos_rows = body_rows = 0
for _ in range(NB):
    x, y, acts, stt = next(it)
    xn, zn = x.numpy(), acts["stub"].numpy()
    for b in range(B):
        for t in range(T):
            v = int(xn[b, t])
            if v == BOS:
                bos_rows += 1
                if not np.all(zn[b, t] == 0.0):
                    mismatches += 1
            else:
                body_rows += 1
                doc = (v - 1) // 100                 # which corpus row this token came from
                if not np.all(zn[b, t] == doc + 1):  # activation must encode abs_row == doc index
                    mismatches += 1
check(mismatches == 0 and body_rows > 0,
      f"loader joined every token to its (shard,row) activation ({body_rows} body, {bos_rows} BOS, {mismatches} bad)")

# lookup_workers>0 (ordered thread pool) must be identical to the serial path
it0 = acts_data_loader_with_state(FakeTok(), {"stub": StubPositionalSource()}, B, T,
                                  split="train", device="cpu", buffer_size=6, lookup_workers=0)
itP = acts_data_loader_with_state(FakeTok(), {"stub": StubPositionalSource()}, B, T,
                                  split="train", device="cpu", buffer_size=6, lookup_workers=3)
par_ok = True
for _ in range(NB):
    (x0, y0, a0, s0), (xp, yp, ap, sp) = next(it0), next(itP)
    par_ok = par_ok and np.array_equal(x0.numpy(), xp.numpy()) \
        and np.array_equal(a0["stub"].numpy(), ap["stub"].numpy()) and s0 == sp
check(par_ok, "lookup_workers=3 threaded lookups are byte-identical to serial (order preserved)")


# --------------------------------------------------------------------------- #
print("\n[D] LiveProbeScoreSource (stub score_fn)")


def stub_score_fn(texts):
    out = []
    for text in texts:
        g_ids, _ = char_gemma_encode(text)
        raw = np.zeros((len(g_ids), 3, K), np.float32)
        for g in range(len(g_ids)):
            raw[g, 1, :] = g + 1                      # per-doc-local gemma index (no offline store)
        out.append(raw)
    return out


LIVE = LiveProbeScoreSource(stub_score_fn, STORE, layer=8, nano_enc=FakeNanoEnc(),
                            gemma_encode=char_gemma_encode)
# "ab cd": live per-doc gemma z = [1,2,3,4]; overlap-mean -> ['ab'->1.5, ' '->0, 'cd'->3.5]
lv, _ = LIVE.lookup("ab cd", 3)
check(np.allclose(lv, np.array([[1.5] * K, [0] * K, [3.5] * K], np.float32)),
      "live backend aligns on-the-fly scores identically to the stored backend (mean)")
LIVE_L = LiveProbeScoreSource(stub_score_fn, STORE, layer=8, nano_enc=FakeNanoEnc(),
                              gemma_encode=char_gemma_encode, align_policy="last")
lvl, _ = LIVE_L.lookup("ab cd", 3)
check(np.array_equal(lvl, np.array([[2] * K, [0] * K, [4] * K], np.float32)),
      "live backend honors align_policy='last'")
try:
    LIVE.sample_activation_stats(8, 0)
    check(False, "live gate='auto' must raise")
except NotImplementedError:
    check(True, "live backend does not support gate='auto' (no corpus to sample)")


# --------------------------------------------------------------------------- #
print("\n[E] single-worker alignment throughput (fixture; excludes real gemma fwd)")
reps = 300
t0 = time.perf_counter()
for _ in range(reps):
    for di, text in enumerate(DOCS):
        SRC.lookup_by_row(0, di, text, len(enc.encode_ordinary(text)))
dt = time.perf_counter() - t0
dps = reps * len(DOCS) / dt
print(f"  fixture: {dps:,.0f} docs/s single-worker alignment (fake tokenizers; real gemma "
      f"tokenization dominates in production — see README cost note)")
check(dps > 0, "throughput measured")


# --------------------------------------------------------------------------- #
print("\n[F] staying-ahead loader stats (Amendment 2) + simulated slow source")


class SlowStub(StubPositionalSource):
    """Positional stub that sleeps per lookup — simulates a source that starves."""
    def lookup_by_row(self, sid, row, text, n_tokens):
        time.sleep(0.003)
        return super().lookup_by_row(sid, row, text, n_tokens)


fast_stats, slow_stats = {}, {}
it_fast = acts_data_loader_with_state(FakeTok(), {"stub": StubPositionalSource()}, B, T,
                                      split="train", device="cpu", buffer_size=6, stats=fast_stats)
it_slow = acts_data_loader_with_state(FakeTok(), {"stub": SlowStub()}, B, T,
                                      split="train", device="cpu", buffer_size=6, stats=slow_stats)
for _ in range(8):
    next(it_fast); next(it_slow)
check(fast_stats.get("produced_tokens", 0) > 0 and fast_stats.get("batches", 0) == 8,
      f"loader stats populate produced_tokens/batches ({fast_stats.get('produced_tokens')}t, {fast_stats.get('batches')}b)")
check("queue_depth" in fast_stats and fast_stats["produce_seconds"] > 0,
      f"queue_depth + cumulative produce_seconds reported ({fast_stats.get('queue_depth')} depth, {fast_stats['produce_seconds']:.3f}s)")
check(slow_stats["produce_seconds"] > fast_stats["produce_seconds"],
      f"slow source spends more time producing ({slow_stats['produce_seconds']:.3f}s > {fast_stats['produce_seconds']:.3f}s) — the starvation signal")
# throughput extrapolation (tokens/s) is finite and the slow source is measurably slower
fast_tps = fast_stats["produced_tokens"] / fast_stats["produce_seconds"]
slow_tps = slow_stats["produced_tokens"] / slow_stats["produce_seconds"]
check(slow_tps < fast_tps, f"slow source has lower tokens/s ({slow_tps:,.0f} < {fast_tps:,.0f}) — startup verdict input")


# --------------------------------------------------------------------------- #
print("\n[G] --activation-config custom source class hook (weekday-geometry wiring)")
# Weekday exp configs name an experiment-side RuntimeProbeScoreSource subclass
# via the source spec's "class" (+ "kwargs") fields; injection_train's
# _open_injection_source resolves it with load_source_class and passes kwargs
# through. Prove, from each COMMITTED config, that the named class RESOLVES,
# CONSTRUCTS (real init against the fixture store, kwargs passed through), and
# BEHAVES (present_z realism threshold post-alignment). Guards the historical
# failure mode where 'class'/'kwargs' were silently ignored.
from nanochat.injection.sources import load_source_class  # noqa: E402

_wk_cfg_paths = [os.path.join(REPO, "runs", "weekdays", f"exp{i}_config.json") for i in (2, 3, 4)]
_prev_cwd = os.getcwd()
os.chdir(REPO)   # config "class" paths are repo-root-relative (launch CWD contract)
try:
    for _cfgp in _wk_cfg_paths:
        with open(_cfgp) as _f:
            _spec = json.load(_f)["sources"]["weekdays"]
        _cls = load_source_class(_spec["class"])
        # Expected name derived from the spec, exactly as injection_train's own guard
        # does (assert type(src).__name__ == spec["class"].rsplit(":", 1)[1]).
        _want = _spec["class"].rsplit(":", 1)[1]
        check(_cls.__name__ == _want and issubclass(_cls, RuntimeProbeScoreSource),
              f"{os.path.basename(_cfgp)}: 'class' resolves to a RuntimeProbeScoreSource subclass")
        check("/" in _spec["class"].split(":", 1)[0],
              f"{os.path.basename(_cfgp)}: 'class' uses the collision-proof file-path form")
        check(dict(_spec.get("kwargs") or {}).get("family") == "weekdays",
              f"{os.path.basename(_cfgp)}: source spec pins family=weekdays")
        # Construct EXACTLY like _open_injection_source (fixture store standing in for
        # the HF score repo; "kwargs" passed through). The family kwarg is dropped here
        # ONLY because this fixture store's columns are a/b/c/d, not the real weekday
        # registry columns the guard checks; that guard is proven in
        # runs/lib/test_probe_source.py against the real registry.
        _kw = {k: v for k, v in dict(_spec.get("kwargs") or {}).items() if k != "family"}
        _src = _cls(STORE, [0], layer=8, nano_enc=FakeNanoEnc(),
                    gemma_encode=char_gemma_encode, concepts=None,
                    align_policy=_spec["align_policy"],
                    noise_sigma=float(_spec["noise_sigma"]), seed=0, name="weekdays",
                    **_kw)
        check(type(_src).__name__ == _want and _src.present_z == float(_kw["present_z"]),
              f"{os.path.basename(_cfgp)}: custom class constructs with the spec's kwargs")
        check(_src.present_z == 0.0,
              f"{os.path.basename(_cfgp)}: present_z=0 — the SITE's relu owns thresholding")
finally:
    os.chdir(_prev_cwd)

# behavioral: a source-level present_z zeroes post-alignment rows whose max z falls
# under it, EXACTLY (>= is kept verbatim). "ab cd" mean-aligns to rows [1.5, 0, 3.5] per
# channel (section [A]). Set explicitly here, NOT read from a config: the dose configs
# ship present_z=0 because the site's relu thresholds instead, but the row-gate behavior
# is still part of the source's contract.
os.chdir(REPO)
try:
    with open(os.path.join(REPO, "runs", "weekdays", "exp2_config.json")) as _f:
        _spec2 = json.load(_f)["sources"]["weekdays"]
    _cls2 = load_source_class(_spec2["class"])

    def _mk(present_z):
        return _cls2(STORE, [0], layer=8, nano_enc=FakeNanoEnc(),
                     gemma_encode=char_gemma_encode, noise_sigma=0.0, present_z=present_z)

    _wsrc, _wsrc_cfg = _mk(2.0), _mk(float(_spec2["kwargs"]["present_z"]))
finally:
    os.chdir(_prev_cwd)
_wz, _ = _wsrc.lookup_by_row(0, 0, "ab cd", 3)
check(np.array_equal(_wz, np.array([[0] * K, [0] * K, [3.5] * K], np.float32)),
      "present_z=2.0 zeroes rows with max z < 2.0 and keeps rows >= 2.0 verbatim")
_wz0, _ = _wsrc_cfg.lookup_by_row(0, 0, "ab cd", 3)
check(np.array_equal(_wz0, np.array([[1.5] * K, [0] * K, [3.5] * K], np.float32)),
      "the config's present_z=0 passes every aligned row through untouched")


print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
if __name__ == "__main__":
    sys.exit(1 if fails else 0)


def test_runtime_probe_source():
    assert not fails, f"{len(fails)} failures: {fails}"
