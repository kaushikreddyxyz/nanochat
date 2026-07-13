"""CPU validation of the ride-along activation loader against the REAL
nanochat packing source (ast-extracted from nanochat/dataloader.py so the
reference cannot drift): bit-identical token stream, activation<->token
alignment through best-fit + crops, exact-zero BOS / missing-doc rows even
with noise on, deterministic noise, store int8 round-trip, injection math.

Script-style on purpose (runs at import; the trailing test_* shim makes pytest
report it): `python tests/test_activation_lockstep.py` also works standalone.
``nanochat.dataloader`` is stubbed in sys.modules (it needs pyarrow + real
parquet shards); the stub is removed again at the end of the module.
"""
import ast
import json
import os
import sys
import tempfile
import types

import numpy as np
import torch

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(TESTS, ".."))          # nanochat repo root
STORE = tempfile.mkdtemp(prefix="activation_store_test_")
sys.path.insert(0, REPO)

# ---------------------------------------------------------------- fake corpus
R = 14
SCALE = 0.05
N_DOCS = 120
BOS = 0
rng = np.random.default_rng(7)
DOC_LENS = rng.integers(3, 41, size=N_DOCS)      # body lengths; capacity=T+1=17 -> crops happen
MISSING = set(range(0, N_DOCS, 9))               # every 9th doc absent from the store
MISMATCH = {5}                                   # stored with wrong n -> must also fall back to zero
ZERO_ROW_DOCS = {7, 16}                          # stored docs whose row 1 is all-zero (unmapped/
                                                 # concept-free token): must stay EXACT zero under noise


def doc_text(i):
    return f"doc-{i}-len-{DOC_LENS[i]}"


def body_ids(i):
    return [int(i) * 10000 + j + 1 for j in range(int(DOC_LENS[i]))]


def int8_vals(i, L):
    j = np.arange(L)[:, None]
    d = np.arange(R)[None, :]
    q = (((i * 7 + j * 13 + d * 29) % 250) - 125).astype(np.int8)
    if i in ZERO_ROW_DOCS and L > 1:
        q[1] = 0
    return q


class FakeTok:
    def get_bos_token_id(self):
        return BOS

    def encode(self, texts, prepend=None, append=None, num_threads=8):
        out = []
        for t in texts:
            i = int(t.split("-")[1])
            ids = body_ids(i)
            if prepend is not None:
                ids = [prepend] + ids
            out.append(ids)
        return out


def fake_batches_factory(tokenizer_batch_size):
    def gen():
        pq = rg = 0
        epoch = 1
        while True:
            for s in range(0, N_DOCS, tokenizer_batch_size):
                batch = [doc_text(i) for i in range(s, min(s + tokenizer_batch_size, N_DOCS))]
                yield batch, (pq, rg, epoch)
                rg += 1
            epoch += 1
    return gen()


# ---------------------------------------------- stub nanochat.dataloader ONLY
# (the real module imports pyarrow and lists real parquet shards; the loader
# under test only needs _document_batches)
fake_dl_mod = types.ModuleType("nanochat.dataloader")
_batch_stream = {"it": None}
fake_dl_mod._document_batches = lambda split, resume, tbs: _batch_stream["it"]
_prev_dl_mod = sys.modules.get("nanochat.dataloader")
sys.modules["nanochat.dataloader"] = fake_dl_mod
sys.modules.pop("nanochat.injection.activation_dataloader", None)  # force rebind to the stub

from nanochat.injection.sources import QwenEncoderSource, doc_hash  # noqa: E402
from nanochat.injection.activation_dataloader import acts_data_loader_with_state  # noqa: E402

# ------------------------------------------------- extract REAL stock loader by ast
src_path = os.path.join(REPO, "nanochat", "dataloader.py")
tree = ast.parse(open(src_path).read())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
          and n.name == "tokenizing_distributed_data_loader_with_state_bos_bestfit")
ns = {"torch": torch, "_document_batches": fake_dl_mod._document_batches}
exec(compile(ast.Module(body=[fn], type_ignores=[]), src_path, "exec"), ns)
stock_loader_fn = ns["tokenizing_distributed_data_loader_with_state_bos_bestfit"]

# ------------------------------------------------- build the activation store on disk
os.makedirs(STORE, exist_ok=True)
rows, recs, off = [], [], 0
for i in range(N_DOCS):
    if i in MISSING:
        continue
    L = int(DOC_LENS[i])
    stored_n = L + 1 if i in MISMATCH else L      # MISMATCH: wrong n on purpose
    q = int8_vals(i, stored_n)
    rows.append(q)
    recs.append((doc_hash(doc_text(i)), off, stored_n))
    off += stored_n
allq = np.concatenate(rows, axis=0)
allq.tofile(os.path.join(STORE, "activations.int8"))
index = np.array(recs, dtype=[("hash", "<u8"), ("off", "<i8"), ("n", "<i4")])
np.save(os.path.join(STORE, "index.npy"), index)
json.dump({"format": "activation-store-v2", "source_kind": "qwen-encoder",
           "r": R, "scale": SCALE}, open(os.path.join(STORE, "meta.json"), "w"))

B, T, NB = 2, 16, 40  # NB batches ~ 2*17*2*40 tokens >> one epoch of docs


def run_acts_loader(noise_sigma, seed=1337):
    _batch_stream["it"] = fake_batches_factory(128)
    src = QwenEncoderSource(STORE, noise_sigma=noise_sigma, seed=seed, name="acts")
    it = acts_data_loader_with_state(FakeTok(), {"acts": src}, B, T, split="train",
                                     device="cpu", buffer_size=8)
    out = []
    for _ in range(NB):
        x, y, acts, st = next(it)
        out.append((x.clone(), y.clone(), acts["acts"].clone(), dict(st)))
    return out, src


def run_stock_loader():
    _batch_stream["it"] = fake_batches_factory(128)
    it = stock_loader_fn(FakeTok(), B, T, split="train", device="cpu", buffer_size=8)
    out = []
    for _ in range(NB):
        x, y, st = next(it)
        out.append((x.clone(), y.clone(), dict(st)))
    return out


def expected_act(v, src):
    """Expected activation row for packed token value v under source src."""
    if v == BOS:
        return np.zeros(R, np.float32)
    i, j = (v - 1) // 10000, (v - 1) % 10000
    if i in MISSING or i in MISMATCH:
        return np.zeros(R, np.float32)
    base = int8_vals(i, int(DOC_LENS[i])).astype(np.float32) * SCALE
    if src.noise_sigma > 0:
        zero_rows = ~np.any(base != 0.0, axis=1)
        base = src.add_noise(base, int(doc_hash(doc_text(i))))
        base[zero_rows] = 0.0            # exact-zero rows stay zero through the noise
    return base[j]


fails = 0

# --- 1. token lockstep vs the real packing code ---
stock = run_stock_loader()
acts0, src0 = run_acts_loader(noise_sigma=0.0)
for k, ((sx, sy, sst), (cx, cy, cz, cst)) in enumerate(zip(stock, acts0)):
    assert torch.equal(sx, cx) and torch.equal(sy, cy), f"token desync at batch {k}"
    assert sst == cst, f"state desync at batch {k}: {sst} vs {cst}"
print(f"[1] token lockstep vs real bos_bestfit source: OK ({NB} batches, B={B}, T={T})")

# sanity: crops actually happened (some doc bodies truncated mid-doc)
n_bos = sum(int((x == BOS).sum()) for x, _, _ in stock)
print(f"    (packed {NB*B*T} positions, {n_bos} BOS rows; doc lens 3..40 vs capacity 17 -> crops exercised)")

# --- 2/3. activation correctness incl. crops, BOS, missing docs (noise=0) ---
checked = 0
for x, y, z, _ in acts0:
    xn, zn = x.numpy(), z.numpy()
    for b in range(B):
        for t in range(T):
            exp = expected_act(int(xn[b, t]), src0)
            if not np.allclose(zn[b, t], exp, atol=0, rtol=0):
                fails += 1
                if fails < 5:
                    print(f"MISMATCH b={b} t={t} tok={xn[b,t]} got={zn[b,t][:3]} exp={exp[:3]}")
            checked += 1
assert fails == 0, f"{fails} activation mismatches (noise=0)"
print(f"[2] activation<->token alignment exact through best-fit+crop: OK ({checked} positions)")
miss_seen = sum(int(((x.numpy() != BOS) & ((x.numpy() - 1) // 10000 % 9 == 0)).sum()) for x, _, _, _ in acts0)
print(f"[3] BOS + missing/mismatch docs -> exact zero activations: OK ({n_bos} BOS, {miss_seen} missing-doc positions)")

# --- 3b/4. noise path: missing docs still EXACT zero; noise deterministic ---
actsN, srcN = run_acts_loader(noise_sigma=0.15)
actsN2, _ = run_acts_loader(noise_sigma=0.15)
nz_checked = zero_checked = zero_row_checked = 0
for (x, y, z, _), (x2, y2, z2, _) in zip(actsN, actsN2):
    assert torch.equal(z, z2), "noise not deterministic across loader instantiations"
    xn, zn = x.numpy(), z.numpy()
    for b in range(B):
        for t in range(T):
            v = int(xn[b, t])
            exp = expected_act(v, srcN)
            assert np.allclose(zn[b, t], exp, atol=1e-6), f"noisy activation mismatch tok={v}"
            if v != BOS and ((v - 1) // 10000 in MISSING or (v - 1) // 10000 in MISMATCH):
                assert np.all(zn[b, t] == 0.0), "missing doc got NOISED activations (injection would fire!)"
                zero_checked += 1
            elif v != BOS and (v - 1) // 10000 in ZERO_ROW_DOCS and (v - 1) % 10000 == 1:
                assert np.all(zn[b, t] == 0.0), \
                    "exact-zero row in a KNOWN doc got NOISED (site would inject pure noise at full gate)"
                zero_row_checked += 1
            elif v != BOS:
                nz_checked += 1
assert torch.equal(actsN[0][0], acts0[0][0]), "noise changed the TOKEN stream"
assert zero_row_checked > 0, "fixture never exercised an all-zero stored row"
print(f"[4] noise=0.15: per-doc-hash deterministic, tokens unchanged, missing docs exactly zero, "
      f"in-doc zero rows stay zero: OK ({nz_checked} noised, {zero_checked} zero-fallback, "
      f"{zero_row_checked} in-doc-zero positions)")

# --- 5. store round-trip ---
i_ok = next(i for i in range(N_DOCS) if i not in MISSING and i not in MISMATCH)
z, h = src0.lookup(doc_text(i_ok), int(DOC_LENS[i_ok]))
assert z is not None and np.array_equal(z, int8_vals(i_ok, int(DOC_LENS[i_ok])).astype(np.float32) * SCALE)
assert src0.lookup(doc_text(list(MISSING)[1]), int(DOC_LENS[list(MISSING)[1]]))[0] is None
assert src0.lookup(doc_text(5), int(DOC_LENS[5]))[0] is None  # stored-n mismatch
assert src0.lookup(doc_text(i_ok), int(DOC_LENS[i_ok]) + 1)[0] is None  # queried-n mismatch
print(f"[5] store int8 round-trip exact; miss/length-mismatch -> None: OK (docs={len(src0)})")

# --- 5b. reader validates format + source_kind ---
BAD = tempfile.mkdtemp(prefix="activation_store_bad_")
allq.tofile(os.path.join(BAD, "activations.int8"))
np.save(os.path.join(BAD, "index.npy"), index)
json.dump({"r": R, "scale": SCALE}, open(os.path.join(BAD, "meta.json"), "w"))  # pre-v2 meta
try:
    QwenEncoderSource(BAD)
    raise AssertionError("reader must reject a store without format=activation-store-v2")
except ValueError:
    pass
json.dump({"format": "activation-store-v2", "source_kind": "probe-scores",
           "r": R, "scale": SCALE}, open(os.path.join(BAD, "meta.json"), "w"))
try:
    QwenEncoderSource(BAD)
    raise AssertionError("QwenEncoderSource must reject a probe-scores store")
except ValueError:
    pass
from nanochat.injection.sources import ProbeScoreSource, open_store  # noqa: E402
assert isinstance(open_store(BAD), ProbeScoreSource), "open_store must dispatch on source_kind"
print("[5b] reader enforces format/source_kind; open_store dispatches: OK")

# --- 6. injection math on CPU (zero activations -> exact no-op; gate convention) ---
x = torch.randn(2, 8, 32, dtype=torch.float32)
P = torch.linalg.qr(torch.randn(32, R, dtype=torch.float64))[0].to(torch.float32)
beta = 0.05
def inject(x, acts):
    zc = acts.to(x.dtype) @ P.t()
    rms_x = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    rms_z = zc.pow(2).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    return x + beta * (rms_x / rms_z) * zc
out0 = inject(x, torch.zeros(2, 8, R))
assert torch.equal(out0, x), "zero activations must be an EXACT no-op"
zr = torch.randn(2, 8, R)
outr = inject(x, zr)
d = outr - x
ratio = d.pow(2).mean(-1).sqrt() / x.pow(2).mean(-1).sqrt()
assert torch.allclose(ratio, torch.full_like(ratio, beta), rtol=1e-4), ratio
assert torch.isfinite(outr).all()
mixed = zr.clone(); mixed[0, :4] = 0.0
outm = inject(x, mixed)
assert torch.equal(outm[0, :4], x[0, :4]) and torch.isfinite(outm).all()
print(f"[6] injection math: zero rows exact no-op; per-token injected RMS == gate*RMS(x) "
      f"(max dev {float((ratio-beta).abs().max()):.2e}); mixed zero/nonzero rows no NaN: OK")

print("\nALL CHECKS PASSED")

# ---------------------------------------------- un-stub nanochat.dataloader
# (activation_dataloader is popped too: it holds a binding to the stub)
if _prev_dl_mod is not None:
    sys.modules["nanochat.dataloader"] = _prev_dl_mod
else:
    sys.modules.pop("nanochat.dataloader", None)
sys.modules.pop("nanochat.injection.activation_dataloader", None)


def test_activation_lockstep():
    """All checks above ran (assert-based) at module import; reaching this
    no-op test means every one of them passed."""
