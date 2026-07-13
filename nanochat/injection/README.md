# nanochat.injection — feature injection

## Status & review guide (2026-07-13)

History reads linearly: `main` → `experimental-setup` (the prior baseline-run
setup, 3 commits) → the injection work on top (`git log --oneline
pre-injection..HEAD`):

1. `9f9469b` — v1 coord-injection diffs applied verbatim (baseline for review)
2. `29fa9a2` — modules moved in as `nanochat.oracle`, imports normalized, align vendored
3. `96b79cc` — gpt.py routed through `InjectionSite` (v2), optimizer contract
4. `197bfdd` — tests + first README
5. `3adf4c8` — review-readiness README pass
6. `9b6129d` — **base_train back to stock** (byte-identical to `pre-injection`)
7. `c6d8f3d` — **package rename `nanochat.oracle` → `nanochat.injection`**, ActivationSource interface, gate default 1.0, activation-store-v2
8. `9d38414` — **`scripts/injection_train.py`** (dedicated injection training script)
9. (tip) — this README

Suggested review order: `sites.py` (the injection contract) → `sources.py`
(ActivationSource + store format) → `gpt.py` (`setup_injection_sites`, forward
hook) → `scripts/injection_train.py` → `activation_dataloader.py` →
`scripts/precompute_activations.py` → `tests/`. All 3 test files +
`python -m nanochat.injection.smoke` pass on CPU.

**Why the package is named `injection`, not `oracle`**: in this project's
terminology "oracle" is reserved for the *reliance failure mode* under study,
not for the machinery that injects features. Both families (geometric-manifold
`inject.py` and the contextual activation injection) live here.

**Blockers before any injected training run** (deliberate, not oversights):

- **No activation store exists.** The old `oracle-coords`/`-b` HF repos were
  deleted 2026-07-09; a store must be precomputed (fleet pipeline below) or
  repackaged from the probe-score stores (see ProbeScoreSource status).
- **Encoder gap** (qwen-encoder flavor): the precompute loader implements the
  legacy Exp-A encoder head, but the Exp-A checkpoint repo (`oracle-encoder`)
  was deleted 2026-07-09. What exists: per-layer oracles on
  `kaushikreddyxyz/oracle-encoders` (`layer06/08/14/best_stripped.pt`, head =
  `OracleMLPHead` 1024→4096→54 — a *different* head). Before a fleet run,
  point the loader at a per-layer checkpoint via a small adapter (natural
  choice: `layer08`) or supply a surviving local Exp-A checkpoint.
  Experiment-design work, not a bug.
- **ProbeScoreSource repackaging is a NotImplemented boundary**: the reader is
  complete; the offline `--mode repackage-probe-scores` pass is not (see below).
- GPU-side validations never ran on the real stack (end of this file).
- Open decision: whether to add an **activations-on eval pass** (eval is
  activations-off by design today).

---

Two feature families:

1. **Geometric-manifold injection** (`inject.py`, `smoke.py`): a frozen
   additive feature that is a pure function of the *token id* (ring / line /
   sphere / helix in reserved residual dims), added before the trunk.
   `python -m nanochat.injection.smoke` validates it end-to-end on CPU.
2. **Contextual activation injection** (`sites.py`, `sources.py`,
   `activation_dataloader.py`, `align.py`, `scripts/precompute_activations.py`,
   `scripts/injection_train.py`): per-token-*occurrence* activations added into
   the residual stream after a chosen block during pretraining. This README
   documents family 2.

## What an injection is

An injection site (`sites.InjectionSite`) decomposes into exactly three parts
with **fixed optimizability rules**:

| part | what it is | optimizable? |
|---|---|---|
| **gate** | loudness dial: injected per-token RMS = `gate` × per-token RMS(residual). `gate=0` is exactly off. **Default 1.0** — as loud as the stream itself. | **NEVER.** A parameter so autograd *assigns* it a gradient every backward (a loggable want-signal, dL/dgate), but it sits in no optimizer group and is never stepped. |
| **activation** | the content: a `(B, T, r)` tensor per batch from a pluggable `ActivationSource`. | **NEVER.** Produced without grad by the dataloader and additionally `detach()`ed by the site. |
| **direction** | `(r, n_embd)` map from activation channels into the residual stream. | **The only optionally-trainable part**, controlled purely by freeze/unfreeze. Frozen + orthonormal init = "tabular" injection (the fixed-P v1 behavior); unfrozen = "free" injection. |

Site math (per token):

```
z     = a @ D                       # activation through direction
z_hat = z / rms(z)                  # D's scale can never fight the gate
x     = x + gate * rms(x).detach() * z_hat
```

`rms(x)` is detached: it *measures* the stream to calibrate amplitude; it is
not a gradient path into the stream's own norm. Invariants (pinned by
`tests/test_injection_sites.py`):

- **zero activation rows (BOS / missing doc) are an EXACT no-op** — no NaN, no
  branch (the `rms(z)` clamp), torch.compile-friendly;
- injected per-token RMS == `gate` × per-token RMS(x);
- `gate=0` is an exact forward no-op AND blocks all gradient to the direction;
- forward values are identical to the retired v1 inline formula
  `x + beta*(rms_x/rms_z)*zc` (the detach only changes gradients, deliberately).

`GPT.setup_injection_sites(cfgs)` attaches sites as an `nn.ModuleDict`
(checkpointed, optimizer-visible); `GPT.forward(..., acts={name: (B,T,r)})`
fires each site after its own block. `acts=None` (eval, inference, vanilla
runs) is bit-identical to a model without sites.

### Gate default changed to 1.0

`InjectionCfg.gate` (and `injection_train.py --gate`) default to **1.0**: the
injected signal is as loud as the residual stream itself. The v1 run used
0.05 — **anyone reproducing v1 must pass `--gate 0.05` explicitly**.

## Activation sources (`sources.py`)

The per-token content is called **activations** everywhere user-facing.
`ActivationSource` is the explicit interface:

- `.name` — the site the source feeds; `.r` — channels;
- `.lookup(doc_text, n_tokens) -> ((n_tokens, r) float32 | None, key)`;
- `.add_noise(z, key)` — deterministic train-time noise, seeded by
  `(train seed, doc content hash)` (DDP-rank/resume-independent, not
  memorizable as a per-position identity).

**Contract (do not "fix" this)**: `lookup` returning `None` (doc missing from
a store, or stored token count mismatching = tokenizer drift) MUST be mapped to
**EXACT zeros with NO noise** by the caller. The site renormalizes any nonzero
row to full gate amplitude, so noised zeros would inject pure noise at full
strength on exactly the docs we know nothing about; exact zeros keep the
injection a strict no-op there. The same reasoning makes store quantization
**zero-preserving with no mean-centering** (raw 0 → int8 0 → dequant 0 → no-op);
per-column mean/std are recorded in `meta.json` but only the single global
`scale` is applied by the reader.

Implementations:

- **`QwenEncoderSource`** (`source_kind: "qwen-encoder"`) — precomputed
  predictions of the frozen Qwen oracle-encoder, structured into r=14
  ring/PCA activations by `scripts/precompute_activations.py` (fit/sweep/
  assemble pipeline below). This is the renamed v1 `CoordSource`.
- **`ProbeScoreSource`** (`source_kind: "probe-scores"`) — gold gemma probe
  scores (one layer's 54 standardized scores, per the binding
  one-layer-per-model rule) repackaged per nanochat token. Reader complete;
  producer is a boundary (status below).
- **`FnSource`** — arbitrary callable `fn(text, n_tokens) -> (n_tokens, r)`
  for synthetic/control injections. A future **runtime gold-probe source**
  (computing gemma probe scores on the fly) plugs in through the same
  protocol.
- `open_store(dir, ...)` dispatches on `meta.json["source_kind"]`.

### Activation store format (`activation-store-v2`)

One directory:

```
activations.int8   memmap int8 [n_doc_tokens, r]   standardized, quantized (zero-preserving)
index.npy          structured [n_docs] (hash uint64, off int64, n int32)
meta.json          {"format": "activation-store-v2", "source_kind": ..., "r", "scale", ...}
P.npy              optional float32 [n_embd, r] fixed orthonormal projection (qwen flavor)
```

Activations are stored **per document keyed by content hash** (the training
loader packs+crops docs in a data-dependent order; hash keying is
order-/DDP-independent) and ride through the exact same best-fit packing as
the tokens (`activation_dataloader.py`, pinned bit-identical to the stock
loader by `tests/test_activation_lockstep.py`). Readers **require** the v2
format keys; legacy `coords.int8` support was **dropped entirely** (no pre-v2
store exists anywhere — the old HF stores were deleted).

### ProbeScoreSource repackaging status (NotImplemented boundary)

What exists: the complete reader (`ProbeScoreSource`), and the alignment core
— `nanochat_char_offsets` (nanochat byte→char offsets) + `align.gemma_to_qwen_map`
prefix mode, which is tokenizer-agnostic and already tested; gemma offsets come
from `align.get_offsets` on the (gated) gemma-2-2b fast tokenizer.

What remains (`scripts/precompute_activations.py --mode repackage-probe-scores`
currently raises NotImplementedError with the same statement): walk each
ClimbMix shard's parquet docs in row order alongside
`hf.co/kaushikreddyxyz/climbmix-scored(+overflow…-7)` shard files
(`scores_<sid>.npy` int8 `[n,3,54]`, `docs_<sid>.jsonl` `{doc,start,n}` spans,
shards 0–184, full coverage); gemma-tokenize each doc
(`add_special_tokens=False`, verify `len == n`, bit-check against
`tokens_<sid>.npy` when local); map each nanochat token to the **last gemma
token whose char span ends at or before it** (prefix mode — causal, no future
leakage; this is the chosen policy, not mean-over-span); slice ONE layer's 54
columns (`--layer 8` default; concept axis order = `columns.json["concepts"]`,
the family-sorted main-block order — the permutation trap), dequantize with
`quant.json` and standardize with `corpus_stats.json`; unmapped tokens get
exact zero rows; write per-shard v2 store files and reuse assemble + the
mandatory preflight.

## Training (`scripts/injection_train.py`)

**base_train is stock again**: `scripts/base_train.py` was designed for
non-injection models and is byte-identical to the `pre-injection` tag —
vanilla runs have zero injection surface. `scripts/injection_train.py` is the
injection script: a deliberate fork whose shared body is kept byte-identical
to base_train (so `diff scripts/base_train.py scripts/injection_train.py`
shows only the injection hunks — keep it that way when either changes). The
old `--inject-coords/--inject-beta/...` flag family is gone.

**Single tabular site from a store** (legacy-equivalent v1 run shown — note
the explicit 0.05 gate):

```
python -m scripts.injection_train -- --activation-store <store_dir> \
    --after-block 7 --gate 0.05 --noise-sigma 0.15
```

One site named `"acts"`: frozen direction pinned to the store's `P.npy` when
present (else seeded orthonormal via the store's `p_seed`), source opened by
`meta.json` kind. Omitting `--gate` gives the new default **1.0**.

**General multi-site form**:

```
python -m scripts.injection_train -- --activation-config path/to/config.json
```

```json
{
  "sites": [
    {"name": "acts", "r": 14, "after_block": 7, "gate": 0.05,
     "trainable_direction": false, "direction_seed": 1337},
    {"name": "free", "r": 54, "after_block": 12, "gate": 1.0,
     "trainable_direction": true, "direction_init": "orthonormal", "optim": "muon"}
  ],
  "sources": {
    "acts": {"kind": "qwen-encoder", "dir": "/workspace/acts_qwen", "noise_sigma": 0.15},
    "free": {"kind": "probe-scores", "dir": "/workspace/acts_probes", "noise_sigma": 0.15}
  }
}
```

Site dicts are `sites.InjectionCfg` fields; `sources` keys must match site
names; `FnSource` remains programmatic. Exactly one of
`--activation-store`/`--activation-config` is required.

## Optimizer contract

Wired in `GPT.setup_optimizer` (asserted by the param-count check there):

- **gates** (`_never_optimize`) go in **no** param group;
- **frozen directions** are skipped;
- **trainable directions** join an AdamW group by default, or Muon with
  `"optim": "muon"` per site;
- **weight decay is 0.0 for directions in both flavors** — the site normalizes
  the direction's scale away (`z / rms(z)`), so decay only shrinks the matrix
  toward the rms clamp. injection_train's wd scheduler skips the
  `injection`-tagged muon groups for the same reason.

After any `load_state_dict(..., assign=True)` (resume, eval load), call
`sites.reassert_optimizability(model.injection_sites)` — assign-loads replace
the Parameter objects and drop the gate's `_never_optimize` stamp.
injection_train and `checkpoint_manager.build_model` already do this.

## Checkpoints

- Injected checkpoints carry `injection_sites.*` keys and
  `injection_sites_config` (+ `injection_source_specs`) in the meta json;
  `checkpoint_manager.build_model` rebuilds the sites from meta so eval
  scripts load them (sites stay dormant — eval is activations-off).
- **Warm-start from a vanilla checkpoint** into an injected run is allowed:
  only `injection_sites.*` keys may be missing (they keep their fresh init).
  Frozen directions leave the optimizer groups identical to vanilla, so the
  optimizer state resumes too; trainable directions change the group
  structure and cannot resume a vanilla optimizer state.
- A vanilla model strict-loading an injected checkpoint fails loudly (by
  design — use `build_model`).

## Eval runs activations-off

Val bpb, CORE, and sampling all call the model with `acts=None`: **evaluation
is activations-off by design** (the model must not need the injected signal to
function). An activations-on eval pass is a separate, deliberate decision.

## Precompute pipeline (`scripts/precompute_activations.py`, qwen flavor)

Produces the doc-hash-keyed int8 activation store. Prereqs on every pod: the
**baseline run's tokenizer** at `$NANOCHAT_BASE_DIR/tokenizer` (alignment is
keyed to its exact merges) and the ClimbMix shards at
`$NANOCHAT_BASE_DIR/base_data_climbmix` (`python -m nanochat.dataset`). The
probe set json lives in the superproject (`attribution/out/probe_set.json`).
Encoder checkpoint: see the encoder-gap blocker above.

```bash
# 1) pod 0 fits continents PCA + the global scale ONCE (shared by all pods)
python -m scripts.precompute_activations --mode fit --encoder-ckpt <expA.pt> \
    --probe-set <superproject>/attribution/out/probe_set.json \
    --shards 0-3 --out /workspace/acts

# 2) every pod sweeps its round-robin shard slice (resumable, atomic per shard)
python -m scripts.precompute_activations --mode sweep --encoder-ckpt <expA.pt> \
    --probe-set <superproject>/attribution/out/probe_set.json \
    --shards 0-190 --out /workspace/acts --pod-index $P --n-pods $NP \
    --fast-forward --feeder-workers 8          # ~3.2x throughput

# 3) after the fleet, on ONE node with all per-shard files present:
python -m scripts.precompute_activations --mode merge-stats --out /workspace/acts
python -m scripts.precompute_activations --mode assemble --encoder-ckpt <expA.pt> \
    --probe-set <superproject>/attribution/out/probe_set.json \
    --shards 0-190 --out /workspace/acts
    # -> activations.int8 / index.npy / meta.json / P.npy
    # assemble HARD-FAILS on missing shards (--allow-missing-shards to override)

# 4) *** MANDATORY pre-launch gate — never skip this ***
python -m scripts.precompute_activations --mode preflight \
    --shards 0-190 --out /workspace/acts --preflight-docs 1024
```

**Why preflight is mandatory**: it cross-checks the CONSUMER token path
(`RustBPETokenizer.encode(batch, prepend=bos)`, exactly as the activation
dataloader) against the assembled store and hard-fails on tokenizer-contract
drift or token coverage < 99.9%. The failure it catches otherwise **silently
trains a baseline**: every lookup misses → all activations fall back to zero →
the injection no-ops on every token and nothing tells you. It is
source-kind-agnostic (works for probe-scores stores too).

Optional QA: `--mode verify` (recompute K docs live, assert int8 round-trip
within one quant step), `--mode measure-crossing` (prefix-mode crossing rate).
`--fast-forward` (length-bucketed cross-doc batching) equals the serial path
within one int8 step with bit-identical zero-fallback; `--feeder-workers`
parallelizes tokenize/align in spawn workers with byte-identical output.

## Tests (CPU, no GPU / tokenizer / checkpoint needed)

```bash
python -m pytest tests/test_injection_sites.py tests/test_activation_lockstep.py \
                 tests/test_precompute_activations.py
python -m nanochat.injection.smoke
```

- `test_injection_sites.py` — site invariants (RMS calibration, exact zero-row
  no-op, gate-0 no-op + zero direction grad, gate default 1.0, optimizer
  split, state-dict keys, v1↔v2 forward equivalence) and the GPT wiring
  (acts=None ≡ vanilla; optimizer contract; step behavior).
- `test_activation_lockstep.py` — the ride-along loader against the REAL
  packing source (ast-extracted from `nanochat/dataloader.py`): bit-identical
  token stream, activation↔token alignment through best-fit + crops,
  exact-zero BOS / missing-doc rows even with noise on, deterministic noise,
  store round-trip, format/source_kind enforcement + `open_store` dispatch.
- `test_precompute_activations.py` — the producer: phase-angle mapping (all 54
  one-hot concepts), PCA determinism, zero-preserving quantization, store
  assemble/read-back through the real reader, pod-sharding coverage, byte→char
  offsets under adversarial UTF-8, chunked + fast-forward flush equivalence,
  preflight drift detection, Welford merge-stats. Needs the superproject's
  `probe_set.json` (`$ORACLE_PROBE_SET` or `../attribution/out/probe_set.json`);
  skips otherwise.

Not validated on CPU (needs a real run): torch.compile + fp8 over the site
graph, store index throughput at the 27M-doc scale (~2-3 GB/rank), and the
precompute encoder wiring on a real checkpoint + real tokenizer pair. SMOKE an
injected launch first (a few steps, nothing saved): step time within ~3% of
baseline and the per-site banner must print the expected `r` / `after_block` /
`gate` / docs count.
