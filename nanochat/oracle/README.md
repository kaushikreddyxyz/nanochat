# nanochat.oracle — oracle-feature injection

## Status & review guide (2026-07-13)

History reads linearly: `main` → `experimental-setup` (the prior baseline-run
setup, 3 commits) → the injection work on top (4 commits):

1. `9f9469b` — the v1 coord-injection diffs applied verbatim (baseline for review)
2. `29fa9a2` — modules moved in as `nanochat.oracle`, imports normalized, align vendored
3. `96b79cc` — gpt.py routed through `InjectionSite` (v2), optimizer contract, legacy-flag compat
4. `197bfdd` — tests + this README

Suggested review order: `injections.py` (the design contract lives in its
docstring) → `gpt.py` (`setup_injection_sites`, the forward hook) →
`scripts/base_train.py` (flags, param groups) → `coords_store.py` /
`coord_dataloader.py` (unchanged v1 semantics) → `scripts/precompute_coords.py`
→ `tests/`. All 3 test files + `python -m nanochat.oracle.smoke` pass on CPU.

**Blockers before any injected training run** (deliberate, not oversights):

- **The coord store does not exist.** The old `oracle-coords`/`-b` HF repos
  were deleted 2026-07-09; the precompute fleet (below) must run first.
- **Encoder gap**: the precompute loader implements the legacy Exp-A encoder
  head, but the Exp-A checkpoint repo (`oracle-encoder`) was also deleted
  2026-07-09. What exists: the per-layer oracles on
  `kaushikreddyxyz/oracle-encoders` (`layer06/08/14/best_stripped.pt`,
  head = `OracleMLPHead` 1024→4096→54 — a *different* head). Before the fleet
  runs, either point the loader at a per-layer checkpoint via a small adapter
  (natural choice: `layer08`, since `build_coords` consumes the L8 block), or
  supply a surviving local Exp-A checkpoint. This is experiment-design work,
  not a bug.
- GPU-side validations never run on the real stack (see the end of this file).
- Open decision: whether to add a **coords-on eval pass** (eval is coords-off
  by design today).

---

Two feature families live here:

1. **Geometric-manifold oracle** (`inject.py`, `smoke.py`): a frozen additive
   feature that is a pure function of the *token id* (ring / line / sphere /
   helix in reserved residual dims), added before the trunk.
   `python -m nanochat.oracle.smoke` validates it end-to-end on CPU.
2. **Contextual coord injection** (`injections.py`, `coords_store.py`,
   `coord_dataloader.py`, `align.py`, plus `scripts/precompute_coords.py`):
   per-token-*occurrence* activations — e.g. probe-score coords produced by a
   frozen Qwen encoder over each document — added into the residual stream
   after a chosen block during pretraining. This README documents family 2.

## What an injection is

An injection site (`injections.InjectionSite`) decomposes into exactly three
parts with **fixed optimizability rules**:

| part | what it is | optimizable? |
|---|---|---|
| **gate** | loudness dial: injected per-token RMS = `gate` × per-token RMS(residual). `gate=0` is exactly off. | **NEVER.** It is a parameter so autograd *assigns* it a gradient every backward (a loggable want-signal, dL/dgate), but it sits in no optimizer group and is never stepped. |
| **activation** | the content: a `(B, T, r)` tensor per batch from a pluggable source (Qwen coord store, gold probe scores, any `FnActivation`). | **NEVER.** Produced without grad by the dataloader and additionally `detach()`ed by the site. |
| **direction** | `(r, n_embd)` map from activation channels into the residual stream. | **The only optionally-trainable part**, controlled purely by freeze/unfreeze (`requires_grad`). Frozen + orthonormal init = the "tabular" injection (the fixed-P v1 behavior); unfrozen = "free" injection (the model learns where the feature lives). |

Site math (per token):

```
z     = a @ D                       # activation through direction
z_hat = z / rms(z)                  # D's scale can never fight the gate
x     = x + gate * rms(x).detach() * z_hat
```

`rms(x)` is detached: it *measures* the stream to calibrate amplitude; it is
not a path for the injection to shape the stream's own norm gradients.
Invariants (all pinned by `tests/test_injection_sites.py`):

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

## Training flags (`scripts/base_train.py`)

**Legacy single-site form** — exactly the v2 single-site special case (one
tabular site named `"coords"`: gate = beta, frozen direction = the store's
fixed orthonormal P, Qwen coord store as source):

```
--inject-coords <coords_dir>    # store dir: coords.int8 / index.npy / P.npy / meta.json
--inject-after-block 7          # inject right AFTER transformer.h[7] (default: 8 of 24 blocks)
--inject-beta 0.05              # gate: injected RMS as a fraction of residual RMS
--inject-noise-sigma 0.15       # loader-side gaussian noise on coords (deterministic per doc hash)
```

**General multi-site form**:

```
--inject-config path/to/config.json
```

```json
{
  "sites": [
    {"name": "coords", "r": 14, "after_block": 7, "gate": 0.05,
     "trainable_direction": false, "direction_seed": 1337},
    {"name": "free",   "r": 14, "after_block": 12, "gate": 0.1,
     "trainable_direction": true, "direction_init": "orthonormal", "optim": "muon"}
  ],
  "sources": {
    "coords": {"kind": "coords", "dir": "/workspace/coords", "noise_sigma": 0.15},
    "free":   {"kind": "coords", "dir": "/workspace/coords", "noise_sigma": 0.15}
  }
}
```

Site dicts are `injections.InjectionCfg` fields. `sources` keys must match the
site names; the JSON config supports `kind: "coords"` (a `CoordSource` store
dir); other sources (`FnActivation` for synthetic/control injections) are
programmatic. Sites may share a source, block, or neither. Absent both flags,
base_train is a byte-identical vanilla run (no extra params, no extra RNG
draws, stock dataloader).

## Optimizer contract

Wired in `GPT.setup_optimizer` (asserted by the param-count check there):

- **gates** (`_never_optimize`) go in **no** param group;
- **frozen directions** are skipped;
- **trainable directions** join an AdamW group by default (embedding-like map;
  betas `(0.8, 0.995)`), or a Muon group with `"optim": "muon"` per site;
- **weight decay is 0.0 for directions in both flavors** — the site normalizes
  the direction's scale away (`z / rms(z)`), so decay is a forward no-op that
  only shrinks the matrix toward the rms clamp. base_train's weight-decay
  scheduler skips the `injection`-tagged muon groups for the same reason.

After any `load_state_dict(..., assign=True)` (resume, eval load), call
`injections.reassert_optimizability(model.injection_sites)` — assign-loads
replace the Parameter objects and drop the gate's `_never_optimize` stamp.
base_train and `checkpoint_manager.build_model` already do this.

## Checkpoints

- Injected checkpoints carry `injection_sites.*` keys and an
  `injection_sites_config` (+ `injection_source_specs`) entry in the meta json;
  `checkpoint_manager.build_model` rebuilds the sites from meta so eval scripts
  load them (sites stay dormant — see below).
- **Warm-start from a vanilla checkpoint** into an injected run is allowed:
  only the `injection_sites.*` keys may be missing (they keep their fresh
  init). With frozen directions the optimizer param groups are identical to
  vanilla, so the optimizer state resumes too; trainable directions change the
  group structure and cannot resume a vanilla optimizer state.
- A vanilla model strict-loading an injected checkpoint fails loudly (by
  design — use `build_model`, which reattaches the sites first).

## Eval runs coords-off

Val bpb, CORE, and sampling all call the model with `acts=None`: **evaluation
is coords-off by design** (the model must not need the oracle to function).
Whether to add a coords-on eval pass is a separate, deliberate decision.

## Precompute pipeline (`scripts/precompute_coords.py`)

Produces the doc-hash-keyed int8 coord store the loader reads. Prereqs on
every pod: the **baseline run's tokenizer** at `$NANOCHAT_BASE_DIR/tokenizer`
(coord/token alignment is keyed to its exact merges) and the ClimbMix shards
at `$NANOCHAT_BASE_DIR/base_data_climbmix` (`python -m nanochat.dataset`).
The probe set json lives in the superproject (e.g.
`attribution/out/probe_set.json`). The encoder checkpoint: historically the
frozen Exp-A Qwen encoder (`best.pt`) — **its HF repo was deleted 2026-07-09**;
see the encoder-gap blocker in the status section for the per-layer-oracle
replacement path.

```bash
# 1) pod 0 fits continents PCA + the global coord scale ONCE (shared by all pods)
python -m scripts.precompute_coords --mode fit --encoder-ckpt <expA.pt> \
    --probe-set <superproject>/attribution/out/probe_set.json \
    --shards 0-3 --out /workspace/coords

# 2) every pod sweeps its round-robin shard slice (resumable, atomic per shard)
python -m scripts.precompute_coords --mode sweep --encoder-ckpt <expA.pt> \
    --probe-set <superproject>/attribution/out/probe_set.json \
    --shards 0-190 --out /workspace/coords --pod-index $P --n-pods $NP \
    --fast-forward --feeder-workers 8          # ~3.2x throughput, see below

# 3) after the fleet finishes, on ONE node with all per-shard files present:
python -m scripts.precompute_coords --mode merge-stats --out /workspace/coords
python -m scripts.precompute_coords --mode assemble --encoder-ckpt <expA.pt> \
    --probe-set <superproject>/attribution/out/probe_set.json \
    --shards 0-190 --out /workspace/coords
    # -> coords.int8 / index.npy / meta.json / P.npy
    # assemble HARD-FAILS on missing shards (--allow-missing-shards to override):
    # a partial store silently zero-coords the missing shards' docs.

# 4) *** MANDATORY pre-launch gate — never skip this ***
python -m scripts.precompute_coords --mode preflight \
    --shards 0-190 --out /workspace/coords --preflight-docs 1024
```

**Why preflight is mandatory**: it cross-checks the CONSUMER token path
(`RustBPETokenizer.encode(batch, prepend=bos)`, exactly as `coord_dataloader`)
against the assembled store, and hard-fails on tokenizer-contract drift or
token coverage < 99.9%. The failure it catches is the one that otherwise
**silently trains a baseline**: every lookup misses → all coords fall back to
zero → the injection no-ops on every token and nothing tells you.

Optional QA: `--mode verify` (recompute K docs live, assert int8 round-trip
within one quant step), `--mode measure-crossing` (prefix-mode crossing rate
for the qwen→nanochat tokenizer pair).

### Zero-fallback design note (do not "fix" this)

A doc missing from the store (or with a stored token count that mismatches —
tokenizer drift) gets **EXACT zero coords with NO noise**. The injection site
renormalizes any nonzero activation to full gate amplitude, so noised zeros
would inject pure noise at full strength on exactly the docs we know nothing
about. Exact zeros keep the injection a strict no-op there. The same reasoning
makes quantization **zero-preserving with no mean-centering**: a concept-free
token (raw coord 0) must stay int8 0 → dequant 0 → no-op. Per-column mean/std
ARE recorded in `meta.json` (required artifact), but only the single global
`scale` is applied by the loader.

### Noise design

`--inject-noise-sigma` gaussian noise is added by the **loader at train time**
(never baked into the store), seeded by `(train seed, doc content hash)`:
DDP-rank- and resume-independent, reproducible, and not memorizable as a
per-position identity.

### Fast-forward sweep (consumers of a mixed store, read this)

`--fast-forward` length-buckets segments **across docs** and packs each padded
forward to a token budget (`--max-batch-tokens`, `--seg-buffer`); with
`--feeder-workers N` the CPU-side tokenize/align runs in worker processes
(order-preserving, byte-identical output). bf16 GEMM results vary with batch
*shape*, so fast vs serial coords differ by fp noise: measured p99.9 of the
perturbation is **below** the sigma=0.15 training noise, and ≥99.9% of int8
values are within one quant step. The **zero-fallback positions are
bit-identical** on both paths (set structurally in `_gather`, independent of
batching). Stores mixing serial- and fast-swept shards are therefore fine;
the store format/index/meta are identical.

## Tests (CPU, no GPU / tokenizer / checkpoint needed)

```bash
python -m pytest tests/test_injection_sites.py tests/test_coord_lockstep.py \
                 tests/test_precompute_coords.py
# or standalone: python tests/test_coord_lockstep.py  etc.
```

- `test_injection_sites.py` — site invariants (RMS calibration, exact zero-row
  no-op, gate-0 no-op + zero direction grad, gate grads assigned but excluded
  from the optimizer split, state-dict keys, v1↔v2 forward equivalence) and the
  GPT wiring (acts=None ≡ vanilla; optimizer contract; step behavior).
- `test_coord_lockstep.py` — the ride-along loader against the REAL packing
  source (ast-extracted from `nanochat/dataloader.py`): bit-identical token
  stream, coord↔token alignment through best-fit + crops, exact-zero BOS /
  missing-doc rows even with noise on, deterministic noise, int8 round-trip.
- `test_precompute_coords.py` — the producer: phase-angle mapping (all 54
  one-hot concepts), PCA determinism, zero-preserving quantization, store
  assemble/read-back through the real `CoordSource`, pod-sharding coverage,
  byte→char offset reconstruction under adversarial UTF-8 splits, chunked +
  fast-forward flush equivalence, preflight drift detection, Welford
  merge-stats. Needs the superproject's `probe_set.json`
  (`$ORACLE_PROBE_SET` or `../attribution/out/probe_set.json`); skips
  otherwise.

Not validated on CPU (needs a real run): torch.compile + fp8 over the site
graph, CoordSource index throughput at the 27M-doc scale (~2-3 GB/rank), and
the precompute encoder wiring on the real Exp-A checkpoint + real
tiktoken/qwen tokenizer pair. SMOKE an injected launch first (3 steps, nothing
saved): step time should stay within ~3% of baseline and the per-site banner
must print the expected `r` / `after_block` / `gate` / docs count.
