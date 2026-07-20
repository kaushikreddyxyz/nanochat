# nanochat.injection — feature injection

## Status & review guide (2026-07-14)

History reads linearly: `main` → `experimental-setup` (the prior baseline-run
setup, 3 commits) → the injection work on top (`git log --oneline
pre-injection..HEAD`):

1. `9f9469b` — v1 coord-injection diffs applied verbatim (baseline for review)
2. `29fa9a2` — modules moved in as `nanochat.oracle`, imports normalized, align vendored
3. `96b79cc` — gpt.py routed through `InjectionSite` (v2), optimizer contract
4. `197bfdd` — tests + first README
5. `3adf4c8` — review-readiness README pass
6. `9b6129d` — **base_train back to stock** (byte-identical to `pre-injection`)
7. `c6d8f3d` — **package rename `nanochat.oracle` → `nanochat.injection`**, ActivationSource interface, activation-store-v2
8. `9d38414` — **`scripts/injection_train.py`** (dedicated injection training script)
9. README for the four directives
10. per-channel loudness calibration (`sites.py`), replacing `channel_weights`
11. **runtime probe-score injection** (`RuntimeProbeScoreSource` /
    `LiveProbeScoreSource`, positional ride-along join, `probe-scores-runtime`
    wiring; the `repackage-probe-scores` skeleton deleted)
12. **injection-stack amendments** — (1) alignment default = **overlap MAX**
    over covering gemma tokens (`align_policy: "max"|"mean"|"last"`) + opt-in
    **compact-tokens** mode (`compact.py`, `--compact-tokens`); (2) **staying-ahead**
    metrics + starvation monitor + startup throughput verdict
    (`activation_dataloader.py` stats, `injection_train.py`); (3) **rolling shard
    prefetch** (`prefetch.py`, config-driven, window + rollover, minimal disk);
    stale `nanochat/oracle/` leftover removed.
13. **video-player buffering / backpressure** (`buffering.py` +
    `activation_dataloader.py` `BufferControl`, `injection_train.py`): token-
    denominated **prefill** before step 1, a **free duty-cycle forecast**
    (replaces the startup verdict), **rebuffer with hysteresis** on a dry buffer,
    **DDP-coordinated** pause/resume, and **multi-stream** shard prefetch
    (`ShardPrefetcher.streams` + bandwidth auto-sizing).
14. `d97945b` — donor loudness from `loudness.json`
    (`attribution/measure_loudness.py`): startup-only resolution incl. live
    sources, concept-order refusal, DDP hash identity, resume-never-rescores.
15. **DOSE is the standard** (this commit): per-token magnitude survives
    (`/rms(z)` retired with the constant-amplitude path), per-event
    equalization, relu threshold at 2.0σ, unit-RMS direction rows, name-indexed
    concept subsets with a subset `L_ref`, a two-form `loudness` spec (dial /
    `abs:`) replacing the five-mode gate grammar, and a startup
    realized-loudness check. Defaults: `after_block=0`, dial `1.0`.

Suggested review order: `sites.py` (the injection contract + loudness calibration) →
`sources.py` (ActivationSource + store format + runtime probe sources + overlap
align + prefetch hook) → `compact.py` (compact-tokens wrapper) → `prefetch.py`
(rolling shard prefetcher, multi-stream) → `buffering.py` (prefill/rebuffer/
forecast/coordinate pure logic) → `gpt.py` (`setup_injection_sites`, forward
hook) → `scripts/injection_train.py` (align_policy/compact/prefetch/buffering
wiring) → `activation_dataloader.py` (positional join, `_ChunkProducer` +
`_pack_batches` + `BufferControl`) → `scripts/precompute_activations.py` →
`tests/`. All 7 test files + `python -m nanochat.injection.smoke` pass on CPU.

**Why the package is named `injection`, not `oracle`**: in this project's
terminology "oracle" is reserved for the *reliance failure mode* under study,
not for the machinery that injects features. Both families (geometric-manifold
`inject.py` and the contextual activation injection) live here.

**Blockers before an injected training run** (deliberate, not oversights):

- **Probe-score injection needs NO offline pass.** For a pre-scored corpus
  (`climbmix-scored`, shards 0–184, full coverage) `RuntimeProbeScoreSource`
  applies the scores at runtime — just stage/prefetch the score shards on the
  pod (~8.7 GB each, rolling window; see the runtime section). This is the ready
  path.
- **Qwen-encoder flavor still needs a store + the encoder.** The old
  `oracle-coords`/`-b` HF repos were deleted 2026-07-09, so that store must be
  precomputed (fleet pipeline below), AND the Exp-A checkpoint repo
  (`oracle-encoder`) was deleted 2026-07-09. What exists: per-layer oracles on
  `kaushikreddyxyz/oracle-encoders` (`layer06/08/14/best_stripped.pt`, head =
  `OracleMLPHead` 1024→4096→54 — a *different* head). Before a fleet run, point
  the precompute loader at a per-layer checkpoint via a small adapter (natural
  choice: `layer08`) or a surviving local Exp-A checkpoint. Experiment-design
  work, not a bug.
- GPU-side validations never ran on the real stack (end of this file). In
  particular the runtime probe path's real gemma-tokenize throughput vs training
  consumption is measured only on the CPU fixture so far.
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
| **loudness** | `channel_scale` (length-r) + `threshold` (`z0`, scalar or length-r). Resolved at startup by `calibrate_dose_gate` from the config's `loudness` spec; a site cannot be built with `channel_scale` unresolved. | **NEVER.** Parameters so autograd *assigns* them a gradient every backward (a loggable want-signal, per channel), but they sit in no optimizer group and are never stepped. |
| **activation** | the content: a `(B, T, r)` tensor per batch from a pluggable `ActivationSource`. | **NEVER.** Produced without grad by the dataloader and additionally `detach()`ed by the site. |
| **direction** | `(r, n_embd)` map from activation channels into the residual stream. | **The only optionally-trainable part**, controlled purely by freeze/unfreeze. Frozen + orthonormal init = "tabular" injection; unfrozen = "free" injection. |

Site math, per token:

```
a_eff = relu(a - z0)                     # z0 = threshold, default 2.0 (σ units)
u     = a_eff * channel_scale            # frozen, from calibrate_dose_gate
D_hat = D / rms_rows(D)                  # rows to UNIT RMS, inside forward
x     = x + rms(x).detach() * (u @ D_hat)
```

`rms(x)` is detached: it *measures* the stream to calibrate amplitude; it is not a
gradient path into the stream's own norm.

Defaults are the standard: `after_block=0`, `threshold=2.0`, `loudness=1.0`,
`trainable_direction=False`, `direction_init="orthonormal"`. A minimal site config is
`{"name": ..., "r": ...}`.

`GPT.setup_injection_sites(cfgs)` attaches sites as an `nn.ModuleDict` (checkpointed,
optimizer-visible); `GPT.forward(..., acts={name: (B,T,r)})` fires each site after its
own block. `acts=None` (eval, inference, vanilla runs) is bit-identical to a model
without sites.

### Why magnitude survives (no `/rms(z)`)

The retired math renormalized every token's projection to a fixed amplitude, so a 4σ
score and a 2.1σ score injected identically — the dose response was a step function at
the threshold. Probes are **linear** readouts of the donor's residual stream: a 4σ score
means 4σ of projection along that concept's direction. Injecting proportionally is
therefore the faithful transplant, and constant amplitude discarded the dose entirely.

**The retired formula**, for anyone reading a pre-2026-07-20 checkpoint's meta. Sites
had a `gate` (scalar, or a length-r vector whose channel mix was `gate/rms(gate)`) and
injected `x + gate * rms(x).detach() * (a @ D) / rms(a @ D)`. The `/rms(z)` forced every
token that passed the source-side `present_z` row gate to inject at *identical*
amplitude, making the dose response a step function; loudness was a `gate` targeting the
54-concept `subspace_total`. Configs used a five-mode gate grammar
(`dial` / `abs:` / `auto[:t]` / `donor[:stat]` / explicit vector). All of it is deleted —
old checkpoints are not loadable by this code, and their results are not comparable to a
rerun.

### Per-event equalization

`channel_scale_c ∝ 1/E_c`, where `E_c` is the RMS of channel `c` **conditional on that
channel firing** (`calibrate_dose_gate`). Every concept is equally loud *when it fires*;
a rare concept stays rare but is audible at the same volume as a common one. Equalizing
per *total* activation mass instead would amplify rare concepts to compensate for
rarity — a different experiment. One global rescale then sets the median injected
loudness over firing tokens to the resolved target.

Consequence, stated plainly: **cross-concept amplitude ordering is intentionally not
meaningful.** The site says "this concept is present, this strongly" per concept; it does
not preserve which concept is natively louder than which.

The calibration target is the **total per-token injected loudness**
(`rms(u @ D_hat)` per token, medianed over firing tokens on the real firing pattern), not
a per-channel budget. Co-firing therefore self-corrects: when `n` concepts fire on the
same token — the multi-family case, e.g. a date token that is winter + january + friday —
the per-channel scale comes out ~`1/sqrt(n)` smaller and the injected total still lands
on target, instead of `n`-ing up. Multi-family needs no new machinery.

### What the calibration sample is allowed to conclude

Under-measured channels get the **median well-measured `E_c`** imputed (unit-consistent —
still a `1/E`), logged loudly with the count. A channel counts as under-measured when it
fires in fewer than `--dose-min-channel-docs` **distinct source documents** (default 8)
*or* fewer than `--dose-min-events` **tokens** (default 50). Documents are the load-bearing
test: concepts are bursty, so one weekday-heavy document can fire hundreds of times and
clear a per-token floor on a scale that actually rests on a single document. The token
count stays in `meta["n_events"]` as a diagnostic.

A channel that never crosses the threshold in the sample is a **hard startup error**.
`channel_scale = 0` means that concept injects nothing for the entire run — an `r=7` site
silently becomes `r=6`, which is not the experiment that was configured, and the realized
check medians across channels so one dead channel barely moves p50. Pass
`--allow-dead-channels` to accept it deliberately.

The sample itself is drawn from the **lowest `--loudness-calib-shards` shards** (default 4,
floored at 2) rather than striped across every configured shard. Calibration runs *before*
the shard prefetcher attaches, so every shard it opens is a full multi-GB download against
the primary repo on every rank's node, and a shard living in an overflow repo 404s. Striping
`ceil(k/n)` docs over a handful of shards keeps the `--loudness-k` doc budget while touching
a fixed, low, primary-resident set; drawing from ≥2 shards keeps one shard's idiosyncrasy
from setting the run's scale. `injection_train.py` refuses at startup if a source's
calibration shards map outside the primary repo under its `prefetch.per_repo`.

### Threshold semantics

`relu(a - z0)`. Negative probe scores are the **noise tail of a linear readout, not
evidence of an anti-concept**, so they inject nothing rather than a negated direction.
Sub-threshold, zero and negative rows inject **bitwise 0** — no branch, no NaN,
torch.compile-friendly.

`threshold` is a scalar (default `2.0`) **or a length-r vector**, so per-concept knees are
expressible without a schema change — probes differ in tail shape and calibration
quality. Calibration applies the same per-channel knee it hands the site.

The site owns thresholding. A source-side `present_z` row gate at the same value is
idempotent with this relu (row-max ≥ z0 ⟺ some channel survives), so an old config that
double-thresholds is harmless; a *higher* `present_z` would clip dose events. Prefer
`present_z=0`.

### Unit-RMS rows: a gauge fix

`D_hat` is reparameterized inside `forward` (not by a post-step hook), so it is
checkpoint-safe and self-enforcing. Scaling D's rows by any positive factor leaves the
injection unchanged — loudness lives in `channel_scale` alone and **cannot be trained
into D**. RMS rather than L2 so `channel_scale[c]` reads directly as a fraction of
residual RMS, with no `sqrt(n_embd)` factor. Row normalization only: the gram is **not**
constrained, so a manifold direction's deliberate off-diagonal structure (the sphere
arm's cosines up to 0.82) survives untouched.

### Loudness: two forms

`loudness` (cfg field; `--loudness` on the trainer) takes exactly two forms:

- **plain number = dial in donor units** (default `1.0`). `dial 1.0` ⇒ the median firing
  event injects at gemma's own median **per-concept** active loudness for *this site's*
  concepts. Requires `loudness.json` + a source gemma layer.
- **`abs:<n>` = absolute target** for that same median, as a fraction of residual RMS.
  No `loudness.json` needed — the escape hatch for sources with no gemma identity
  (`FnSource`, a Qwen-encoder store).

**`L_ref` is the subset median, not `subspace_total`.** Real sites are subsets of the 54
concepts (seasons r=4, weekdays r=7). The reference is
`median(ridge.active_loudness[L].p50[c])` over the site's own concepts.
`subspace_total.ridge[L].p50` is the *whole 54-concept packet*; concepts add in
quadrature, so using it would over-inject an r-concept site by ~`sqrt(54/r)` (≈3.7x at
r=4). The subset median reproduces the hand-calibrated gates the campaigns used
(weekdays 0.0273, seasons 0.0262). Both numbers are logged at startup.

Concepts are matched to `loudness.json` **by name**, so column order is irrelevant by
construction (a stronger permutation guarantee than an order check) and an unknown or
duplicated name is a hard error.

### Alignment caveat (measured, not assumed)

Calibration samples *pre-alignment* gemma rows; the site sees *post-alignment* nanochat
rows, so the pool op moves the realized dose. Measured on a synthetic source (fires at
4σ, `z0=2σ`, 50% of tokens per channel — a deliberately dense fixture):

| gemma tokens pooled per nanochat token | mean: vs target | mean: correction | max: vs target | max: correction |
|---|---|---|---|---|
| 1 (no pooling) | +0.0% | 1.00 | +0.0% | 1.00 |
| 2 | −29.3% | 1.41 | +22.5% | 0.82 |
| 4 | −50.0% | 2.00 | +41.4% | 0.71 |
| 8 | −64.6% | 2.83 | +41.4% | 0.71 |

The two errors have **opposite signs and different causes**. `mean` is *dilution*: relu
is convex, so `relu(mean(z)-z0) ≤ mean(relu(z-z0))` and a firing token pooled with
sub-threshold neighbours is dragged under the knee. `max` loses no peak at all —
`relu(max(z)-z0) == max(relu(z-z0))` per channel — but it *concentrates* co-firing:
pool two gemma tokens that fired on different channels and one nanochat row now has
both live, and loudness adds in quadrature. That residual scales with firing density,
so at the realistic rates the real sites run at it is small while `mean` is
catastrophic (same fixture, magnitudes jittered ±50%):

| per-channel firing rate | row-active frac | depth 2 mean / max | depth 4 mean / max | depth 8 mean / max |
|---|---|---|---|---|
| 0.50 | 0.94 | −33.7% / +32.9% | −56.6% / +68.5% | −69.6% / +99.8% |
| 0.20 | 0.59 | −75.7% / +19.0% | −83.4% / +53.4% | −89.8% / +100.3% |
| 0.05 | 0.18 | −81.7% / +4.5% | −87.3% / +13.1% | −93.3% / +32.5% |
| 0.02 | 0.08 | −82.3% / +1.0% | −87.0% / +3.9% | −95.6% / +11.6% |

(The weekday campaign measured a row-active fraction of 0.117, i.e. the bottom two rows.)

`injection_train.py` therefore runs a **realized-loudness check** on the real stream at
startup: it peeks batches (replayed into training — no data skipped), logs the realized
ladder beside the calibration and donor ladders, and WARNs with the exact correction
factor when realized p50 misses the target by more than `--dose-check-tol` (default 25%).
It never auto-corrects — the dial is the user's decision. Skipped on resume.

Beside the ladder it logs the **pooling-depth distribution** (p50/p90/p99/max and a
histogram of how many gemma tokens each nanochat token pools, plus the fraction covered by
no gemma token at all). Depth is the *mechanism* the ladder is the *symptom* of: under
`max` both the co-firing concentration and the raised false-firing floor
(`P(fire) = 1 − Φ(z0)^depth`) grow with it, under `mean` the dilution does. Depth 1
everywhere means neither concern is live for this tokenizer pair; a fat tail means read
the deviation as real. It is accumulated during alignment, so it costs nothing.
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
row to full injection amplitude, so noised zeros would inject pure noise at full
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
- **`RuntimeProbeScoreSource`** (`kind: "probe-scores-runtime"`) — gold gemma
  probe scores (one layer's 54 standardized scores, per the binding
  one-layer-per-model rule) applied per nanochat token **at runtime, nothing
  stored offline** (design below). The primary probe-score path.
- **`LiveProbeScoreSource`** — same interface, scores computed live by an
  injected `score_fn` callable (bounded stub; real gemma wiring is a follow-up).
- **`ProbeScoreSource`** (`source_kind: "probe-scores"`) — reader for a
  pre-built probe-scores v2 store; kept for completeness, superseded by the
  runtime path (which needs no offline pass for a pre-scored corpus).
- **`FnSource`** — arbitrary callable `fn(text, n_tokens) -> (n_tokens, r)`
  for synthetic/control injections.
- `open_store(dir, ...)` dispatches STORE readers on `meta.json["source_kind"]`;
  the runtime sources are not stores — injection_train constructs them directly.

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

### Runtime probe-score injection (no offline pass)

Gold gemma probe scores are applied per nanochat token **at runtime, nothing
stored offline** — replacing the old repackage-to-a-store idea. Two backends,
one `ActivationSource` interface (`nanochat/injection/sources.py`):

**Stored-scores backend (`RuntimeProbeScoreSource`, primary, fully working).**
The scores already exist per gemma token in `hf.co/kaushikreddyxyz/climbmix-scored`
(+`-overflow`, `-overflow-2..7`): `scores_<sid>.npy` int8 `[n,3,54]` (axis1:
0=L6,1=L8,2=L14), `docs_<sid>.jsonl` `{doc,start,n}` spans (row order, full
coverage), plus `quant.json`/`corpus_stats.json`/`columns.json`. Per doc:

- **Positional join, no hashing, no startup walk.** The ride-along loader
  re-runs the real corpus enumeration, so it knows each doc's `(shard, row)`;
  `docs_<sid>.jsonl[row]` gives `(start, n_gemma)` into the score memmap. The
  loader tracks `(shard, row)` from `_document_batches`' `(pq_idx, rg_idx,
  epoch)` state plus a per-row-group cursor and parquet row-group metadata
  (cheap); `docs_<sid>.jsonl` and the score memmap load lazily per shard. (A
  text-keyed hash-index fallback exists — `build_hash_index`/`index_path`,
  needs `climbmix_dir` — for callers without the position; it *does* pay the
  parquet-walk startup cost.)
- **On-the-fly alignment (overlap, MEAN by default).** Retokenize the doc with
  the gemma fast tokenizer (`add_special_tokens=False`, `len == n_gemma` guard)
  for char offsets, nanochat byte→char offsets via `align.nanochat_char_offsets`,
  then map each nanochat token to the gemma tokens whose **char span OVERLAPS**
  it. A nanochat token nested inside one big gemma token inherits that token's
  score (broadcast); a nanochat token spanning several gemma tokens pools them.
  `align_policy` (source config, **default `"max"`**) picks the pool op. None of
  the three re-standardizes.
  - **`"max"`** (default, standard) takes the **per-channel maximum** over the
    covering z-scores. The site applies `relu(a − z0)`, so the question a span
    answers is whether it **contains** the concept, not how concept-y it is on
    average; max is the pool that makes those two the same question. Concretely,
    covering z-scores `[4.0, 0.1]` give 4.0 under max (`relu(·−2)` = 2.0) but 2.05
    under mean (`relu(·−2)` = 0.05, 40× quieter), and a lone firing token keeps its
    full value at any pool depth. Implemented with `np.maximum.reduceat` over
    interleaved range starts/ends.
  - **`"mean"`** averages the covering z-scores. Standardized inputs, so averaging
    over *k* tokens shrinks variance — which also **dilutes peaks**, and is why it
    is no longer the default (table above).
  - **`"last"`** keeps only the rightmost covering gemma token (the historical
    behavior for the multi-gemma→one-nano direction), for per-experiment override.

  **The cost of max: a raised noise floor.** The maximum of *k* draws is biased
  upward (≈`sqrt(2 ln k)` for gaussians), so a nanochat token covering many gemma
  tokens is likelier to cross `z0=2.0` on background alone. Measured on an N(0,1)
  source (exact: `1 − Φ(2)^k` for max, `1 − Φ(2√k)` for mean):

  | pool depth *k* | mean: P(fire) | max: P(fire) |
  |---|---|---|
  | 1 | 0.0228 | 0.0228 |
  | 2 | 0.00234 | 0.0450 |
  | 4 | 0.0000317 | 0.0879 |
  | 8 | ~8e-9 | 0.1682 |

  So max roughly doubles the per-channel false-firing rate every time pool depth
  doubles, where mean drove it to zero. Mean's "low false-firing rate" was never a
  feature — it came from suppressing true firings just as hard (the loudness table
  above). Max buys peak fidelity and pays in background events; at *r*=7 channels
  and depth 4, 47.5% of pure-noise rows have at least one channel over threshold.

  A nanochat token that overlaps **no** gemma token (a char the gemma tokenizer
  dropped) stays an **exact zero row**. Overlap is causal except in the genuine
  broadcast case (a coarse gemma token legitimately covering a finer nanochat
  token) — there is no finer signal to use there.

- **Compact-tokens mode (opt-in, `--compact-tokens`, OFF by default).** A wrapper
  tokenizer (`injection/compact.py`, `CompactGemmaTokenizer`) cuts nanochat tokens
  at gemma boundaries so every nanochat token nests inside exactly one gemma token
  (gemma `"XYZ ABC"` → `"XY","Z"," ","A","BC"`); alignment is then trivially 1:1
  (mean ≡ last). **Two loud caveats:** (a) it **CHANGES the training token
  stream** — breaking BPE merges at gemma edges inflates the token count (measured
  **1.097×** on the compact-test fixture doc set), so a compact run is **not
  bit-comparable to a standard baseline**; (b) it needs gemma char-offsets in the
  hot loader path. **Wiring gap (documented, not a bug):** the wrapper currently
  gemma-tokenizes for segmentation and the source re-tokenizes for alignment (two
  gemma passes per doc); sharing the offsets through the loader — and, since
  compact makes alignment 1:1, skipping the source's gemma retokenize entirely —
  is a follow-up. The default path stays standard nanochat tokenization +
  overlap-max alignment; core nanochat modules are untouched (the ride-along
  loader already accepts an injected tokenizer).
- **Dequantize + standardize** one layer's columns (`layer` config field,
  default 8; default all 54 in `columns.json` order, or a `concepts` subset)
  with the frozen `quant.json` (`raw = int8·scale + zero`) and
  `corpus_stats.json` (`z = (raw−mean)/std`). Unknown shard / row-out-of-range /
  tokenizer drift → `None` (loader maps to exact zeros, no noise; counted in
  `.stats()`). Note: standardized scores are z-scores, so covered tokens carry a
  nonzero signal on *every* token (unlike the qwen-encoder ring coords, which
  are zero on no-concept tokens).

All per-doc work runs in the loader worker path, so it overlaps training;
`--lookup-workers N` runs the per-doc gemma-tokenize+align in an ordered thread
pool (the `(shard,row)` cursor is assigned serially first, so the parallel work
is order-independent and byte-identical to serial).

**Live-gemma backend (`LiveProbeScoreSource`, bounded stub).** Same interface;
its constructor takes a `score_fn(texts) -> [ (n_gemma, L, 54) raw float, … ]`
callable (tests inject a stub; real gemma wiring is a documented follow-up), then
slices+standardizes+aligns identically. **Cost math:** a gemma-2-2b forward is
≈5.2 GFLOP/tok vs ≈11 GFLOP/tok to train d24, and one H100 scores ≈40k tok/s —
so live scoring at pretraining throughput needs a scorer fleet *larger* than the
trainer. Intended for small runs, evals, and unscored corpora; for a pre-scored
corpus the stored backend is free at runtime.

### Rolling shard prefetch (ops, `injection/prefetch.py`)

The score shards are ~8.7 GB each, so a real run downloads them in background
threads **while training runs** rather than bulk pre-downloading. `ShardPrefetcher`
keeps a small window (`ahead`, default 2) of shards staged beyond the consumption
frontier and **deletes consumed shards behind** (`keep_behind`, default 1) so disk
stays at ~2 shards. Consumption is monotonic (the loader enumerates shards in
corpus order), so `ensure(sid)` blocks only if the window has fallen behind — and
that block is exactly what the staying-ahead machinery below reports. HF downloads
set `HF_HUB_DISABLE_XET=1` (xet stalls on pods). When a shard is evicted the source
drops its cached memmap first (the `on_delete` hook), so a delete never races an
open memmap. A **local score dir with everything present is a no-op passthrough**
(no prefetcher, no thread).

**Multi-stream (`streams`, `set_streams`).** One 8.7 GB shard covers ~50 s of
full-speed training but takes ~45–210 s to fetch depending on NIC, so a single
stream can't keep the window ahead. `ShardPrefetcher` runs `streams` worker
threads; each **reserves** the nearest un-staged in-window shard under the lock
before fetching (so siblings pick different shards, never the same one), and the
inline `ensure` fallback adds at most one catch-up fetch. If `streams` isn't set
explicitly, `injection_train` **sizes it from the prefill bandwidth measurement**:
`buffering.size_prefetch_streams(shard_bytes, measured_MB/s, cover_seconds)` where
`cover_seconds = --tokens-per-shard / (per-rank consumption × world)` — ranks
stride row groups *within* a shard, so every rank crosses shard boundaries at the
**global** pace and each rank downloads every shard — logging the one-line arithmetic
(`prefetch sizing: 8.7GB / 90MB/s = 97s download vs 50s/shard training (~1.9x) =>
2 streams, ahead=3`), calling `set_streams` (workers only grow) and widening
`ahead` to match. Omit
`--tokens-per-shard` to skip auto-sizing and honor `--download-streams` / a
configured `streams`.

Wired per runtime source via a `"prefetch"` block in the source spec (else off):

```json
"probes": {"kind": "probe-scores-runtime",
           "score_shards_dir_or_repo": "kaushikreddyxyz/climbmix-scored",
           "shards": "0-184", "layer": 8,
           "prefetch": {"staging_dir": "/workspace/scores_staging",
                        "climbmix_dir": "/workspace/climbmix",
                        "repos": ["kaushikreddyxyz/climbmix-scored",
                                  "kaushikreddyxyz/climbmix-scored-overflow"],
                        "per_repo": 25, "ahead": 2, "keep_behind": 1, "streams": 3}}
```

`repos`/`per_repo` give the count-based shard→repo assignment (25 shards/repo, as
the scorer wrote them); omit them to pull every shard from
`score_shards_dir_or_repo`. `streams` is optional (auto-sized when absent). Store
metadata (`columns/quant/corpus_stats.json`) is read once at init from the repo;
only per-shard files roll through `staging_dir`.

### Buffering (video-player prefill / rebuffer, `injection/buffering.py`)

The activation source must keep up with training, so consumption is decoupled from
production behind a **token-denominated buffer** (like a video player pre-buffering
then rebuffering on a stall). A single background producer thread runs the
tokenize+lookup+align (`_ChunkProducer.next_chunk`) into a bounded deque; the
best-fit packer draws chunks from it (`BufferControl`, in
`activation_dataloader.py`). Packing is **byte-identical** to the synchronous
`acts_data_loader_with_state` (same chunks, same best-fit — a test pins it), so a
buffered run is bit-comparable to a baseline. All targets are **per rank** (the
loader feeds one rank).

- **Prefill.** Before step 1 the buffer warms to `--prefill-tokens` (default `4×`
  the per-rank per-step tokens = `total_batch_size / world`). `wait_prefill` blocks
  the trainer until then; a tty-aware progress bar (carriage-return on a tty, plain
  periodic lines off it; DDP rank 0 only) shows percent + ETA.
- **Free duty-cycle forecast** (replaces the old startup sampling verdict). Production
  rate is measured *during* prefill — already producing — so the forecast costs
  nothing extra. After warm-up the script prints one line:
  `activation production ~2.4x consumption — forecast duty cycle ~100%`, or when the
  source lags: `~0.4x consumption — forecast duty cycle ~40%, consider more
  --lookup-workers / download streams`. Consumption =
  `total_batch_size / world / --target-step-time`. `--starvation-abort` hard-fails a
  forecast below 1× (otherwise it is advisory).
- **Rebuffer with hysteresis.** Mid-training, if the buffer falls below
  `--dry-tokens` (default one per-rank batch) it counts as **dry**: consumption pauses
  and the producer refills to `--rebuffer-tokens` (default `prefill/2`) before
  resuming — deliberately past the dry mark so a source hovering at the dry line
  can't thrash (`dry < rebuffer ≤ prefill`, `BufferState`). A loud one-time banner,
  then a progress line every few seconds
  (`buffering 42% (~35s) — downloading shard 71 / aligning`).
- **DDP-coordinated.** Independent per-rank stalls amplify at allreduce, so ranks
  pause/refill/resume **together**: each step boundary does one cheap
  `all_reduce(MAX)` of a buffer-low int (`coordinate_rebuffer`, a pure function
  unit-tested with simulated ranks) — if ANY rank is dry, all rebuffer, and a
  `barrier` after the refill keeps the resume in lockstep. `--buffer-check-every`
  spaces the check (default every step).
- **Backpressure.** Production blocks once the buffer reaches `--buffer-max-tokens`
  (default `2× prefill`) — never buffer the whole "video".

### Staying-ahead metrics (Amendment 2)

`_ChunkProducer` / `BufferControl` update a cheap `stats` dict (no hot-path overhead
when unused); `injection_train` reads it:

- **Step log** adds `act_wait: <ms> (<%>)` — cumulative time-blocked-waiting-on-
  activations this step (the wall time inside `next(train_loader)`) as a fraction of
  step time — plus `qdepth: <n>` (docs staged ahead of the packer) and `buftok: <n>`
  (live buffer depth in tokens). wandb logs `train/act_wait_ms`,
  `train/act_blocked_frac`, `train/act_queue_depth`, `train/buffer_tokens`.
- **Starvation policy**: over `--starvation-window` steps, if the blocked fraction
  exceeds `--starvation-threshold` (default 0.15) it **warns loudly** (throttled);
  `--starvation-abort` hard-fails instead (and also fails a sub-1× startup forecast).
  Alignment on the CPU fixture (fake tokenizers) is ~20–30k docs/s single-worker; in
  production the **gemma retokenization dominates**, so size `--lookup-workers` to the
  real per-doc cost and let the monitor confirm.

## Training (`scripts/injection_train.py`)

**base_train is stock again**: `scripts/base_train.py` was designed for
non-injection models and is byte-identical to the `pre-injection` tag —
vanilla runs have zero injection surface. `scripts/injection_train.py` is the
injection script: a deliberate fork whose shared body is kept byte-identical
to base_train (so `diff scripts/base_train.py scripts/injection_train.py`
shows only the injection hunks — keep it that way when either changes). The
old `--inject-coords/--inject-beta/...` flag family is gone.

**Single tabular site from a store** (the v1 recipe — note `abs:0.05`: a qwen
store has no gemma-layer identity, so the dial doesn't apply and a plain number
would hard-error):

```
python -m scripts.injection_train -- --activation-store <store_dir> \
    --after-block 7 --loudness abs:0.03 --noise-sigma 0.15
```

One site named `"acts"`: frozen direction pinned to the store's `P.npy` when
present (else seeded orthonormal via the store's `p_seed`), source opened by
`meta.json` kind. `--loudness` grammar (dial / `abs:` — see the loudness section
above); `--loudness-k`/`--loudness-min-docs`/`--loudness-calib-shards` tune the
sampled calibration, `--loudness-json` points the dial at a specific artifact,
`--lookup-workers N` overlaps runtime scoring with training.

`--loudness`, `--after-block` and `--noise-sigma` belong to THIS branch only: the
multi-site form below carries them per site, so passing a non-default value
alongside `--activation-config` is a hard error rather than a silent no-op.

A `file:` direction whose npz declares its own row order (`store_order` or
`weekday_store_order` — both spellings exist in the committed artifacts) is
hard-checked against the site's `concepts` list at startup. Row *i* of `D` is
concept *i*, so reordering a config's `concepts` would otherwise silently permute
the geometry.

**General multi-site form**:

```
python -m scripts.injection_train -- --activation-config path/to/config.json
```

```json
{
  "sites": [
    {"name": "acts", "r": 14, "after_block": 7, "loudness": "abs:0.03",
     "trainable_direction": false, "direction_seed": 1337},
    {"name": "probes", "r": 54, "after_block": 8, "loudness": 1.0,
     "trainable_direction": true, "direction_init": "orthonormal", "optim": "muon"}
  ],
  "sources": {
    "acts": {"kind": "qwen-encoder", "dir": "/workspace/acts_qwen", "noise_sigma": 0.15},
    "probes": {"kind": "probe-scores-runtime",
               "score_shards_dir_or_repo": "/workspace/climbmix-scored",
               "shards": "0-184", "layer": 8, "noise_sigma": 0.15}
  }
}
```

Site dicts are `sites.InjectionCfg` fields (`loudness` follows the grammar: a
plain number is the donor dial, `"abs:<n>"`/`"auto[:target]"` are absolute, a
length-r list is an absolute per-channel vector); `sources` keys must match site
names; `FnSource` and `LiveProbeScoreSource` remain programmatic. The `probe-scores-runtime`
source takes `score_shards_dir_or_repo` (local dir or HF dataset repo),
`shards`, `layer`, optional `concepts` subset / `climbmix_dir` /
`index_path` / `build_hash_index`, `align_policy` (`"max"` default / `"mean"` / `"last"`),
and a `prefetch` block (rolling shard prefetch, above). Exactly one of
`--activation-store`/`--activation-config` is required. The startup banner
prints each site's source kind, doc coverage, and resolved (incl. calibrated)
gate.

## Optimizer contract

Wired in `GPT.setup_optimizer` (asserted by the param-count check there):

- **loudness params** (`channel_scale`, `threshold`; `_never_optimize`) go in **no** param group;
- **frozen directions** are skipped;
- **trainable directions** join an AdamW group by default, or Muon with
  `"optim": "muon"` per site;
- **weight decay is 0.0 for directions in both flavors** — the site normalizes
  the direction's scale away (`z / rms(z)`), so decay only shrinks the matrix
  toward the rms clamp. injection_train's wd scheduler skips the
  `injection`-tagged muon groups for the same reason.

After any `load_state_dict(..., assign=True)` (resume, eval load), call
`sites.reassert_optimizability(model.injection_sites)` — assign-loads replace
the Parameter objects and drop the loudness params' `_never_optimize` stamp.
injection_train and `checkpoint_manager.build_model` already do this.

## Checkpoints

- Injected checkpoints carry `injection_sites.*` keys and
  `injection_sites_config` (+ `injection_source_specs`) in the meta json;
  `checkpoint_manager.build_model` rebuilds the sites from meta so eval
  scripts load them (sites stay dormant — eval is activations-off).
- A **resume** reuses the persisted `channel_scale` and never re-runs calibration
  (loudness.json / the shard set may have moved since). The checkpoint's
  `gate_calibration` record is carried forward into the resumed run's provenance,
  so later checkpoints keep the dial / `L_ref` / layer / dead-channel count the
  surviving vector was actually fit under.
- Persisted calibration is keyed by **site name**. Renaming a site — or warm-starting
  from a checkpoint whose meta has no `channel_scale` — therefore recalibrates
  *fresh* against today's artifacts and may not reproduce the vector the checkpoint
  was trained with. `injection_train.py` says so loudly rather than logging it as a
  normal first calibration.
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
                 tests/test_precompute_activations.py tests/test_runtime_probe_source.py \
                 tests/test_compact_tokens.py tests/test_shard_prefetch.py tests/test_buffering.py
python -m nanochat.injection.smoke
```

- `test_injection_sites.py` — config defaults, direction init (incl. `file:`
  against the real committed manifold), module plumbing, freeze/unfreeze + the
  optimizability contract, the GPT wiring (acts=None ≡ vanilla; optimizer
  contract; step behavior), the two-form loudness grammar, `abs:` needing no
  gemma identity, loudness.json discovery + loud fallback, layer requirement,
  calibration-precedes-training, name-indexed concepts, DDP hash identity.
- `test_injection_dose.py` — the site math and its calibration: exact linear
  dose response, bitwise-zero no-op (sub-threshold/zero/negative), the D row
  gauge, the sphere gram surviving row normalization, PER-EVENT equalization,
  `min_events` imputation, dead channels, subset name-indexing, the subset
  `L_ref`, checkpoint round-trip, and the realized-loudness check (including
  peek/replay skipping no data).
- `test_runtime_probe_source.py` — the runtime probe path: `lookup_by_row`
  dequant+standardize+**overlap-align** for BOTH policies (hand-checked
  multi-gemma→one-nano, one-gemma→multi-nano **broadcast**, unmapped→exact-zero,
  drift/miss→None; default is `"mean"`), positional-join == content-hash-join,
  the ride-along loader's `(shard,row)` cursor driving `lookup_by_row`,
  `lookup_workers` threaded == serial, `LiveProbeScoreSource` stub (mean + last),
  single-worker alignment throughput, and the **staying-ahead loader stats**
  (produce time/tokens, queue depth) under a simulated slow source.
- `test_compact_tokens.py` — compact mode: standard tokenization straddles a
  gemma boundary but compact never does (nesting), the token-count inflation
  number, and that compact tokenization makes the overlap alignment 1:1
  (mean ≡ last).
- `test_shard_prefetch.py` — the rolling prefetcher with a FAKE fetcher: window
  stages ahead + deletes behind + stays bounded (not bulk), `on_delete` evict
  hook, block+`on_wait` when the window is behind, local-dir no-op passthrough,
  and **multi-stream** (concurrent fetches bounded by `streams` + one inline
  catch-up, no double-fetch, order preserved; `set_streams` grows the pool).
- `test_buffering.py` — the video-player buffering layer: `BufferState`
  hysteresis (prefill→running→rebuffer→running, stays paused past dry to
  rebuffer), `coordinate_rebuffer` all-or-none across simulated DDP ranks
  (incl. only-one-rank-low), `duty_cycle_forecast` / `rebuffer_progress` /
  `size_prefetch_streams` math, `BufferControl` (prefill fills to target BEFORE
  the first pull, dry detected, `do_rebuffer` refills to target), and the
  buffered loader packing **byte-identically** to the synchronous loader.
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
graph, the runtime probe path's real gemma-tokenize throughput vs training
consumption (fixture only so far), the qwen-encoder store index throughput at
the 27M-doc scale, the precompute encoder wiring on a real checkpoint + real
tokenizer pair, and the buffering layer under real load — the DDP-coordinated
rebuffer collective and the multi-stream download bandwidth auto-sizing are
CPU-tested with fakes/simulated ranks but never against real DDP or a real NIC.
SMOKE an injected launch first (a few steps, nothing saved): step time within
~3% of baseline and the per-site banner must print the expected `r` /
`after_block` / `loudness` / docs count.
