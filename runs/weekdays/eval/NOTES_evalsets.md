# Weekday-geometry eval sets + runners — design notes

New files under `runs/weekdays/eval/` (no existing source edited, nothing
committed):

| file | purpose |
|------|---------|
| `weekday_evalset.py` | deterministic generator for the weekday-knowledge eval set |
| `evalsets/weekday_v1.jsonl` | generated set, **422 items** (regenerate: `python runs/weekdays/eval/weekday_evalset.py`) |
| `run_evals.py` | pod-side on-vs-off orchestrator (weekday / val-bpb / CORE) |
| `test_evalset.py` | CPU, no-network, plain-assert tests (14, all green) |

All model forwards go through the **pinned** `runs/weekdays/eval/harness.py`
(sibling agent) — `load_model`, `GemmaScorer.score`, `build_acts`,
`forward_metrics`, `ce_report`. On-vs-off = `gate_scale` 1.0 vs 0.0 through the
same code path. Because the injection site renormalizes `z/rms(z)`, scaling the
activations is a **binary** on/off (0 → exact no-op; any nonzero → full via the
renorm), so `gate_scale ∈ {0.0, 1.0}` is exactly the on/off knob and intermediate
values are not meaningful loudness dials.

---

## 1. Eval set (`weekday_v1`)

Completion-style, SHORT prompts (no chat/instruction format — these are 124M-param
1.3B-token base models). Scoring = length-normalized CE over each option's
continuation; prediction = argmin-CE option. Report accuracy + mean answer-CE +
margin (2nd-best − best), per category and overall.

Item counts (total **422**):

| category | n | what it probes | example |
|----------|---|----------------|---------|
| `order_next` | 70 | +1 cyclic step | "The day after Monday is" → Tuesday |
| `order_prev` | 70 | −1 cyclic step | "The day before Monday is" → Sunday |
| `order_k` | 168 | ±k step, **wraps** (k=2..4) — the cyclic-structure test | "Two days after Monday is" → Wednesday |
| `fact_position` | 36 | position-in-week, convention stated in-prompt | "If the week starts on Monday, the second day of the week is" → Tuesday |
| `usage_weekend` | 12 | weekend membership (pair + MC classification) | "The two days of the weekend are Saturday and" → Sunday |
| `usage_context` | 42 | day association with a self-contained in-prompt fact | "In a town where the market is held every Monday, the market day is" → Monday |
| `sanity` | 24 | trivial non-weekday calibration | "On a clear day the sky is" → blue |

Design guarantees (all enforced by `test_evalset.py`):

- **Deterministic**: `generate_items()` is a pure function of the templates; JSONL
  bytes are identical every run; the committed file is checked fresh.
- **Balanced days**: over the four fully-balanced categories (`order_next`,
  `order_prev`, `order_k`, `usage_context`) each weekday is the answer **50×**.
  `fact_position` excludes position 1 (it would equal the named start day →
  leak), so it is mildly day-imbalanced and is excluded from the strict balance
  assertion. `usage_weekend`/`sanity` are semantically day-specific and also
  excluded.
- **No answer leak**: every item carries `meta.fill_in`. For fill-in items
  (inference: all `order_*`, `fact_position`, weekend pair-completion, `sanity`)
  the answer word never appears in the prompt (whole-word check). `fill_in=False`
  is set only where a day legitimately appears in the prompt by design —
  association/copy (`usage_context` "market held every Wednesday → Wednesday",
  the task's own example) and weekend-MC (options listed in-prompt). `usage_context`
  also includes one **inference** template (`ctx_setup`: "…traders set up the
  evening before, which is" → *previous* day) that is fill-in and leak-free.
- **Well-formed options**: answer ∈ options, options unique, ≥2 options; day-answer
  categories present the full 7-day option list (1/7 chance baseline); each
  reasoning category has ≥3 surface templates so it is not one memorizable pattern.

Multiple surface templates per category (e.g. 10 for `order_next`, 8 phrasings ×
2 directions × 3 k for `order_k`) keep the set from being a single string pattern.

---

## 2. Held-out shard rationale (val-bpb)

**Choice: climbmix shards 100 and 101** (`--heldout-shards`, override as needed).

Evidence they are untouched by training:

- Training pinned `TRAIN_SHARDS=45` → `nanochat.dataset -n 45` downloads climbmix
  shards **0–44** (train) plus shard 6542 (the always-last val split). The token
  stream is read sequentially from shard 0.
- At the 1.321 B-token horizon (step 2520) the run **actually consumed only
  ~shards 0–28**: the final training log lines show
  `epoch: 1 pq: 28 rg: 0` (`runs/weekdays/pod_logs/exp*.log`) — i.e. the loader
  reached parquet index 28 of the 0–44 train list, still on epoch 1 (no wrap).
- Weekday probe **scores** exist for climbmix-scored shards **0–184**.

So any shard in **[45, 184]** is scored yet never seen by training (and [29, 44]
were downloaded but never read either). **100/101 are clearly beyond** both the
~28 consumed and the 45 downloaded, and are comfortably inside the scored range.
They map to `kaushikreddyxyz/climbmix-scored-overflow-4` (shards 100–124) under
the training layout (`per_repo=25`, `sid → repos[sid//25]`; `run_evals.scored_repo_for_shard`).

val-bpb is bucketed by the `ce_report` buckets — **overall / injected / after /
rest** — where an *injected* token is exactly one whose 7-channel weekday-probe
row is non-zero (present_z ≥ 2.0), i.e. where training actually fired the
injection; *after* is the token immediately following an injected one. See
`run_evals.bucket_masks` for the exact partition (mirror this in the harness's
`ce_report` so the CE and bpb buckets agree).

**Two source modes** (`--valbpb-source`):

- `prescored` (default) — the **positional runtime source machinery**:
  `WeekdayProbeScoreSource` over the climbmix-scored store, joined positionally
  (docs_`<sid>`.jsonl row ↔ climbmix parquet row). This reproduces the training
  injection **exactly** (same pre-computed gemma L8 z, same present_z threshold,
  same overlap-mean alignment) and needs only the gemma *tokenizer* (offsets),
  not gemma inference — but it downloads the ~8.7 GB score shard per held-out
  shard. Held-out climbmix parquet TEXT is fetched separately into
  `<out-dir>/climbmix_heldout/`.
- `gemma` — score the held-out text live with `GemmaScorer` (reuses the already-
  loaded gemma from the weekday/CORE metrics, no score-store download). Lighter;
  equivalent probe up to gemma numerical determinism.

Both cap docs per shard (`--valbpb-max-docs`, default 2000) and forward each doc
once, computing acts **once** and forwarding at every gate scale so on/off see
byte-identical activations. Forward is per-document (BOS + doc tokens), not the
training best-fit packing — a deliberate simplification that makes the
injected-token buckets unambiguous and is fine for a relative on-vs-off bpb.

---

## 3. CORE wiring verdict: **clean adapter, NO source diff**

`nanochat/core_eval.py` needs **no change**. The whole CORE path funnels through
`core_eval.forward_model(model, input_ids)` which does exactly
`outputs = model(input_ids)` and treats `outputs` as `(B, T, vocab)` logits. So an
*activations-on* model adapter that (a) is callable `adapter(input_ids) -> logits`
and (b) injects acts is a drop-in — this is `run_evals.CoreActivationsAdapter`:

1. `core_eval` tokenizes each prompt (`tokenizer(prompts, prepend=bos)`) → padded
   `input_ids` and calls `adapter(input_ids)`.
2. The adapter decodes each row back to text (strip the right-pad BOS and the
   leading BOS), re-encodes to verify no BPE drift (drift → that row gets zero
   acts, a safe no-op), scores gemma inline, `build_acts` → `[T,7]`, stacks to
   `[B,T,7]`, and calls `self.model(input_ids, acts={site: acts * gate_scale})`.
3. `nanochat.gpt.GPT.forward(idx, acts=...)` returns logits when `targets=None`
   (verified) — the adapter returns them unchanged.

Why no diff is possible-to-avoid rather than required:
- `forward_model` passes only `input_ids`, not the source text — but the nanochat
  tokenizer round-trips its own `encode_ordinary` output, so the adapter
  reconstructs text by decoding. This is the only slightly delicate part; it is
  guarded (re-encode check → zero-acts fallback).
- `evaluate_example` only truncates when `hasattr(model, 'max_seq_len')`; the
  adapter deliberately does **not** define it (matching a raw nanochat model), so
  no truncation/index math is disturbed.
- The **baseline** arm has no injection site → the adapter detects
  `_injection_by_block is None` and skips gemma entirely (on == off, trivially —
  the negative control).
- gemma scores are cached by text across the on/off passes (`z_cache`), and
  `--core-skip-gemma-when-off` lets the off pass skip gemma outright (numerically
  identical to gate 0) if the doubled gemma cost matters on the pod.

**No proposed diffs to existing source.** All deliverables are new files; the
CORE metric is wired through the adapter above.

---

## 4. Interface — verified against the committed `harness.py`

`run_evals.py` is aligned to the real harness (not just the pinned contract):

1. **Per-token CE convention** — `forward_metrics` returns `per_token_ce` as
   `[B,T]` fp32 where `ce[t]` scores the **predicted** token t
   (`-log p(ids[t]|ids[:t])`), position 0 = NaN. `run_evals._ce1d` takes row 0
   (we forward one sequence at a time, B=1). Weekday continuation CE =
   `mean(ce[start:end])`; val-bpb attributes `ce[t]`+`token_bytes[ids[t]]` to the
   predicted token t. (Earlier code used the "predict t+1" convention — fixed.)
2. **acts shape** — passed as `[1,T,7]` aligned to input positions (row t = input
   token t, BOS row zero); the harness's `_acts_dict` maps it onto the single r=7
   `weekdays` site. `ids` is a python `list[int]` (BOS included); forward_metrics
   adds the batch dim.
3. **gemma offsets** — `run_evals._gemma_z_and_offsets` uses
   `GemmaScorer.score_with_offsets` (z + offsets from one BOS-free tokenization);
   falls back to `score` + `gemma_offsets`/`gemma_encode`.
4. **nano tokenizer** — `load_model` returns `(model, meta)` with NO tokenizer
   (rustbpe is off-pod-unavailable, but PRESENT on the eval pod), so
   `run_evals._nano_tokenizer` builds it via `get_tokenizer()` and registers it
   with `harness.set_nano_tokenizer` so `build_acts` can reconstruct nano offsets.
5. **buckets** — `run_evals.bucket_masks` uses the **exact** injected/after/rest
   definition as `harness.ce_report` (bucket by the predicted token's injection
   status: injected = `acts[t]≠0`, after = `acts[t-1]≠0 & ¬injected`), so the bpb
   buckets line up with ce_report's CE buckets.
6. **on/off** — `forward_metrics` scales every site's gate in place
   (`_scaled_gates`, 0.0 = exact no-op). The CORE adapter can't go through
   `forward_metrics` (core_eval calls `model(input_ids)`), so it wraps its direct
   `model(input_ids, acts=...)` call in the SAME `harness._scaled_gates(gate_scale)`
   context manager (unified during consolidation review; the original acts×gate_scale
   zeroing was also an exact no-op — `test_consolidation.py` asserts all off
   mechanisms are bit-identical — but one knob everywhere is strictly cleaner).
   Baseline has no site → adapter detects it, skips gemma, and on==off (the
   negative control).

---

## 5. Test table (`test_evalset.py`, 14 tests, CPU, no network, no rustbpe)

| test | asserts |
|------|---------|
| `test_deterministic_bytes` | two generations → identical bytes |
| `test_committed_jsonl_is_fresh` | on-disk `weekday_v1.jsonl` == freshly generated |
| `test_unique_ids_and_size` | unique ids; 400 ≤ n ≤ 600 |
| `test_balanced_day_counts` | each weekday answered equally over balanced categories |
| `test_no_answer_leak_in_fill_in` | fill-in items never contain their answer word |
| `test_options_well_formed` | answer∈options, unique, ≥2, no trailing space, 7-day sets |
| `test_categories_have_multiple_templates` | ≥3 surface templates per reasoning category |
| `test_uniform_logits_are_chance` | stub uniform model → all option CEs equal (margin 0) |
| `test_rigged_logits_are_perfect` | stub oracle model → 100% accuracy |
| `test_cli_parses` | `run_evals.build_parser` parses flags + defaults |
| `test_scored_repo_mapping` | sid→overflow-repo mapping (per_repo=25) |
| `test_bucket_masks_partition` | injected/after/rest partition every valid loss position |
| `test_bpb_math` | `bpb = nats/(ln2·bytes)` edge cases |
| `test_summary_merge_schema` | `metric→arm→gate→{value,n}` shape |

Scoring math is tested with a word-level `StubTokenizer` + numpy CE from stub
logits — no torch, no real tokenizer.

---

## 6. Open questions

- **Injection at eval is out-of-distribution vs training**: training fired the
  injection on natural climbmix docs; the weekday eval prompts are short synthetic
  templates. The gemma weekday probe still fires on the day tokens there, but the
  *density* of injected tokens is much higher than in pretraining. On-vs-off deltas
  should be read as "does the trained direction change day-token behavior", not a
  calibrated effect size.
- **`gate_scale` is binary** (see top): if a graded loudness sweep is ever wanted,
  the harness must scale the site's `gate` parameter, not the acts.
- **val-bpb packing**: per-doc forward (not best-fit 2048 packing) is used for
  clean buckets; absolute bpb will differ slightly from the training-time val bpb
  (which also used a different, non-scored val shard 6542) — compare on/off within
  this harness, not against the training log's bpb.
- **`prescored` download cost**: ~8.7 GB per held-out score shard. For a quick pod
  pass prefer `--valbpb-source gemma`; for exact reproduction of the stored scores
  use `prescored`.
- **CORE text reconstruction**: relies on nanochat BPE round-tripping its own
  output (true in practice); any drift row is a zero-acts no-op, so CORE-on can
  only *under*-inject, never mis-inject.

---

## 7. Consolidation-review changes (Fable reviewer, 2026-07-14)

- **CORE off unified to the gate knob** (section 4.6 above updated): the adapter
  now uses `harness._scaled_gates` instead of acts-zeroing. Equivalence of both
  mechanisms (bit-identical to the vanilla forward) is pinned by
  `test_consolidation.py::test_off_mechanisms_bit_identical`.
- **`--valbpb-max-tokens` (default 2048)**: docs are truncated to the trained
  context AFTER the acts lookup (the prescored positional join needs the full-doc
  token count). Uncapped, a >20,480-token climbmix megadoc would trip the rotary
  cache assert and kill the run; 2048 also keeps the bpb in-distribution.
- **Gemma z cache** (`_GEMMA_CACHE`, keyed by text) + **shared CORE acts cache**
  (`_CORE_ACTS_CACHE`): activations are arm/gate-independent, so the 4-arm run
  now scores gemma once per unique text instead of once per arm (~4x less gemma).
- **Shadow-proof imports**: `_harness()` imports the sibling `harness` module
  with this dir on sys.path (NOT `runs.weekdays.eval.…` — an installed PyPI
  `runs` package shadows the local namespace dir; see the exp2_config.json
  comment); `WeekdayProbeScoreSource` comes from the harness's absolute-path
  re-export for the same reason. Dead `--model-source` flag removed.
