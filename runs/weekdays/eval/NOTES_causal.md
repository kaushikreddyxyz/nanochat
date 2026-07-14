# Causal / counterfactual protocol — design notes

Implements the CAUSAL/COUNTERFACTUAL arm of the weekday-geometry injection-on-vs-off
eval. New files only, under `runs/weekdays/eval/`; no existing source edited.

- `causal_items.py`  — deterministic item set (pure).
- `empirical_patterns.py` — offline helper: the 7 empirical weekday z-vectors.
- `causal.py` — protocol runner (patterns, transforms, condition grid, readout,
  aggregation) + the runtime driver that calls `harness.py`.
- `test_causal.py` — CPU, plain-assert, no harness/gemma/model.

## The scientific question
"When the TEXT says day X but the INJECTION says day Y, which does the model
follow — and how does that scale with the gate?" Everything is one forward
through the SAME injection code path, varied only by `gate_scale` (0 = off) and a
caller-supplied `transform: acts -> acts` applied before injection. That is the
harness's pinned invariant (on/off is always gate 1 vs 0 through the acts path),
so the counterfactual is a strict, apples-to-apples perturbation.

## Item set (`causal_items.py`)
Two families, both completion-style (no chat formatting — 124M base models). Every
prompt ends at the article "a" so the answer is the completion " <Day>".

| family | how many | structure |
|---|---|---|
| `mention` | 5 names x 8 contexts x 7 days = **280** | passage names day X; query asks for a day. `recall` (kq=0, repeat X) x5 contexts; `derived` (kq in {+1,-1,+2}, day arithmetic) x3 contexts. text_answer = day_plus(X, kq). |
| `implant` | 5 names x 2 contexts x 7 injected-days = **70** | DAY-FREE passage + neutral noun; inject day Y on the noun's tokens. cf_answer = day_plus(Y, kq=0). |

`text_answer`/`cf_answer` generalize both query kinds: under a swap X->Y the
counterfactual answer is `day_plus(Y, kq)` (for `derived`, the model would have to
do the arithmetic on the *injected* day).

### The store-vs-calendar trap (mapped explicitly, asserted for all 7 days)
- **STORE order** = r=7 activation-channel order = climbmix-scored name-sorted
  weekday cols 47..53: `friday, monday, saturday, sunday, thursday, tuesday,
  wednesday`. Channel index of a day = `STORE_ORDER.index(day)`. Onehot/empirical
  patterns and the swap channel live here.
- **CALENDAR order** = `monday..sunday`. `day_plus()` and the answer-logit readout
  live here.
- `store_idx(friday)=0` but `cal_idx(friday)=4` — conflating them mislabels every
  counterfactual. `test_store_is_name_sorted` / `test_mapping_all_days` pin both.

## Conditions and the forward grid (`causal.py`)
Per mention item (X, kq): `off` (gate 0), `clean_on` (gate 1), `dose@{.25,.5,2,4}`
(correct-pattern dose curve; 0/1 are off/clean_on), `cf_swap_onehot_{near=X+1,
far=X+3}`, `cf_swap_emp_{near,far}` (if empirical present), `cf_dose_onehot_far@{2,4}`
(can louder injection override the still-present text token?).
Per implant item (Y): `implant_off` (gate 0), `implant_zeroacts_on` (gate 1 on the
zero acts — control that zero acts = no injection), `implant_onehot@{1,2,4}`,
`implant_emp@1`.

Forwards are **deduped on (transform, gate_scale)** per item, then re-labelled, so
the dose curve is complete without redundant compute (`run_item_forwards`).

| | onehot only | + empirical |
|---|---|---|
| unique forwards / mention item | 10 | 12 |
| unique forwards / implant item | 5 | 6 |
| per arm config (280 mention + 70 implant) | 3,150 | 3,780 |
| **x 6 arm configs** | **18,900** | **22,680** |

6 arm configs = 3 real (`trainable`, `sphere`, `orthogonal`) + 3 negative controls
(`baseline_<arm>` = the untrained baseline with that arm's actual checkpoint
direction bolted on via `attach_site`). Well under the ~50k ceiling; `--arms` and
`--limit` subset it, `--empirical-json` toggles the empirical conditions.

### Readout
One forward per (item, condition); read the answer-position logits (last position,
which predicts the day), take each of the 7 day-name **first tokens**' logits
(CALENDAR-indexed), argmax = predicted day. The 7 first tokens are asserted
distinct at startup (they are for " Monday".." Sunday" under GPT-2-style BPE);
if that ever trips, `ce_over_name` implements the CE-over-full-name fallback (wire
`--full-name-ce` — currently the readout fn exists but the per-item multi-forward
loop is not wired; see Proposed diffs). Metrics per condition:
- **gap** = logit(cf_answer) − logit(text_answer): injection-vs-text preference.
- **flip_rate** = argmax == cf_answer (injection won).
- **follow_text_rate** = argmax == text_answer.
- **effect vs off**: dgap_vs_off (swap), dlogit_text_vs_off (dose), dp_cf_vs_off
  (implant). Plus a by-day breakdown (mention: by X; implant: by injected Y).

### Output
`results/causal_<arm>.json` (per-item conditions + arm summary) and
`results/causal_summary.json` (condition -> arm -> {mean_gap, flip_rate, n,
by_day, ...}) with two dose-response matrices: `correct_vs_off` (arm -> gate ->
mean correct-day logit lift vs off) and `counterfactual_far` (arm -> swap gate ->
{flip_rate, mean_gap}).

## Empirical-pattern helper — provenance (`empirical_patterns.py`)
`cf_swap`/`implant` come in two pattern variants:
- **onehot**: z=3.0 on Y's STORE channel, 0 elsewhere (a clean single-channel
  firing).
- **empirical**: the *median 7-channel z-vector over the tokens where Y genuinely
  fires* — carries the real cross-channel structure the gemma probes emit.

The helper computes the 7 empirical vectors from a **climbmix-scored** shard
(the training corpus), gemma **layer 8**, weekday columns looked up **by name**
(store order). int8 -> raw -> z uses the store's own `quant.json` +
`corpus_stats.json` — the identical dequant/standardize the training source ran.
"Y fires" = the realism-threshold semantics (`max weekday z >= present_z=2.0`)
AND Y is the argmax channel (a day-Y token). Ranged .npy reads (header + first
`--max-rows` rows over HTTP Range, no 7.5 GB download) are adapted verbatim from
`../attribution/examples/read_corpus_scores.py`. No gemma tokenizer needed
(nothing is decoded). Output cached to `empirical_patterns.json`; absent it,
`causal.py` runs the onehot conditions only and says so.

## Predictions per arm (written BEFORE seeing data)
- **trainable** — learned its own [7,768] direction (from orthonormal init) to
  READ the injection. Expect the **strongest, cleanest counterfactual following**:
  `cf_swap` flips recall toward Y, monotone `correct_vs_off` dose curve,
  `cf_dose@{2,4}` able to override the literally-present text token. Best chance
  of `derived` items following injected-day *arithmetic*.
- **orthogonal** — 7 mutually orthogonal rows: maximally decodable, zero
  cross-talk, so a onehot(Y) swap lands exactly on one clean direction. IF the
  model uses the injection, orthogonal should give the **strongest onehot swap
  response**. Competing hypothesis: it had the worst CORE (0.1334), consistent
  with the model partly *ignoring* an unrealistic signal -> low flip. The eval
  discriminates; I lean "strong onehot response if used".
- **sphere** — frozen realistic-LOOKING circle (not gemma's actual near-flat
  clump). Days sit at cos/sin phases, so **adjacent** days share direction: expect
  `near`(X+1) swaps to flip LESS reliably than `far`(X+3), and generally weaker/
  more entangled following than trainable.
- **baseline_\*** (never trained with any injection) — at the trained gate
  (k=1) expect **~nothing** (flip ≈ chance, dose lift ≈ 0). At high gate (dose/
  cf_dose @4 ≈ 0.11 of residual RMS) expect SOME perturbation but **unsystematic**
  (prediction degradation, not a clean semantic flip to Y). This is the control
  that attributes any real-arm following to LEARNED use, not raw perturbation.
- **implant** — belief-from-nothing is the hardest ask (no supporting text).
  Expect a weak Y-lift for the real arms (esp. trainable) above `implant_off`,
  larger for `empirical` than `onehot`; baseline_\* ≈ nothing. If real arms
  implant a day into a day-free context, the injection *writes* beliefs, not just
  biases existing ones.
- **cross-cutting**: `far` > `near` flips (day-embedding + circle geometry);
  `recall` > `derived` effects; following likely graded in the gate, not
  threshold — the dose matrices are the payoff plots.

## Open questions the grid answers
1. Does following happen at the trained gate (k=1), or only under amplification?
2. Graded (dose-response) or threshold-like?
3. Can injection override an *explicit* text day-word (flip at k=1), or only bias?
4. Does empirical cross-channel structure beat onehot?
5. Are near-day confusions systematic (sphere's circle)?
6. Is baseline high-gate perturbation noise or structured toward Y?

## Harness reconciliation / proposed diffs
Coded against the landed `harness.py` (the sibling foundation). Interface used:
`set_nano_tokenizer(enc)`; `load_model(arm, device)`; `attach_site(model,
direction, gate)`; `GemmaScorer(device).score_with_offsets([text])`;
`build_acts(text, nano_ids, gemma_z, offsets, threshold, policy, nano_enc)`;
`forward_metrics(...) -> dict['logits'] [B,T,V]`. **No harness edits required.**
Two dependencies to be aware of:
1. **Tokenizer**: obtained on-pod via `nanochat.tokenizer.get_tokenizer().enc`
   (rustbpe present on the training pod) and registered with
   `set_nano_tokenizer`. If the harness later exposes a tokenizer accessor,
   switch to it (one line in `run()`).
2. **Day-name first-token distinctness**: asserted at startup. If it ever trips
   (a " <Day>" whose BPE first token collides), the only needed change is wiring
   the CE-over-full-name readout: `ce_over_name` is implemented and unit-tested;
   the driver just needs a 7-completion multi-forward loop behind `--full-name-ce`
   (currently raises `NotImplementedError` to fail loud rather than silently). No
   other file is affected.

Transforms are pure (never mutate input acts — the same clean acts are reused
across every condition of an item) and axis-agnostic (`[T,7]` in tests, robust to
a leading batch dim), verified for both numpy and torch inputs.

## Test table (`test_causal.py`, 19 tests, CPU, plain assert)
| test | asserts |
|---|---|
| `test_store_is_name_sorted` / `test_mapping_all_days` | store == name-sorted; store vs calendar index differ; both round-trip all 7 days |
| `test_day_plus` | calendar arithmetic incl. wrap (sun+1=mon, mon-1=sun, fri+3=mon) |
| `test_onehot_pattern` | onehot exact: ONEHOT_Z on Y's store channel, 0 else |
| `test_swap_hits_exactly_active_x_and_is_pure` | swap overwrites EXACTLY the rows where X's channel >= present_z; others untouched; input not mutated |
| `test_swap_empirical_pattern` / `test_swap_torch_numpy_parity` | empirical vector applied; numpy == torch, torch input not mutated |
| `test_implant_transform` | injects exactly the given token positions; input not mutated |
| `test_noun_token_span` | prefix-length char->token span correct on a stub tokenizer |
| `test_conditions_dose_grid_and_targets` | full dose grid {0,.25,.5,1,2,4}; near=X+1/far=X+3 targets + cf_answers |
| `test_empirical_conditions_gated_on_presence` | empirical conditions present iff empirical loaded |
| `test_derived_counterfactual_answer` | derived kq applied to both text_answer and swapped cf_answer |
| `test_run_item_forwards_dedup_and_gate_wiring` | forwards deduped on (tkey,gate); every gate_scale reaches forward_fn; off uses gate 0 + no transform |
| `test_implant_conditions` | implant labels/ctype/cf_answer; transforms built |
| `test_day_logits_from_indexing` | 7 day logits gathered at the right first-token ids |
| `test_readout_flip_case` / `test_readout_agree_case` | rigged logits: flip -> follow_cf, gap>0; agree -> follow_text, no gap |
| `test_ce_over_name` | CE ~0 on rigged-correct, large on wrong tokens |
| `test_aggregate_arm_basic` | flip_rate/mean_gap/dgap_vs_off/by_day aggregation |

Run: `python runs/weekdays/eval/test_causal.py` (or pytest). Plumbing additionally
smoke-verified end-to-end against a stub matching the real harness interface
(6 arms, 18,900 forwards, transforms exercised on torch `[T,7]`).

---

## Consolidation-review changes (Fable reviewer, 2026-07-14)

- **BOS prepended to every causal forward** (`with_bos`, unit-tested): ids =
  `[bos] + body`, acts get a zero BOS row, implant noun spans shift +1. This
  matches training (BOS-prefixed docs, acts row t = input token t) and
  run_evals.py's convention, so causal numbers and weekday/val-bpb numbers come
  from the same input distribution. The readout is unchanged (last logits row).
- **Repo-root sys.path insert** so `python runs/weekdays/eval/causal.py` finds
  `nanochat.*` regardless of launch CWD (same convention as run_evals.py).
- **Store-order runtime guard** in `run()`:
  `harness.WEEKDAY_CONCEPTS == STORE_ORDER` asserted at startup (one canonical
  channel order — weekday_source.py; cross-suite copy pinned by
  `test_consolidation.py::test_store_calendar_single_source`).
- **Checkpoint-direction provenance** for `baseline_<arm>` verified end-to-end:
  `_extract_direction` reads the LOADED state dict (learned direction for
  trainable), `attach_site` copies it verbatim
  (`test_consolidation.py::test_attach_site_direction_verbatim_roundtrip`), and
  real arms run before controls so the cache is warm
  (`test_resolve_arms_order_and_controls`).
- `empirical_patterns.json` computed and committed (shard 0, 3M rows, L8):
  all 7 days' own channel dominant at ~2.4σ, cross-channel 0.15–0.95σ,
  40k–62k active tokens/day — sanity pinned by
  `test_consolidation.py::test_empirical_patterns_artifact`.
