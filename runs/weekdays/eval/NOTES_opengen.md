# Open-generation battery — design notes

The follow-up to `causal.py`: instead of reading a single answer-position logit,
LET THE MODEL GENERATE and ask whether an injected day-Y pattern actually makes a
weekday surface in free text, which day, and — the user's core critique —
**how that compares to the model's natural propensity**, because a "flip to
Friday" is meaningless if the model already says Friday three times more often
than Wednesday. New files only, under `runs/weekdays/eval/`; no existing source
edited. Pattern builders and empirical loading are REUSED from `causal.py`;
model loading + the acts convention are REUSED from `harness.py`.

- `opengen_items.py` — 30 DAY-FREE, completion-style prompts (pure).
- `open_gen.py` — acts construction / day detection / seeding / delta math /
  summary (all pure) + a hand-rolled generation loop (injected forward_fn) + the
  runtime driver.
- `test_opengen.py` — CPU, plain-assert, stub model/tokenizer (no network/rustbpe).

## Why day-free prompts
Every prompt names NO weekday (asserted in `opengen_items.py`), so the clean
activations are **exactly zero** (like `causal.py`'s implant family). Any day in
the continuation is therefore the model's prior (the `none` condition) or the
injected pattern — never a copied prompt word. Three families, the user's three
shapes: `personal_fact` ("The day Mark was born on is"), `forward_looking`
("My birthday is next"), `schedule` ("My {item} delivers on", swept over 6 item
nouns + 4 variants).

## Conditions (injection geometries)
Acts row `t` aligns to INPUT token `t`; the BOS row (index 0) is ALWAYS zero
(the `with_bos`/training convention). With `ids = [BOS] + <Lb body> + <generated>`,
the final prompt token sits at index `Lb`, the frontier at index `T-1`.

| condition | acts each step | distinction |
|---|---|---|
| `none` | none injected (acts=None) | natural propensity — THE reference for every delta |
| `inject_all` | pattern on rows `1..T-1` (every non-BOS position, prompt + every generated token, frontier included) | reinforced everywhere, grows with the frontier |
| `inject_last` | pattern on row `Lb` only (final prompt token), STATIC | a one-shot nudge at the prompt boundary; generated rows zero |
| `inject_frontier` | pattern on row `T-1` only (current last), MOVES | coincides with `inject_last` at step 0, diverges after |

Doses `k ∈ {1,2,4}` multiply the trained gate (`gate_scale=k`, effective
`k×0.0273`). Pattern: **onehot** (z=3.0 on Y's STORE channel — magnitude is
irrelevant after the site's `z/rms(z)` renorm; only the gate sets loudness) at
all doses; **empirical** (the median 7-channel z-vector from
`empirical_patterns.json`, the real cross-channel structure) at `k=1` only. Y
sweeps all 7 days. => **3 conditions × 7 days × (3 onehot + 1 emp) = 84 injection
combos per template**, plus `none`.

## Generation
Hand-rolled incremental loop, **full recompute per step, no KV cache** (d12,
prompts ~10 tokens — correctness over speed). Row 0 is greedy (argmax), rows
1..16 are seeded samples (T=0.8, top-k 50), all in ONE batched forward per step
(acts are position-only, hence identical across the batch). Seeds are keyed by
`(arm, template, day, condition, pattern, dose)` — reproducible. `max_new=12`.
The forward reuses `harness._scaled_gates` + `_acts_dict` + `model(acts=...)` —
the SAME code path as `harness.forward_metrics`, minus the CE + `[B,T,V]` CPU copy
(a generation loop needs only the last row; copying full logits every step would
dominate the budget). BOS-row-zero and the acts convention are asserted in
`acts_for_step` and covered end-to-end by the stub-forward test.

## Readouts (per generation)
1. **Generation** — does a weekday appear in the 12 new tokens (word-boundary,
   case-insensitive, on decoded text so multi-token days match)? Which day FIRST?
   Reported per (arm, condition, dose): `day_mention_rate`, the full 7-day
   first-mention distribution, `P(first == Y)`, and **Δ vs the SAME template's
   `none`** (per-template — natural propensity varies by template; equal N per
   template makes pooled-minus-pooled == mean of per-template deltas).
2. **First-token day logits** at the first generation position (calendar-ordered,
   via `causal._day_token_ids`) — the SAME mechanism the OLD implant readout used.
   Kept as a comparability + **wiring check** (`_verification`): the 7-day softmax
   sums to 1, so `mean_Y p_Y == 1/7` under any flat/unaffected readout. If
   generation moves under the strongest injection (`inject_all/onehot/@4`) but
   this logit readout stays ~1/7, the old implant/first-token readout was
   mis-wired for this geometry — it never saw the effect generation shows. The
   verdict string says which of the four cases holds.

   Note: `inject_last` and `inject_frontier` are IDENTICAL at step 0 (both inject
   only at index `Lb`), so their first-token logit readout is identical; only
   generation distinguishes them (they diverge at step ≥ 1).

## Output
- `results/opengen_<arm>.json` — per-template `none` + injection combos (sample
  first-days, greedy text, 7-day logits) + the arm summary.
- `results/opengen_summary.json` — **natural-propensity tables FIRST** (per arm:
  `per_day_base_rate`, `day_mention_rate`, pooled distribution, per-template
  breakdown, `logit_argmax_base_rate`), THEN injection effects as deltas
  (`mean_d_p_first_eq_Y_vs_none`, `mean_d_mention_vs_none`, `by_Y`), THEN the
  per-arm wiring verdicts. `--dump-samples` additionally stores every sampled
  completion string (large).

## Arms (7)
`trainable`, `sphere`, `orthogonal` (site in checkpoint) run the full grid; their
3 `baseline_<arm>` controls (baseline + `attach_site` of the arm's checkpoint
direction) run the full grid as the untrained-model attribution control; the
plain `baseline` (NO site) runs `none` only — pure natural propensity, injection
literally impossible. `baseline`'s `none` should equal each `baseline_<arm>`'s
`none` (zero-acts no-op) — a free consistency check.

## Forward budget (computed)
- generations = 6 sited × 30 templates × (1 `none` + 84 combos) + 1 plain × 30 ×
  1 = **15,330 generations**.
- batched forwards = 15,330 × `max_new`(12) = **183,960** at batch B = 1+16 = 17.
- ≈ **45 M token-positions** through the d12 (124M) forward.

On one H100 the compute is trivial (<1 min of matmul); wall time is the
~184k kernel-launch/Python overhead of the un-compiled per-step recompute,
≈ **10–20 min** (A100 ~2×, still ≈ under ~35 min). Under the 30-min target; if
tight, drop `--samples` or `--max-new`, or subset with `--arms`/`--limit`. Model
loads (7 arms, HF-cached) are negligible.

## Predictions (BEFORE data)
- **Natural propensity (`none`)** — base models likely **weekend-heavy**
  (Friday/Saturday/Sunday over-represented, Wednesday/Thursday rare); many
  12-token continuations name NO day at all, so `day_mention_rate` for `none` is
  probably only ~20–50%. This asymmetry is exactly why every effect is a Δ vs
  `none`: "the model flipped to Friday" is unimpressive if Friday is already its
  top natural completion.
- **`inject_all`** — strongest generation shift (pattern reinforced at every
  position + the frontier). At `k=2–4`, expect the clearest `P(first==Y)` lift
  above `none`, largest for `trainable`/`orthogonal` (orthogonal's onehot lands on
  one maximally-decodable direction). Best chance of overriding the prior.
- **`inject_last`** — a static early nudge at index `Lb`, then 8 blocks of mixing
  (site fires after block index 3 of 12) plus free generation must carry it to
  distant generated tokens: predicted **weak → ~nothing** beyond step 0.
- **`inject_frontier`** — injects at the read position each step, so it directly
  perturbs the very-next-token distribution where a day would be emitted;
  predicted **stronger than `inject_last`**, possibly near `inject_all` for the
  FIRST mention, but without accumulation it may not sustain across the 12 tokens.
- **`baseline_<arm>` controls** — `k=1` ≈ nothing (untrained direction); `k=4`
  some perturbation but **unsystematic** (degraded text, not a clean flip to Y).
  Attributes any real-arm effect to LEARNED use, not raw stream perturbation.
- **empirical vs onehot @ k=1** — empirical carries the cross-channel structure
  the models trained on; may beat onehot at equal dose for the real arms.
- **Wiring verification** — for `inject_all/@4` the readout position (`Lb`) IS
  injected, so I expect the first-token logit AND generation to BOTH move
  (verdict: "old readout correctly wired; implant genuinely biases the answer").
  A "generation moves, logit flat (~1/7)" verdict would be a genuine red flag that
  the old implant readout was mis-wired — that is the check this battery adds.

## Exact pod run command
From the nanochat repo root, after `pod_smoke.py` passes (same env as the rest of
the suite — rustbpe + CUDA; NO gemma/probe constants needed here, the prompts are
day-free so there is no `GemmaScorer` call):

```bash
# full battery (7 arms, ~15.3k generations, ~10–20 min on H100)
.venv/bin/python runs/weekdays/eval/open_gen.py --device cuda --arms all

# quick smoke first (2 arms, 3 prompts, 4 samples)
.venv/bin/python runs/weekdays/eval/open_gen.py --device cuda \
    --arms trainable,baseline --limit 3 --samples 4
```

`empirical_patterns.json` is already committed (shard 0, 3M rows, gemma L8), so
the empirical conditions run without recomputation; absent it, pass a bad
`--empirical-json` path and the grid runs onehot-only (63 combos/template) with a
printed notice.
