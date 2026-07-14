# RUNBOOK — injection-on-vs-off eval suite (weekday-geometry d12)

One-GPU pod run. Everything below is scripted in `run_all.sh`; this file is the
context, the exact sequence, the budget, and the definition of done.

## What this produces

For each arm (`baseline`, `trainable`, `sphere`, `orthogonal`) × gate
(`on`=gate_scale 1.0, `off`=0.0, same code path — gate 0 is a verified exact
no-op):

| artifact | content |
|---|---|
| `results/{arm}_{on,off}.json` | weekday_v1 accuracy/CE/margin per category; val-bpb bucketed overall/injected/after/rest; CORE metric + per-task |
| `results/summary.json` | metric → arm → gate → {value, n} |
| `results/causal_{label}.json` | per-item counterfactual readouts, 6 arm configs (3 real + 3 `baseline_<arm>` negative controls) |
| `results/causal_summary.json` | condition → arm aggregates + the two dose-response matrices |

**DONE =** `results/summary.json` has all 4 arms × on/off for weekday/valbpb/core,
AND `results/causal_summary.json` covers all 6 arm configs. Commit the JSONs to
`weekday-geometry` (they are small).

## Pod spec

- **1× H100 80GB** (A100 80GB fine; ~2× slower). Use the torch-v240 template
  (**the CUDA-11.8 template does not boot** — prior-run lesson).
- Disk ≥ **80 GB**: 2× score shards (~8.7 GB each, prescored val-bpb) + 2×
  climbmix parquet (~1 GB) + 4 checkpoints (~450 MB each) + gemma-2-2b (~10 GB)
  + CORE eval bundle + uv env.
- Secrets: `HF_TOKEN` **required** — `google/gemma-2-2b` is GATED (tokenizer and
  weights) and the checkpoints/score stores live on HF. `GH_TOKEN` if the repo
  clone needs it. No WANDB needed for evals.

## Bootstrap (pod_bootstrap.sh conventions)

```bash
printf 'GH_TOKEN=%s\nHF_TOKEN=%s\n' "$gh" "$hf" \
  | ssh <pod> 'bash -s' -- < runs/weekdays/pod_bootstrap.sh runs/weekdays/eval/run_all.sh
```

pod_bootstrap clones the oracle-encodings superproject + nanochat submodule at
branch **weekday-geometry**, writes `nanochat/.env` (HF token), `uv sync
--extra gpu`, and nohups the given script from the nanochat repo root. Logs at
`nanochat/run_all.log`, PID at `nanochat/run_all.pid`.

**Probe constants are vendored** — `attr_out/probe_set_arrays.npz` +
`probe_set.json` ship in this directory (sha256 `c450f286…` / `5dc370ff…`),
because the superproject gitignores the npz and it exists on NO HF repo. The
harness finds them automatically (`$ORACLE_ATTR_OUT` → superproject
`attribution/out` → vendored `attr_out/`). Nothing to do unless you want to
override with `$ORACLE_ATTR_OUT`.

## Sequence (what run_all.sh does)

1. **Local tests** (~1 min, CPU): `python -m pytest runs/weekdays/eval -q` —
   48 tests; catches a broken checkout/env before any GPU time.
2. **Pod smoke gate** (~5 min): `python runs/weekdays/eval/pod_smoke.py`
   - tokenizer round-trip `decode(encode(t))==t` over EVERY weekday_v1
     prompt+option and causal prompt (+ unicode edge cases) — the CORE
     adapter's acts path depends on it;
   - day-name first tokens distinct (causal readout precondition);
   - `load_model("trainable")` on CUDA: gate 0 forward BIT-identical to
     acts=None under bf16 (the on/off invariant, on-device); gate restored;
   - baseline has no site; `attach_site`(checkpoint direction) round-trips;
   - GemmaScorer end-to-end: gated gemma loads, the day's own STORE channel is
     argmax and ≥ 2.0 on a day token (validates probe constants + store stats).
   **If this fails, stop — nothing downstream is meaningful.**
3. **run_evals** (~2.5–4 h): `python runs/weekdays/eval/run_evals.py --device cuda`
   - weekday_v1: 422 items × ~7 options × 2 gates × 4 arms; gemma z cached by
     text across arms (~3k unique texts).
   - val-bpb: shards **100 & 101** (verified held-out: all four training logs
     end at `epoch: 1 pq: 28` of the 45 downloaded shards; scored range is
     0–184), default `--valbpb-source prescored` (downloads ~17.4 GB of score
     shards; reproduces the training injection exactly). Use
     `--valbpb-source gemma` to skip the download (live scoring, equal up to
     int8 quantization ~0.03σ). Docs are capped at 2048 tokens
     (`--valbpb-max-tokens`), acts looked up on the full doc then truncated.
   - CORE: `--core-max-per-task 500` default; the activations-on adapter
     decodes each row, re-encodes (drift → zero acts, counted and WARNed in the
     output as `adapter_drift_rows` — expect ~0 given smoke step 2), scores
     gemma (cached across arms/gates), forwards under `_scaled_gates`.
     Add `--core-skip-gemma-when-off` to halve gemma time if needed
     (numerically identical to gate 0).
4. **causal** (~30–45 min): `python runs/weekdays/eval/causal.py --device cuda --arms all`
   - 350 items, ~22.7k deduped forwards over 6 arm configs (3 real + 3
     `baseline_<arm>` controls with the arm CHECKPOINT's direction bolted on).
   - `empirical_patterns.json` is committed (shard 0, 3M rows, gemma L8) so the
     empirical conditions run without recomputation.

Interim sanity while it runs: `results/{arm}_{gate}.json` land per arm —
baseline MUST show on == off for every metric (no site → both are the vanilla
forward). If baseline on ≠ off, kill the run.

## Budget

~3–5 h wall on one H100 ≈ **$10–20** (single-GPU pod ~$2.5–4/hr). The dominant
cost is CORE's gemma scoring (~50k unique row texts, once thanks to the shared
cache) and the val-bpb forwards.

## Failure modes / gotchas

- `401` on gemma → HF_TOKEN missing or not gemma-gated-approved.
- PyPI `runs`/`scripts` package shadowing: the suite imports the harness by
  sibling-path (`import harness`), immune; but launch from the **repo root** so
  `nanochat.*` resolves (both drivers also insert the repo root themselves).
- A `file:` direction error while loading the sphere arm should be impossible
  (`load_model` rewrites `file:` → `zeros` before the strict assign-load
  overwrites the direction with the checkpoint's; unit-tested against a
  synthetic meta and verified against the real HF meta).
- Score-store 404 on shards ≥ 100: per-shard files live in the overflow repos
  (`sid // 25` → `climbmix-scored-overflow-4` for 100/101); the metadata JSONs
  only in the primary repo. `_stage_scored_shard` handles the split — do not
  point `WeekdayProbeScoreSource` at a single repo id for held-out shards.
- OOM is not expected (124M model + gemma-2-2b both resident ≈ <15 GB); if it
  happens, run metrics in separate invocations via `--metrics`.
