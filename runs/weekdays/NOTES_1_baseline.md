# NOTES — Exp 1 (baseline) + shared launch infra

Owner deliverables: `exp1_baseline.sh`, `pod_bootstrap.sh`, `hf_push.py`,
`REPORT.md`, `test_exp1.py`. All new files under `runs/weekdays/`. No source
files were edited (see "Proposed source diffs" below).

## What Exp 1 is

The negative control for the 4-run weekday-geometry study: a stock nanochat base
model, **depth 12**, **value embeddings disabled** (`--no-value-embeds`), **no
oracle / no probe-score injection**. `base_train.py` never attaches an oracle
(`model.oracle_fn` stays `None`, its forward hook is a no-op), so "no injection"
is true by construction. Runs 2-4 add weekday probe-score injection with three
different direction geometries; everything else is held identical.

## Token budget (the shared, quick horizon)

d12 geometry (from `base_train.build_model_meta`):
- `model_dim = ((12*64 + 127)//128)*128 = 768`, `n_head = 768/128 = 6`.
- scaling params (transformer matrices + lm_head, the quantity that sets the
  horizon) `= 12*(4*768^2 + 2*768*3072) + 32768*768 = 84,934,656 + 25,165,824
  = 110,100,480 ~= 110.1M`.

Horizon via `--target-param-data-ratio 12` (base_train's d12 compute-optimal
default; **< the ~2.5B Chinchilla-20x ceiling, so per the brief we take the
smaller d12 default** — quicker and matches the sibling injection runs, which
also use ratio 12):
- `target_tokens = 12 * 110.1M = 1,321,205,760`.
- `total_batch_size` auto-computes to `2^19 = 524,288` (d12 is the muP reference:
  `B_REF = 2^19`, and `target_tokens == D_REF` so the `D^0.383` batch factor and
  the sqrt-LR / weight-decay factors are all exactly 1.0 -> hyperparameters stay
  at their tuned d12 defaults).
- `num_iterations = 1,321,205,760 // 524,288 = 2520 steps`.
- **actual tokens trained = 2520 * 524,288 = 1,321,205,760 ~= 1.321B.**

It is pinned **explicitly** in the script via `--target-param-data-ratio 12`
(the `RATIO` env knob), so the horizon is documented rather than implicit. To
push toward the 2.5B ceiling instead, set `RATIO=22` on **all four** runs
(22*110.1M/524288 = 4620 steps -> 2.42B); left off by default for speed + parity.

Why not pin `--num-iterations`: `injection_train` (runs 2-4) computes the horizon
from the SAME `--target-param-data-ratio 12` and the same scaling-param count
(injection adds only a tiny r*n_embd direction, not transformer matrices or
lm_head), so all four derive the identical `2520 steps / 524,288 batch`. Using the
same ratio-driven code path guarantees the match while keeping the `DEPTH` knob
meaningful.

**Data-order caveat:** the deterministic loader's order depends on `world_size`.
All four runs must launch with the **same `nproc`** (recommended `nproc=8`). Mixing
1-GPU and 8-GPU runs would silently change the data order and break the control.

## GPU sizing recommendation

Estimates anchored to the nanochat speedrun (d20, 8xH100, ~3 h end-to-end for a
~1.4e19-FLOP capability model). Exp 1 is d12 @ ratio 12: FLOPs/token ~= 6 x 110.1M
+ attention ~= 7.5e8; total pretraining ~= 7.5e8 x 1.321e9 ~= **1.0e18 FLOPs**,
roughly 14-40x cheaper than the d20 speedrun's pretraining. Prices at ~$3/GPU/hr.

| GPU config | grad_accum | pretrain compute | end-to-end wall* | $/hr | est. cost / run |
|------------|-----------:|-----------------:|-----------------:|-----:|----------------:|
| **8xH100** (recommended) | 1 | ~6-10 min | ~30-45 min | $24 | **~$12-18** |
| 1xH100 | 8 | ~50-80 min | ~2-3.5 h | $3 | ~$6-11 |

*end-to-end includes cold shard download (~45 shards), one-time tokenizer train
(if absent), periodic val-bpb/CORE/sampling evals, and the final `base_eval`.
Compute-only pretraining is the first number.

**Recommendation: run all four on 8xH100.** Identical `world_size` => identical
data order (required for the control); ~10x faster wall clock than 1xH100 for
~the same total GPU-hours; and clean `grad_accum=1` at
`device_batch_size=32 x seq 2048 x 8 = 524,288 = total_batch_size`. The full
4-run set is ~**$50-72** on 8xH100 (well under the $400 budget), and all four
can share one 8xH100 pod run back-to-back (tokenizer + shards are reused from
`$NANOCHAT_BASE_DIR`). 1xH100 is the cheapest per run but ~2-3.5 h each and must
not be mixed with 8-GPU runs.

## Proposed source diffs (NOT applied — per the hard rule)

### Diff 1 (needed only to route wandb to project `weekday-geometry`)

`scripts/base_train.py:121` hardcodes `project="nanochat"`, which overrides the
`WANDB_PROJECT` env var this script sets. Make it env-overridable (`os` is already
imported at line 14). Mirror of the sibling injection-run diff (which patches the
identical line in `scripts/injection_train.py:161`).

```diff
--- a/scripts/base_train.py
+++ b/scripts/base_train.py
@@ -121,1 +121,1 @@
-wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)
+wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project=os.environ.get("WANDB_PROJECT", "nanochat"), name=args.run, config=user_config)
```

Without it the run still trains and logs fine, just under wandb project
`nanochat` (run name `exp1-baseline` is unchanged, so it's still findable). No
other source change is required for Exp 1.

## Exact launch command sequence

### A. In-repo (local box with GPUs, env already set)
```bash
# from the nanochat repo root, on branch weekday-geometry
export WANDB_API_KEY=...          # or `wandb login`, or nanochat/.env WANDB_TOKEN
export HF_TOKEN=...               # or nanochat/.env HF_TOKEN
SMOKE=1 bash runs/weekdays/exp1_baseline.sh   # 3-step OOM/pipeline check (optional)
bash runs/weekdays/exp1_baseline.sh           # full run (defaults: 8 GPUs, bf16)
```

### B. Fresh RunPod 8xH100 pod (recommended)
```bash
# 1) create an 8xH100 pod (confirm cost first: 8 x ~$3 = ~$24/hr) via the
#    runpod-spinup skill's create-pod.sh, then get direct SSH ip/port.
#    e.g.: create-pod.sh weekday "NVIDIA H100 80GB HBM3" SECURE runpod-torch-v240 8 200

# 2) from the laptop, pipe secrets over stdin (never argv) and bootstrap+launch.
#    GH_TOKEN only if the superproject is private; else omit (ssh-agent/anon clone).
printf 'GH_TOKEN=%s\nHF_TOKEN=%s\nWANDB_API_KEY=%s\n' "$(gh auth token)" "$HF_TOKEN" "$WANDB_API_KEY" \
  | ssh runpod-weekday 'cat > /tmp/pb.sh <<EOS
$(cat runs/weekdays/pod_bootstrap.sh)
EOS
bash /tmp/pb.sh exp1_baseline.sh'
# (simpler, if the repo is already cloned on the pod: run pod_bootstrap.sh directly
#  and feed the same KEY=VALUE block on its stdin.)

# 3) monitor
ssh runpod-weekday 'tail -f /workspace/oracle-encodings/nanochat/exp1_baseline.log'
#    + wandb project weekday-geometry (or nanochat until Diff 1 is applied), run exp1-baseline

# 4) checkpoints auto-push to hf.co/kaushikreddyxyz/weekday-geometry-d12/baseline/
#    every 10 min + a final sync. Tear the pod down when CORE/bpb have landed.
```

Runs 2-4 use the same `pod_bootstrap.sh` (pass `exp2_*.sh` / `exp3_*.sh` /
`exp4_orthogonal.sh`) on a pod with the same `nproc`.

## Notes / assumptions
- `SEED=1337` is base_train's default; passed explicitly and shared with runs 2-4.
- Precision defaults to **bf16** (cleanest numerics for a geometry comparison;
  the run is short so the fp8 speedup is minor). Switch with `PRECISION=fp8` on
  all four if desired — but keep it identical across runs.
- `SAVE_OPTIMIZER=final` (weights every 2000 steps, optimizer only at the end):
  lighter, trajectory-friendly. Set `SAVE_OPTIMIZER=every` for resume-anywhere.
- `hf_push.py` is generic shared infra (best-effort, never kills training); runs
  2-4 can reuse it by changing `--path-in-repo` to `trainable/realistic/orthogonal`.
- Standard climbmix data + default tokenization (NO `--compact-tokens`), matching
  the sibling runs' scored corpus which was produced under default tokenization.

## GOTCHA — `runs/weekdays/REPORT.md` is shadowed by `.gitignore`

The repo `.gitignore` line 5 is `report.md` (to drop the generated training
report at repo root). Git matches it **case-insensitively** on macOS, so
`runs/weekdays/REPORT.md` is *ignored* and won't be committed by a plain
`git add`. Confirmed via `git check-ignore -v runs/weekdays/REPORT.md`.
Fix (consolidator's choice — NOT applied here per the no-source-edit rule):
- `git add -f runs/weekdays/REPORT.md` each time, **or**
- add a negation to `.gitignore`: `!runs/weekdays/REPORT.md`.
The file exists on disk and is the intended living log regardless.
