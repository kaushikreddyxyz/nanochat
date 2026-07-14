# Weekday-geometry — living run log

A 4-run controlled study on nanochat (branch `weekday-geometry`, consolidated
2026-07-14). Depth 12, `n_embd=768`, **value embeddings disabled**
(`--no-value-embeds`), default nanochat tokenization (NO `--compact-tokens`).
Only the injection direction's geometry differs across runs 2-4; run 1 is the
no-injection negative control.

**Shared invariants (all 4 runs MUST match — enforced by
`test_exp1.py::test_cross_run_consistency`):** `depth=12`, `seed=1337`,
`max_seq_len=2048`, `device_batch_size=32`, `total_batch_size=524288` (2^19),
horizon **double-pinned** `--num-iterations 2520` + `--target-param-data-ratio
12` -> **2520 steps = 1,321,205,760 tokens (~1.321B)**, `nproc=8` (world_size
fixes data order), `TRAIN_SHARDS=45` (the shard COUNT fixes the train/val
split: val = the LAST downloaded parquet file), precision bf16, standard
climbmix data. Verified: injection_train counts the trainable 7x768 direction
under a separate `injection` scaling-param key (never in
transformer_matrices+lm_head), so its ratio-12 derivation also lands on exactly
2520 — actual d12 scaling params are 110,100,912 and
`(12*110100912)//524288 == 2520` with or without a site.

Injection runs read weekday probe scores from `kaushikreddyxyz/climbmix-scored`
(+ `-overflow`..`-overflow-7`), **shards 0-184**, gemma layer 8 (store axis-1
index 1), weekday store columns 47..53
(friday,monday,saturday,sunday,thursday,tuesday,wednesday), realism threshold
`present_z=2.0` (row zeroed unless max weekday z >= 2.0), overlap-MEAN
alignment, noise 0, gate `abs:0.0273`, site after block 3, active from step 0.
The three exp configs' `sources` blocks are byte-identical and their site dicts
differ ONLY in direction fields (verified).

HF checkpoints: **`kaushikreddyxyz/weekday-geometry-d12`**, one subfolder per
run, pushed every 10 min in-run by `hf_push.py` + a final sync.

**wandb.** Injected runs (2-4) log to project **`weekday-geometry`** via
injection_train's new `--wandb-project` flag. **Cosmetic exception:** exp1 uses
stock `base_train.py` (standing rule: byte-identical to stock, verified with
`git diff`), which hardcodes project `nanochat` — so **exp1 logs to wandb
project `nanochat` under run name `exp1-baseline`**. Find it there; do not
patch base_train.

## Runs

| # | Run | Script / config | Direction | Pod | wandb (project/run) | Status | CORE | val bpb | Notes |
|---|-----|-----------------|-----------|-----|---------------------|--------|------|---------|-------|
| 1 | exp1 baseline | `exp1_baseline.sh` (stock base_train) | — (no injection) | _tbd_ | `nanochat`/`exp1-baseline` (cosmetic, see above) | not launched | - | - | tag `weekday-geometry-d12-baseline`; HF `baseline/` |
| 2 | exp2 trainable | `exp2_trainable.sh` + `exp2_config.json` | **trainable**, orthonormal init (seed 1337), adamw wd=0 | _tbd_ | `weekday-geometry`/`exp2-trainable` | not launched | - | - | tag `weekday_exp2_trainable`; HF `trainable/` |
| 3 | exp3 sphere | `exp3_sphere.sh` + `exp3_config.json` | **frozen** circle manifold from `direction_sphere.npz` (`file:` init) | _tbd_ | `weekday-geometry`/`exp3-sphere` | not launched | - | - | tag `weekday_exp3_sphere`; HF `sphere/` |
| 4 | exp4 orthogonal | `exp4_orthogonal.sh` + `exp4_config.json` | **frozen** 7x orthonormal null (seed 1337) | _tbd_ | `weekday-geometry`/`exp4-orthogonal` | not launched | - | - | tag `weekday_exp4_orthogonal`; HF `orthogonal/` |

Status legend: not launched / smoke-passed / running / done / failed.

For cross-run comparison use the **in-training** val-bpb + CORE series (same
cadence + `--core-metric-max-per-task 500` in all four trainers); each script
also runs a final `base_eval` (injected checkpoints load cleanly —
checkpoint_manager rebuilds sites from meta; eval forwards never inject).

## Config knobs (canonical values)

| knob | value | source |
|------|-------|--------|
| depth | 12 | `--depth 12` |
| n_embd | 768 | derived (12x64, rounded to head_dim 128) |
| value embeddings | disabled | `--no-value-embeds` |
| seed | 1337 | `--seed 1337` |
| max_seq_len | 2048 | `--max-seq-len 2048` |
| device_batch_size | 32 | shared (grad_accum=1 at 8 GPUs; per-rank data order does not depend on it) |
| total_batch_size | 524,288 | 2^19 (d12 reference; auto-computed from ratio 12) |
| horizon | 2520 steps | `--num-iterations 2520` (+ `--target-param-data-ratio 12` for batch/LR/wd) |
| tokens | 1,321,205,760 | 2520 x 524288 |
| precision | bf16 (default) | `PRECISION` knob |
| nproc | 8 | `torchrun --nproc_per_node=8` (all runs incl. baseline) |
| train shards | 45 | `TRAIN_SHARDS` (same count on every pod; val = last file) |
| tokenization | default (32768 vocab) | NO `--compact-tokens` |

## Exp3 scientific note (decided; do not relitigate)

The exp3 circle (alpha=0.7210, beta=0.6929, rho=0.5199) is the LS-calibrated
unit-norm 1-sphere, **imposed by design**. Gemma's measured L8 weekday geometry
is a near-flat all-positive clump (mean pairwise cosine ~0.30; profile
0.338/0.305/0.252 by calendar distance), which a rigid evenly-spaced circle
cannot reproduce: fitted profile 0.819/0.413/0.087, 3-pt max |resid| = 0.481,
circle-model R^2 vs the 21 pairs = -15.1. Both matrices + residuals are
recorded in `manifold_validation.json`; rows are unit-norm (max err 5.6e-9),
store-channel order (friday=row0), calendar phases (monday theta=0). Interpret
exp3 as "realistic-LOOKING cyclic manifold", not "gemma's manifold".

## Framework changes landed on this branch (consolidation, 2026-07-14)

1. `scripts/injection_train.py::_open_injection_source` — generic custom-source
   hook: source spec `"class"` (`path/to/file.py:Class` preferred, dotted form
   accepted) + `"kwargs"` passthrough, subclass assert, post-construction hard
   assert + startup banner `[injection] site 'weekdays':
   source=WeekdayProbeScoreSource ...`. Resolution lives in
   `nanochat/injection/sources.py::load_source_class`. Tests:
   `tests/test_runtime_probe_source.py` section [G] (resolves + constructs +
   thresholds from each committed exp config).
2. `nanochat/injection/sites.py` — `direction_init: "file:<path.npy|.npz>"`
   (npz key `D` preferred; verbatim rows; loud shape check; frozen unless
   `trainable_direction`). Test:
   `tests/test_injection_sites.py::test_direction_init_file` (bit-exact vs
   exp3's real npz).
3. `scripts/injection_train.py` — `--wandb-project` flag (default `nanochat`).
4. `scripts/base_train.py` — **UNTOUCHED** (verified `git diff` empty).

## Launch checklist (per pod, in order)

0. Create an **8xH100** pod (~$24/hr), e.g. runpod-spinup
   `create-pod.sh weekday-expN "NVIDIA H100 80GB HBM3" SECURE runpod-torch-v240 8 200`.
1. Bootstrap (never puts secrets on argv; clones the superproject, HARD-checks
   the nanochat submodule out at `weekday-geometry`, writes `.env`, uv-syncs):
   `printf 'GH_TOKEN=%s\nHF_TOKEN=%s\nWANDB_API_KEY=%s\n' ... | bash pod_bootstrap.sh expN_*.sh`
   (for the smoke-first flow, ssh in after bootstrap and run steps 2-4 manually
   from `/workspace/oracle-encodings/nanochat`).
2. **SMOKE preflight (mandatory; 3 steps @ real config, nothing saved):**
   `SMOKE=1 bash runs/weekdays/expN_*.sh`
   Check: (a) no OOM; (b) for runs 2-4 the banner
   `[injection] site 'weekdays': source=WeekdayProbeScoreSource (kind=probe-scores-runtime, class=runs/weekdays/weekday_source.py:WeekdayProbeScoreSource) r=7`;
   (c) score-shard prefetch progressing (`/workspace/scores_staging`).
3. Tokenizer identity across pods (data order depends on it):
   `sha256sum $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl` must match on all
   4 pods (rustbpe is deterministic given the same first 8 shards).
4. Full run: `bash runs/weekdays/expN_*.sh` (or via pod_bootstrap.sh, which
   nohups it). Monitor: `tail -f .../nanochat/expN_*.log`, wandb, and
   `hf_push_*.log`; checkpoints appear under
   `kaushikreddyxyz/weekday-geometry-d12/<subdir>/` every ~10 min.
5. When CORE/bpb land: fill the run table above, tear the pod down.

Budget: ~1.0e18 pretraining FLOPs/run; on 8xH100 ~30-60 min end-to-end,
**~$12-18/run, ~$50-75 for the set** (4 pods in parallel recommended for speed;
data order is unaffected either way at fixed nproc=8).

## Timeline / notes
- 2026-07-14 — consolidation: framework hooks landed + tested (35/35 injection
  suite, smoke OK, all runs/weekdays tests green); configs standardized
  (file-path `class` form, shards 0-184 everywhere); all four scripts pinned to
  the same horizon/batch/nproc/shard-count; exp4 script brought to parity
  (torchrun, SMOKE, pusher, eval); pod_bootstrap superproject-vs-submodule
  branch fix; `.gitignore` negation so this file is tracked.
- _(append dated entries as runs launch / land)_
