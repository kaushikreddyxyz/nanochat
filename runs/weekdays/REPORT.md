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
| 1 | exp1 baseline | `exp1_baseline.sh` (stock base_train) | — (no injection) | `uqigccwlyg9ri2` / ssh `runpod-weekday-exp1` (8xH100 SECURE, $23.92/hr) | [`nanochat`/`exp1-baseline`](https://wandb.ai/kaushikreddyxyz-/nanochat/runs/iz7c84vq) (cosmetic, see above) | **done** (launched 13:18 UTC, done ~13:44 UTC 2026-07-14) | **0.1449** | **0.855679** | smoke PASSED (loss 10.40→10.36, peak 27.6GiB); in-training CORE 0.1457 @2520; HF `baseline/` complete (13 files incl. optim ranks + report.md) |
| 2 | exp2 trainable | `exp2_trainable.sh` + `exp2_config.json` | **trainable**, orthonormal init (seed 1337), adamw wd=0 | `0qudfq8c8y28p4` / ssh `runpod-weekday-exp2` (8xH100 SECURE, $23.92/hr); pod DELETED ~15:08 UTC | [`weekday-geometry`/`exp2-trainable`](https://wandb.ai/kaushikreddyxyz-/weekday-geometry/runs/g3ms1w95) | **done** (launched 13:54, done 15:06 UTC) | **0.1371** | **0.856495** | smoke PASSED (banner OK trainable_direction=True gate=0.0273 shards=185; duty forecast ~79% warm); in-training CORE 0.1289 @2000, 0.1397 @2520; HF `trainable/` complete (13 files incl. report.md); log archived |
| 3 | exp3 sphere | `exp3_sphere.sh` + `exp3_config.json` | **frozen** circle manifold from `direction_sphere.npz` (`file:` init) | `szmpknkipifrv0` / ssh `runpod-weekday-exp3` (8xH100 SECURE, $23.92/hr); pod DELETED ~15:02 UTC | [`weekday-geometry`/`exp3-sphere`](https://wandb.ai/kaushikreddyxyz-/weekday-geometry/runs/g2cfve7t) | **done** (launched 13:48, done 14:59 UTC) | **0.1423** | **0.855686** | smoke PASSED (banner OK trainable_direction=False, `file:` direction loaded; duty forecast ~19% cold); in-training CORE 0.1440 @2000, 0.1474 @2520; HF `sphere/` complete (13 files incl. report.md); log archived |
| 4 | exp4 orthogonal | `exp4_orthogonal.sh` + `exp4_config.json` | **frozen** 7x orthonormal null (seed 1337) | **reused pod 1** `uqigccwlyg9ri2` after exp1 (RunPod $80/hr account spend cap blocked a 4th pod); pod DELETED ~14:55 UTC | [`weekday-geometry`/`exp4-orthogonal`](https://wandb.ai/kaushikreddyxyz-/weekday-geometry/runs/h3qk7zod) | **done** (launched 13:50, done ~14:50 UTC) | **0.1334** | **0.855631** | smoke PASSED (banner OK; duty forecast ~13% cold; no OOM 27.7GiB); HF `orthogonal/` complete (13 files incl. report.md); full log archived in `runs/weekdays/pod_logs/` (local) |

Status legend: not launched / smoke-passed / running / done / failed.

**Tokenizer identity (2026-07-14): all three pods MATCH** —
`sha256(tokenizer.pkl) = 387cfc082b0bee45467774fd6f1310a922ad170886a58ccddcb468f275e06a6c`
(exp4 reuses pod 1's tokenizer, so identity holds across all four runs by construction).

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
- 2026-07-14 (deploy day) — 3 pods created SECURE 8xH100 @$23.92/hr each
  (weekday-exp1 `uqigccwlyg9ri2`, weekday-exp2 `0qudfq8c8y28p4`, weekday-exp3
  `szmpknkipifrv0`). A 4th pod was blocked by the RunPod ACCOUNT spend cap
  ($80/hr; 3 pods = $71.9/hr): **exp4 reuses pod 1 after exp1 completed**
  (env/tokenizer/data already provisioned; GPUs verified idle first).
- 2026-07-14 — **four framework/env bugs found by the smoke gates**, all fixed
  on this branch (each fix committed with full forensics in its message):
  1. `7512a36` — fresh pods lack `python3.10-dev` -> Inductor JIT dies on
     `Python.h`; AND `transformers`/`python-dotenv` were dev-group-only while
     `[tool.uv] default-groups=[]`, so `uv sync --extra gpu` skipped them ->
     `ModuleNotFoundError: transformers` in the runtime source (and `.env`
     silently not loaded). Moved to core deps (a `uv pip install` would NOT
     stick: uv sync prunes non-lockfile packages on every run-script start);
     bootstrap now apt-installs dev headers + warns on empty HF_TOKEN.
  2. `a63f5fa` — startup banner called `len(source)`: walks ALL 185 shards'
     docs files BEFORE prefetchers attach -> 404 on shard 25 (overflow-repo
     layout). Banner prints the shard count instead.
  3. `b7f698e` — cross-RANK staging race: every DDP rank ran its own
     ShardPrefetcher over the shared staging dir and deleted files on its
     rank-LOCAL frontier -> fast ranks unlinked shards slow ranks were reading.
     Deletion now keys off the MIN frontier across ranks via atomic
     `.frontier_r{rank}` files (+ dist.barrier before workers start; memmap
     eviction still fires on the OWN frontier). Unit-tested (section [J]).
  4. `210941b` — `hf_hub_download(local_dir=...)` is NOT concurrent-safe for
     same-file callers: hf 0.34.4 `file_download.py:1299` unlinks the
     destination ("delete outdated file first") based on metadata read at
     ENTRY, deleting the file a sibling rank just materialized (destination
     ping-pongs; survived a clean staging wipe). Fixed with an exclusive
     per-file flock + exists-check under the lock; each shard now downloads
     once per NODE instead of once per rank.
- 2026-07-14 — **exp1 baseline COMPLETE** on pod 1: ~25 min end-to-end,
  final CORE **0.1449**, val bpb **0.855679** (in-training CORE 0.1417 @2000,
  0.1457 @2520). HF `baseline/` fully pushed (final sync confirmed).
- 2026-07-14 — ops note (laptop-side): `create-pod.sh`'s ssh-alias step uses
  `grep -oP` (GNU-only) and silently fails on macOS/BSD grep — pod aliases for
  weekday-exp1/2/3 were added to `~/.ssh/config` manually (and removed at
  teardown). Worth fixing in the runpod-spinup skill.
- 2026-07-14 — **ALL FOUR RUNS COMPLETE.** Final standings (final `base_eval`
  CORE / val bpb): baseline **0.1449** / **0.855679** · trainable **0.1371** /
  **0.856495** · sphere **0.1423** / **0.855686** · orthogonal **0.1334** /
  **0.855631**. All four HF subfolders complete (13 files each: model_002000 +
  model_002520 + 8 optimizer ranks + metas + report.md). All three pods
  deleted after per-run HF artifact verification (exp1/exp4 pod ~14:55, exp3
  pod ~15:02, exp2 pod ~15:08 UTC). Full run + smoke logs archived locally in
  `runs/weekdays/pod_logs/` (gitignored). Cost actuals: 3 pods x $23.92/hr,
  ~1.6-2.3h each ≈ **~$120 total** (vs $50-75 estimate; delta = the ~1.5h
  debugging window in which the five bugs above were found and fixed — the
  smoke-gate flow did exactly its job: no full run ever crashed).
- _(append dated entries as runs launch / land)_
