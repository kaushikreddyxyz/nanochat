#!/usr/bin/env bash
# exp1_baseline.sh — weekday-geometry study, RUN 1 of 4: BASELINE (no injection)
# =============================================================================
# Stock nanochat base-model pretraining. depth=12, value embeddings DISABLED
# (--no-value-embeds), NO oracle / no probe-score injection. This is the negative
# control the three weekday-injection runs (exp2 trainable, exp3 realistic-frozen,
# exp4 orthogonal-frozen) are compared against. base_train.py never attaches an
# oracle (model.oracle_fn stays None), so "no injection" holds BY CONSTRUCTION.
#
# base_train.py is used STOCK (never edited). Every experiment-specific choice is
# passed on the command line below.
#
# CANONICAL SHARED KNOBS (identical across all 4 runs — do NOT diverge):
#   depth=12 · seed=1337 · max_seq_len=2048 · --no-value-embeds
#   --num-iterations 2520 (explicit horizon pin) + --target-param-data-ratio 12
#   nproc=8 (world_size fixes data order — ALL 4 runs must use the same nproc)
#   device_batch_size=32 · TRAIN_SHARDS=45 (shard COUNT fixes the train/val split:
#   the loader's val split is the LAST downloaded parquet file — ALL 4 runs must
#   download the same count) · default tokenization (NO --compact-tokens)
#
# TOKEN BUDGET (see NOTES_1_baseline.md for the derivation):
#   d12 scaling params ≈ 110.1M ; total_batch_size auto-computes to 2^19 = 524,288
#   ratio 12  =>  target_tokens = 12 * 110.1M = 1,321,205,760
#   num_iterations = 1,321,205,760 / 524,288 = 2520 steps  =>  ~1.321B tokens.
#   This is base_train's d12 default (< the ~2.5B Chinchilla-20x ceiling), the
#   QUICK budget shared by all 4 runs. The horizon is DOUBLE-PINNED: --num-iterations
#   2520 fixes the step count explicitly (all 4 runs match token-for-token regardless
#   of derivation), and --target-param-data-ratio 12 keeps the batch-size / LR /
#   weight-decay muP derivations at their d12 reference values (all factors = 1.0).
#   The two agree by construction; injection_train counts scaling params identically
#   (a trainable 7x768 direction is counted under a separate 'injection' key, never
#   in transformer_matrices+lm_head), so the injected runs derive the same 2520.
#   (To push toward 2.5B, set RATIO=22 AND NUM_ITERATIONS=4620 on ALL 4 runs
#    together: 22*110.1M/524288 = 4620 steps => 2.42B. Off by default for speed.)
#
# USAGE (from anywhere):
#   bash runs/weekdays/exp1_baseline.sh              # full run, 8 GPUs
#   SMOKE=1 bash runs/weekdays/exp1_baseline.sh      # 3-step pipeline/OOM check
#   NPROC=1 bash runs/weekdays/exp1_baseline.sh      # single-GPU (grad_accum=8)
#   PRECISION=fp8 bash runs/weekdays/exp1_baseline.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"   # -> nanochat repo root
cd "$REPO_ROOT"

# --- canonical config (env-overridable; keep identical across runs 1-4) ------
DEPTH=${DEPTH:-12}
RATIO=${RATIO:-12}                               # data:param ratio (12 = d12 compute-optimal); drives batch/LR/wd derivations
NUM_ITERATIONS=${NUM_ITERATIONS:-2520}           # explicit horizon pin (== ratio-12 derivation for d12); change RATIO and this TOGETHER, on ALL 4 runs
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-32}       # per-GPU microbatch (d12 fits easily on H100)
NPROC=${NPROC:-8}                                # GPUs; world_size fixes data order -> keep equal across runs
SEED=${SEED:-1337}                               # base_train.py default; shared with the injection runs
PRECISION=${PRECISION:-bf16}                     # bf16 (clean numerics, default) | fp8 (faster, H100+) | fp16 | fp32
SAVE_EVERY=${SAVE_EVERY:-2000}                   # weights checkpoint cadence (steps); -1=final only, 0=never
SAVE_OPTIMIZER=${SAVE_OPTIMIZER:-final}          # optimizer state only at the last step (lighter; trajectory-friendly)
EVAL_EVERY=${EVAL_EVERY:-250}                    # val bpb cadence          (base_train default)
CORE_METRIC_EVERY=${CORE_METRIC_EVERY:-2000}     # CORE metric cadence      (base_train default)
SAMPLE_EVERY=${SAMPLE_EVERY:-2000}               # in-training sampling      (base_train default)
SMOKE=${SMOKE:-0}

# --- experiment identity (aligned with sibling exp2/exp3/exp4 naming) --------
RUN_NAME=${RUN_NAME:-exp1-baseline}
MODEL_TAG=${MODEL_TAG:-weekday-geometry-d${DEPTH}-baseline}
HF_REPO=${HF_REPO:-kaushikreddyxyz/weekday-geometry-d12}
HF_SUBDIR=${HF_SUBDIR:-baseline}
# NOTE: base_train.py stays STOCK (standing rule) and hardcodes wandb project
# "nanochat", so exp1 logs to project "nanochat" under run name "exp1-baseline"
# (cosmetic; documented in REPORT.md). The injected runs use injection_train's
# --wandb-project and land in "weekday-geometry". The export below is inert for
# base_train; kept for intent/tooling.
export WANDB_PROJECT=${WANDB_PROJECT:-weekday-geometry}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}       # xet stalls on pods (see README rolling prefetch)

# --- precision slider -> flags ----------------------------------------------
case "$PRECISION" in
  fp8)  FP8_FLAG="--fp8 --fp8-recipe=tensorwise" ;;
  bf16) FP8_FLAG="" ;;
  fp16) export NANOCHAT_DTYPE=float16; FP8_FLAG="" ;;
  fp32) export NANOCHAT_DTYPE=float32; FP8_FLAG="" ;;
  *) echo "ERROR: PRECISION must be fp8|bf16|fp16|fp32 (got '$PRECISION')" >&2; exit 1 ;;
esac

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
TOK_DIR="$NANOCHAT_BASE_DIR/tokenizer"
CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"

echo "############################################################"
echo "# WEEKDAY-GEOMETRY EXP 1 — baseline (no injection)"
echo "#   depth=$DEPTH  ratio=$RATIO  seed=$SEED  precision=$PRECISION  value_embeds=DISABLED"
echo "#   nproc=$NPROC  device_batch=$DEVICE_BATCH_SIZE  max_seq_len=$MAX_SEQ_LEN"
echo "#   tag=$MODEL_TAG"
echo "#   wandb=$WANDB_PROJECT/$RUN_NAME   hf=$HF_REPO/$HF_SUBDIR/"
echo "############################################################"

# --- python env with uv (nanochat convention) -------------------------------
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# =============================================================================
# SMOKE: 3 optimization steps at the REAL depth/precision/device-batch (faithful
# per-GPU memory), tiny total batch, nothing saved. Catches OOM before the run.
# =============================================================================
if [ "$SMOKE" = "1" ]; then
  echo ">> SMOKE: 3 steps @ real config (nothing saved)"
  if [ ! -f "$TOK_DIR/tokenizer.pkl" ] || [ ! -f "$TOK_DIR/token_bytes.pt" ]; then
    python -m nanochat.dataset -n 8; python -m scripts.tok_train; python -m scripts.tok_eval
  fi
  python -m nanochat.dataset -n 8
  SMOKE_TBS=$(( DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NPROC ))
  torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
    --depth="$DEPTH" --max-seq-len="$MAX_SEQ_LEN" --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$SMOKE_TBS" --num-iterations=3 --seed="$SEED" --no-value-embeds $FP8_FLAG \
    --save-every=0 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --run=dummy
  echo ">> SMOKE PASSED — build + fwd/bwd + optimizer step OK at this config"
  exit 0
fi

# --- run report header ------------------------------------------------------
python -m nanochat.report reset

# --- tokenizer: REUSE if present, else train on the first 8 shards ----------
if [ -f "$TOK_DIR/tokenizer.pkl" ] && [ -f "$TOK_DIR/token_bytes.pt" ]; then
  echo ">> reusing existing tokenizer at $TOK_DIR"
else
  echo ">> no tokenizer -> training a fresh one (vocab 32768) on the first 8 shards"
  python -m nanochat.dataset -n 8
  python -m scripts.tok_train
  python -m scripts.tok_eval
fi

# --- data: download enough standard-climbmix shards to stay < 1 epoch -------
# d12 ratio-12 needs ~1.32B tokens; ~40M tok/shard (conservative) => ~40 shards,
# +margin. 45 keeps us safely under one epoch (no token repeats).
TRAIN_SHARDS=${TRAIN_SHARDS:-45}
echo ">> downloading $TRAIN_SHARDS standard-climbmix train shards"
python -m nanochat.dataset -n "$TRAIN_SHARDS"

# --- background HF checkpoint pusher (push artifacts as they complete) -------
# base_train writes model_/meta_/optim_ files to $CKPT_DIR every SAVE_EVERY steps.
# hf_push.py mirrors $CKPT_DIR -> $HF_REPO/$HF_SUBDIR/ every 10 min, then one final
# sync when $STOP_FILE appears. Best-effort: it can never kill training.
mkdir -p "$CKPT_DIR"
STOP_FILE="$NANOCHAT_BASE_DIR/.exp1_${MODEL_TAG}.pushdone"
rm -f "$STOP_FILE"
python "$SCRIPT_DIR/hf_push.py" \
  --repo "$HF_REPO" --local-dir "$CKPT_DIR" --path-in-repo "$HF_SUBDIR" \
  --watch 600 --stop-file "$STOP_FILE" \
  > "$NANOCHAT_BASE_DIR/hf_push_${MODEL_TAG}.log" 2>&1 &
HF_PUSH_PID=$!
echo ">> HF pusher PID=$HF_PUSH_PID -> $HF_REPO/$HF_SUBDIR/ (every 600s + final)"
cleanup() { touch "$STOP_FILE" 2>/dev/null || true; }
trap cleanup EXIT

# --- pretraining ------------------------------------------------------------
echo ">> pretraining d$DEPTH baseline (ratio $RATIO, $PRECISION, NO injection, no value embeds)"
torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
  --depth="$DEPTH" \
  --target-param-data-ratio="$RATIO" \
  --num-iterations="$NUM_ITERATIONS" \
  --max-seq-len="$MAX_SEQ_LEN" \
  --device-batch-size="$DEVICE_BATCH_SIZE" \
  --model-tag="$MODEL_TAG" \
  --seed="$SEED" \
  --no-value-embeds \
  $FP8_FLAG \
  --save-every="$SAVE_EVERY" \
  --save-optimizer="$SAVE_OPTIMIZER" \
  --eval-every="$EVAL_EVERY" \
  --core-metric-every="$CORE_METRIC_EVERY" \
  --sample-every="$SAMPLE_EVERY" \
  --run="$RUN_NAME"

# --- final evaluation (CORE, bpb, samples) ----------------------------------
echo ">> evaluating base model (CORE, bpb, samples)"
torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_eval -- \
  --device-batch-size="$DEVICE_BATCH_SIZE" \
  --model-tag="$MODEL_TAG"

# --- report + final HF sync -------------------------------------------------
python -m nanochat.report generate || true
cp -f "$REPO_ROOT/report.md" "$CKPT_DIR/report.md" 2>/dev/null || true
touch "$STOP_FILE"                 # tell the watcher to push once more and exit
wait "$HF_PUSH_PID" 2>/dev/null || true
trap - EXIT
echo ">> DONE — exp1 baseline complete."
echo "   checkpoints: $CKPT_DIR"
echo "   pushed to:   https://huggingface.co/$HF_REPO/tree/main/$HF_SUBDIR"
