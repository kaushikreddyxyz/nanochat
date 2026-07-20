#!/usr/bin/env bash
# launch_arm.sh — the one launcher for every injection/baseline arm. Identity comes
# from env; the physics comes from $CONFIG. Prefix SMOKE=1 for 3 steps, nothing saved.
#   MODEL_TAG=weekday-geometry-d12-baseline RUN_NAME=exp1-baseline HF_SUBDIR=baseline HF_REPO=kaushikreddyxyz/weekday-geometry-d12 WANDB_PROJECT=weekday-geometry bash runs/lib/launch_arm.sh
#   MODEL_TAG=weekday_exp2_trainable RUN_NAME=exp2-trainable HF_SUBDIR=trainable HF_REPO=kaushikreddyxyz/weekday-geometry-d12 WANDB_PROJECT=weekday-geometry CONFIG=runs/weekdays/exp2_config.json bash runs/lib/launch_arm.sh
#   MODEL_TAG=weekday_exp3_sphere RUN_NAME=exp3-sphere HF_SUBDIR=sphere HF_REPO=kaushikreddyxyz/weekday-geometry-d12 WANDB_PROJECT=weekday-geometry CONFIG=runs/weekdays/exp3_config.json bash runs/lib/launch_arm.sh
#   MODEL_TAG=weekday_exp4_orthogonal RUN_NAME=exp4-orthogonal HF_SUBDIR=orthogonal HF_REPO=kaushikreddyxyz/weekday-geometry-d12 WANDB_PROJECT=weekday-geometry CONFIG=runs/weekdays/exp4_config.json bash runs/lib/launch_arm.sh
#   MODEL_TAG=seasons_trainable_L0 RUN_NAME=seasons-trainable-L0 WANDB_PROJECT=seasons-geometry CONFIG=runs/seasons/exp_trainable_L0.json bash runs/lib/launch_arm.sh
#   MODEL_TAG=seasons_sphere_L0 RUN_NAME=seasons-sphere-L0 WANDB_PROJECT=seasons-geometry CONFIG=runs/seasons/exp_sphere_L0.json bash runs/lib/launch_arm.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

: "${MODEL_TAG:?set MODEL_TAG (checkpoint dir + HF subfolder identity)}"
CONFIG=${CONFIG:-}                               # empty => stock base_train, no injection
HF_REPO=${HF_REPO:-kaushikreddyxyz/nanochat-d12-injections}
HF_SUBDIR=${HF_SUBDIR:-$MODEL_TAG}
RUN_NAME=${RUN_NAME:-${MODEL_TAG//_/-}}
WANDB_PROJECT=${WANDB_PROJECT:-nanochat}

# --- canonical config: pinned across ALL arms of a campaign (env-overridable) --
DEPTH=${DEPTH:-12}
RATIO=${RATIO:-12}                               # data:param ratio; drives batch/LR/wd derivations
NUM_ITERATIONS=${NUM_ITERATIONS:-2520}           # 2520 x 524,288 = 1,321,205,760 tokens (== the ratio-12 derivation for d12); change TOGETHER with RATIO, on every arm
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-32}       # rows pack sequentially, so an OOM fallback to 16 does NOT desync data order — but keep it equal across arms anyway
NPROC=${NPROC:-8}                                # world_size fixes data order -> must be equal across arms
SEED=${SEED:-1337}
PRECISION=${PRECISION:-bf16}                     # bf16 | fp8 | fp16 | fp32
SAVE_EVERY=${SAVE_EVERY:-2000}
SAVE_OPTIMIZER=${SAVE_OPTIMIZER:-final}
EVAL_EVERY=${EVAL_EVERY:-250}
CORE_METRIC_EVERY=${CORE_METRIC_EVERY:-2000}
SAMPLE_EVERY=${SAMPLE_EVERY:-2000}
LOOKUP_WORKERS=${LOOKUP_WORKERS:-0}  # 0=inline is fastest; workers are GIL-bound threads (see injection_train --lookup-workers)
TRAIN_SHARDS=${TRAIN_SHARDS:-45}                 # COUNT fixes the train/val split (val = LAST parquet) -> must match across arms
SMOKE=${SMOKE:-0}

export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}   # xet stalls on pods (nanochat/injection/README.md)
export WANDB_PROJECT

# --- config preflight: any file: direction the config names --------------------
if [ -n "$CONFIG" ]; then
  [ -f "$CONFIG" ] || { echo "ERROR: CONFIG not found: $CONFIG" >&2; exit 1; }
  # A frozen "direction_init": "file:<path>" arm needs that npz. It is deterministic
  # and CPU-only, so regenerate it from the family's builder rather than failing.
  DIR_FILE=$(sed -n 's/.*"direction_init"[^"]*"file:\([^"]*\)".*/\1/p' "$CONFIG" | head -1)
  if [ -n "$DIR_FILE" ] && [ ! -f "$DIR_FILE" ]; then
    BUILDER="$(dirname "$CONFIG")/build_directions.py"
    [ -f "$BUILDER" ] || { echo "ERROR: $DIR_FILE missing and no $BUILDER to rebuild it" >&2; exit 1; }
    echo ">> $DIR_FILE missing -> rebuilding via $BUILDER"
    python3 "$BUILDER"
  fi
fi

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

# Stock base_train hardcodes wandb project "nanochat" and takes no --wandb-project /
# --lookup-workers / --compress-checkpoints / --activation-config.
if [ -n "$CONFIG" ]; then
  TRAINER="scripts.injection_train"
  CONFIG_FLAG=(--activation-config "$CONFIG")
  EXTRA_FLAGS=(--compress-checkpoints=1 --lookup-workers="$LOOKUP_WORKERS"
               --wandb-project="$WANDB_PROJECT")
else
  TRAINER="scripts.base_train"
  CONFIG_FLAG=()
  EXTRA_FLAGS=()
fi

echo "############################################################"
echo "# ARM $MODEL_TAG — ${CONFIG:-no injection (stock base_train)}"
echo "#   depth=$DEPTH  ratio=$RATIO  iters=$NUM_ITERATIONS  seed=$SEED  precision=$PRECISION  value_embeds=DISABLED"
echo "#   nproc=$NPROC  device_batch=$DEVICE_BATCH_SIZE  max_seq_len=$MAX_SEQ_LEN"
echo "#   wandb=$WANDB_PROJECT/$RUN_NAME   hf=$HF_REPO/$HF_SUBDIR/"
echo "############################################################"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# rustbpe training is deterministic given the same shards, so every pod reproduces the
# identical tokenizer. Verify across pods before comparing arms:
#   sha256sum $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl
if [ -f "$TOK_DIR/tokenizer.pkl" ] && [ -f "$TOK_DIR/token_bytes.pt" ]; then
  echo ">> reusing existing tokenizer at $TOK_DIR"
else
  echo ">> no tokenizer -> training a fresh one (vocab 32768) on the first 8 shards"
  python -m nanochat.dataset -n 8
  python -m scripts.tok_train
  python -m scripts.tok_eval
fi

# SMOKE covers the real depth/precision/device-batch/world size (faithful per-GPU
# memory) and, for injection arms, custom-source construction + direction load +
# score-shard prefetch. Check the banner for source=<the config's class>.
if [ "$SMOKE" = "1" ]; then
  echo ">> downloading 8 shards (smoke)"
  python -m nanochat.dataset -n 8
  echo ">> SMOKE: 3 steps @ real config (nothing saved)"
  SMOKE_TBS=$(( DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NPROC ))
  torchrun --standalone --nproc_per_node="$NPROC" -m "$TRAINER" -- \
    "${CONFIG_FLAG[@]}" \
    --depth="$DEPTH" --max-seq-len="$MAX_SEQ_LEN" --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$SMOKE_TBS" --num-iterations=3 --seed="$SEED" --no-value-embeds $FP8_FLAG \
    --save-every=0 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
    ${CONFIG:+--lookup-workers="$LOOKUP_WORKERS"} --run=dummy
  echo ">> SMOKE PASSED — source wiring + build + fwd/bwd + optimizer step OK"
  exit 0
fi

echo ">> downloading $TRAIN_SHARDS standard-climbmix train shards"
python -m nanochat.dataset -n "$TRAIN_SHARDS"

python -m nanochat.report reset

# hf_push mirrors $CKPT_DIR -> $HF_REPO/$HF_SUBDIR/ every 10 min, then one final sync
# when $STOP_FILE appears. Best-effort: it can never kill training.
mkdir -p "$CKPT_DIR"
STOP_FILE="$NANOCHAT_BASE_DIR/.${MODEL_TAG}.pushdone"
rm -f "$STOP_FILE"
python "$REPO_ROOT/runs/lib/hf_push.py" \
  --repo "$HF_REPO" --local-dir "$CKPT_DIR" --path-in-repo "$HF_SUBDIR" \
  --watch 600 --stop-file "$STOP_FILE" \
  > "$NANOCHAT_BASE_DIR/hf_push_${MODEL_TAG}.log" 2>&1 &
HF_PUSH_PID=$!
echo ">> HF pusher PID=$HF_PUSH_PID -> $HF_REPO/$HF_SUBDIR/ (every 600s + final)"
cleanup() { touch "$STOP_FILE" 2>/dev/null || true; }
trap cleanup EXIT

echo ">> pretraining d$DEPTH $MODEL_TAG ($PRECISION, no value embeds)"
torchrun --standalone --nproc_per_node="$NPROC" -m "$TRAINER" -- \
  "${CONFIG_FLAG[@]}" \
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
  "${EXTRA_FLAGS[@]}" \
  --run="$RUN_NAME"

# Injected checkpoints load cleanly here: checkpoint_manager rebuilds the sites from
# meta, and eval forwards never inject.
echo ">> evaluating base model (CORE, bpb, samples)"
torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_eval -- \
  --device-batch-size="$DEVICE_BATCH_SIZE" \
  --model-tag="$MODEL_TAG"

python -m nanochat.report generate || true
cp -f "$REPO_ROOT/report.md" "$CKPT_DIR/report.md" 2>/dev/null || true
touch "$STOP_FILE"
wait "$HF_PUSH_PID" 2>/dev/null || true
trap - EXIT
echo ">> DONE — $MODEL_TAG complete."
echo "   checkpoints: $CKPT_DIR"
echo "   pushed to:   https://huggingface.co/$HF_REPO/tree/main/$HF_SUBDIR"
