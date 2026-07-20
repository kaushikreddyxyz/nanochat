#!/usr/bin/env bash
# run_sphere_L0.sh — seasons probe-score injection at BLOCK 0, FROZEN 4-point circle
# direction (direction_sphere.npz). Pinned config shared with run_trainable_L0.sh: depth=12,
# --no-value-embeds, seed 1337, max_seq_len 2048, device_batch_size 32, nproc=8,
# TRAIN_SHARDS=45, --num-iterations 2520 + --target-param-data-ratio 12. Source: gemma
# L8 scores from climbmix-scored (+overflow..-7, shards 0-184), 4 season channels in
# STORE order, present_z=0 (the site's relu owns thresholding), align mean, noise 0.
# BEFORE RUNNING: python runs/seasons/build_directions.py
# USAGE:  bash runs/seasons/run_sphere_L0.sh
#         SMOKE=1 bash runs/seasons/run_sphere_L0.sh   # 3 steps, nothing saved
# Fresh pod: pod_bootstrap.sh runs/seasons/run_sphere_L0.sh (runs/weekdays/pod_bootstrap.sh)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DEPTH=${DEPTH:-12}
RATIO=${RATIO:-12}
NUM_ITERATIONS=${NUM_ITERATIONS:-2520}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-32}
NPROC=${NPROC:-8}
SEED=${SEED:-1337}
PRECISION=${PRECISION:-bf16}
SAVE_EVERY=${SAVE_EVERY:-2000}
SAVE_OPTIMIZER=${SAVE_OPTIMIZER:-final}
EVAL_EVERY=${EVAL_EVERY:-250}
CORE_METRIC_EVERY=${CORE_METRIC_EVERY:-2000}
SAMPLE_EVERY=${SAMPLE_EVERY:-2000}
LOOKUP_WORKERS=${LOOKUP_WORKERS:-8}
TRAIN_SHARDS=${TRAIN_SHARDS:-45}
SMOKE=${SMOKE:-0}

RUN_NAME=${RUN_NAME:-seasons-sphere-L0}
MODEL_TAG=${MODEL_TAG:-seasons_sphere_L0}
HF_REPO=${HF_REPO:-kaushikreddyxyz/nanochat-d12-injections}
HF_SUBDIR=${HF_SUBDIR:-seasons_sphere_L0}
WANDB_PROJECT=${WANDB_PROJECT:-seasons-geometry}
CONFIG="$SCRIPT_DIR/exp_sphere_L0.json"
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export WANDB_PROJECT

if [ ! -f "$SCRIPT_DIR/direction_sphere.npz" ]; then
  echo "ERROR: $SCRIPT_DIR/direction_sphere.npz is missing." >&2
  echo "       Run: python runs/seasons/build_directions.py" >&2
  exit 1
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

echo "############################################################"
echo "# SEASONS L0 — FROZEN circle direction (injection after block 0)"
echo "#   depth=$DEPTH  iters=$NUM_ITERATIONS  seed=$SEED  precision=$PRECISION"
echo "#   nproc=$NPROC  device_batch=$DEVICE_BATCH_SIZE  max_seq_len=$MAX_SEQ_LEN"
echo "#   tag=$MODEL_TAG  config=$CONFIG"
echo "#   wandb=$WANDB_PROJECT/$RUN_NAME   hf=$HF_REPO/$HF_SUBDIR/"
echo "############################################################"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -f "$TOK_DIR/tokenizer.pkl" ] && [ -f "$TOK_DIR/token_bytes.pt" ]; then
  echo ">> reusing existing tokenizer at $TOK_DIR"
else
  echo ">> no tokenizer -> training a fresh one (vocab 32768) on the first 8 shards"
  python -m nanochat.dataset -n 8
  python -m scripts.tok_train
  python -m scripts.tok_eval
fi

echo ">> downloading $TRAIN_SHARDS standard-climbmix train shards"
python -m nanochat.dataset -n "$TRAIN_SHARDS"

if [ "$SMOKE" = "1" ]; then
  echo ">> SMOKE: 3 steps @ real config (nothing saved)"
  SMOKE_TBS=$(( DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NPROC ))
  torchrun --standalone --nproc_per_node="$NPROC" -m scripts.injection_train -- \
    --activation-config "$CONFIG" \
    --depth="$DEPTH" --max-seq-len="$MAX_SEQ_LEN" --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$SMOKE_TBS" --num-iterations=3 --seed="$SEED" --no-value-embeds $FP8_FLAG \
    --save-every=0 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
    --lookup-workers="$LOOKUP_WORKERS" --run=dummy
  echo ">> SMOKE PASSED — source wiring + build + fwd/bwd + optimizer step OK"
  exit 0
fi

python -m nanochat.report reset

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

echo ">> pretraining d$DEPTH seasons sphere-L0 ($PRECISION, no value embeds)"
torchrun --standalone --nproc_per_node="$NPROC" -m scripts.injection_train -- \
  --activation-config "$CONFIG" \
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
  --compress-checkpoints=1 \
  --eval-every="$EVAL_EVERY" \
  --core-metric-every="$CORE_METRIC_EVERY" \
  --sample-every="$SAMPLE_EVERY" \
  --lookup-workers="$LOOKUP_WORKERS" \
  --wandb-project="$WANDB_PROJECT" \
  --run="$RUN_NAME"

echo ">> evaluating base model (CORE, bpb, samples)"
torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_eval -- \
  --device-batch-size="$DEVICE_BATCH_SIZE" \
  --model-tag="$MODEL_TAG"

python -m nanochat.report generate || true
cp -f "$REPO_ROOT/report.md" "$CKPT_DIR/report.md" 2>/dev/null || true
touch "$STOP_FILE"
wait "$HF_PUSH_PID" 2>/dev/null || true
trap - EXIT
echo ">> DONE — seasons sphere L0 complete."
echo "   checkpoints: $CKPT_DIR"
echo "   pushed to:   https://huggingface.co/$HF_REPO/tree/main/$HF_SUBDIR"
