#!/usr/bin/env bash
#
# exp3_sphere.sh — weekday-geometry EXPERIMENT 3: weekday probe-score injection
# with a FROZEN circle-manifold direction (gemma-L8-calibrated 1-sphere).
# =============================================================================
# Shared weekday-geometry config (PINNED, identical across all 4 runs):
#   depth=12 (n_embd=768), --no-value-embeds, default tokenization (NO
#   --compact-tokens), seed 1337, max_seq_len 2048, device_batch_size 32,
#   nproc=8 (world_size fixes data order), TRAIN_SHARDS=45 (shard COUNT fixes
#   the train/val split: val = the LAST downloaded parquet file), horizon
#   DOUBLE-PINNED: --num-iterations 2520 + --target-param-data-ratio 12
#   (2520 x 524,288 = 1.321B tokens).
#   Source: runtime probe scores from kaushikreddyxyz/climbmix-scored
#   (+ -overflow..-overflow-7), shards 0-184, GEMMA LAYER 8 (store axis-1 idx 1),
#   7 weekday channels (friday,monday,saturday,sunday,thursday,tuesday,wednesday
#   == store cols 47..53), realism threshold present_z=2.0, align_policy=mean,
#   noise_sigma=0, gate abs:0.0273 (absolute scalar), injection site
#   after_block=3, active from step 0. (Site + source live in exp3_config.json.)
#
# EXP-3 SPECIFIC: the direction is FROZEN and read from a FILE
#   (trainable_direction=false, direction_init="file:runs/weekdays/direction_sphere.npz").
#   The 7 rows are a unit-norm circle manifold: m_d = alpha*u0 + beta*(cos th*u1 +
#   sin th*u2), alpha=0.7210 beta=0.6929 (LS-fit to gemma's L8 weekday cosine
#   profile), rows in STORE channel order, phases th from CALENDAR position
#   (monday theta=0). Build/verify offline: python runs/weekdays/build_manifold.py
#   && python runs/weekdays/test_exp3.py . (See NOTES_3_sphere.md — the circle is
#   a POOR fit to gemma's near-flat ~0.3 profile; it is imposed BY DESIGN for
#   this condition and the discrepancy is documented in manifold_validation.json.)
#
# Framework hooks (APPLIED on this branch — no pending diffs):
#   * _open_injection_source honors "class"/"kwargs" (WeekdayProbeScoreSource +
#     present_z=2.0); hard assert + startup banner prove the class constructed.
#   * sites.py direction_init "file:<path>" loads the frozen manifold (bit-exact,
#     loud shape check).
#   * injection_train --wandb-project -> weekday-geometry.
#
# USAGE (from anywhere):
#   bash runs/weekdays/exp3_sphere.sh
#   SMOKE=1 bash runs/weekdays/exp3_sphere.sh      # 3 steps @ real config, nothing saved
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# --- canonical config (env-overridable; keep identical across runs 1-4) ------
DEPTH=${DEPTH:-12}
RATIO=${RATIO:-12}                               # drives batch/LR/wd derivations
NUM_ITERATIONS=${NUM_ITERATIONS:-2520}           # explicit horizon pin (== ratio-12 derivation for d12); change with RATIO, on ALL 4 runs
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-32}       # shared with the baseline; per-rank data order does NOT depend on this (rows pack sequentially), so an OOM fallback to 16 would not desync data — but keep it equal across runs anyway
NPROC=${NPROC:-8}                                # GPUs; world_size fixes data order -> keep equal across runs
SEED=${SEED:-1337}
PRECISION=${PRECISION:-bf16}                     # bf16 (default) | fp8 | fp16 | fp32 — keep identical across runs
SAVE_EVERY=${SAVE_EVERY:-2000}
SAVE_OPTIMIZER=${SAVE_OPTIMIZER:-final}
EVAL_EVERY=${EVAL_EVERY:-250}
CORE_METRIC_EVERY=${CORE_METRIC_EVERY:-2000}
SAMPLE_EVERY=${SAMPLE_EVERY:-2000}
LOOKUP_WORKERS=${LOOKUP_WORKERS:-8}
TRAIN_SHARDS=${TRAIN_SHARDS:-45}                 # MUST match the baseline (val split = last file)
SMOKE=${SMOKE:-0}

# --- experiment identity ------------------------------------------------------
RUN_NAME=${RUN_NAME:-exp3-sphere}
MODEL_TAG=${MODEL_TAG:-weekday_exp3_sphere}
HF_REPO=${HF_REPO:-kaushikreddyxyz/weekday-geometry-d12}
HF_SUBDIR=${HF_SUBDIR:-sphere}
WANDB_PROJECT=${WANDB_PROJECT:-weekday-geometry}
CONFIG="$SCRIPT_DIR/exp3_config.json"
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}   # xet stalls on pods (README rolling prefetch)
export WANDB_PROJECT

# The frozen direction is deterministic ($0, CPU) — regenerate if missing.
[ -f "$SCRIPT_DIR/direction_sphere.npz" ] || python3 "$SCRIPT_DIR/build_manifold.py"

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
echo "# WEEKDAY-GEOMETRY EXP 3 — FROZEN circle manifold (injection)"
echo "#   depth=$DEPTH  iters=$NUM_ITERATIONS  seed=$SEED  precision=$PRECISION  value_embeds=DISABLED"
echo "#   nproc=$NPROC  device_batch=$DEVICE_BATCH_SIZE  max_seq_len=$MAX_SEQ_LEN"
echo "#   tag=$MODEL_TAG  config=$CONFIG"
echo "#   wandb=$WANDB_PROJECT/$RUN_NAME   hf=$HF_REPO/$HF_SUBDIR/"
echo "############################################################"

# --- python env with uv (nanochat convention) -------------------------------
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# --- tokenizer: REUSE if present, else train on the first 8 shards ----------
# NOTE: rustbpe training is deterministic given the same shards, so every pod
# reproduces the identical tokenizer. Verify across pods before comparing runs:
#   sha256sum $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl
if [ -f "$TOK_DIR/tokenizer.pkl" ] && [ -f "$TOK_DIR/token_bytes.pt" ]; then
  echo ">> reusing existing tokenizer at $TOK_DIR"
else
  echo ">> no tokenizer -> training a fresh one (vocab 32768) on the first 8 shards"
  python -m nanochat.dataset -n 8
  python -m scripts.tok_train
  python -m scripts.tok_eval
fi

# --- data: SAME shard count as the baseline (val split = last file) ---------
echo ">> downloading $TRAIN_SHARDS standard-climbmix train shards"
python -m nanochat.dataset -n "$TRAIN_SHARDS"

# =============================================================================
# SMOKE: 3 optimization steps at the REAL depth/precision/device-batch/world
# size, nothing saved. Exercises custom-source construction (check the banner:
# source=WeekdayProbeScoreSource), the file: direction load, score-shard
# prefetch, buffering, fwd/bwd + optimizer. Catches OOM + wiring errors early.
# =============================================================================
if [ "$SMOKE" = "1" ]; then
  echo ">> SMOKE: 3 steps @ real config (nothing saved)"
  SMOKE_TBS=$(( DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NPROC ))
  torchrun --standalone --nproc_per_node="$NPROC" -m scripts.injection_train -- \
    --activation-config "$CONFIG" \
    --depth="$DEPTH" --max-seq-len="$MAX_SEQ_LEN" --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$SMOKE_TBS" --num-iterations=3 --seed="$SEED" --no-value-embeds $FP8_FLAG \
    --save-every=0 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
    --lookup-workers="$LOOKUP_WORKERS" --run=dummy
  echo ">> SMOKE PASSED — source wiring + direction file + fwd/bwd + optimizer step OK"
  exit 0
fi

# --- run report header ------------------------------------------------------
python -m nanochat.report reset

# --- background HF checkpoint pusher (push artifacts as they complete) -------
mkdir -p "$CKPT_DIR"
STOP_FILE="$NANOCHAT_BASE_DIR/.${MODEL_TAG}.pushdone"
rm -f "$STOP_FILE"
python "$SCRIPT_DIR/hf_push.py" \
  --repo "$HF_REPO" --local-dir "$CKPT_DIR" --path-in-repo "$HF_SUBDIR" \
  --watch 600 --stop-file "$STOP_FILE" \
  > "$NANOCHAT_BASE_DIR/hf_push_${MODEL_TAG}.log" 2>&1 &
HF_PUSH_PID=$!
echo ">> HF pusher PID=$HF_PUSH_PID -> $HF_REPO/$HF_SUBDIR/ (every 600s + final)"
cleanup() { touch "$STOP_FILE" 2>/dev/null || true; }
trap cleanup EXIT

# --- pretraining with injection ----------------------------------------------
echo ">> pretraining d$DEPTH exp3 (frozen circle manifold, $PRECISION, no value embeds)"
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

# --- final evaluation (CORE, bpb, samples) — injected checkpoints load cleanly
# (checkpoint_manager rebuilds sites from meta; eval forwards never inject) ----
echo ">> evaluating base model (CORE, bpb, samples)"
torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_eval -- \
  --device-batch-size="$DEVICE_BATCH_SIZE" \
  --model-tag="$MODEL_TAG"

# --- report + final HF sync -------------------------------------------------
python -m nanochat.report generate || true
cp -f "$REPO_ROOT/report.md" "$CKPT_DIR/report.md" 2>/dev/null || true
touch "$STOP_FILE"
wait "$HF_PUSH_PID" 2>/dev/null || true
trap - EXIT
echo ">> DONE — exp3 sphere complete."
echo "   checkpoints: $CKPT_DIR"
echo "   pushed to:   https://huggingface.co/$HF_REPO/tree/main/$HF_SUBDIR"
