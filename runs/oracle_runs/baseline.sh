#!/bin/bash
#
# baseline.sh — oracle-encodings NEGATIVE CONTROL pretraining run
# =============================================================================
# Trains a stock nanochat base model with NO oracle injection attached, to serve
# as the negative control for the oracle-injection experiments.
#
# base_train.py never attaches an oracle (model.oracle_fn stays None, so the hook
# in GPT.forward is a no-op), therefore this run is a clean "no-oracle" baseline
# BY CONSTRUCTION — nothing here needs to disable the oracle explicitly.
#
# Defaults: d24, fp8 (bf16 base + fp8 GEMM for speed), compute-optimal data ratio (12),
# periodic checkpoints, gzip-compressed checkpoints, deterministic seed.
#
# Expected cost @ d24 / fp8 / compute-optimal, $26 per 8xH100-node-hr:
#   ~4.2 h wall-clock  ->  ~$110.   (PRECISION=bf16: ~5.4-6.1 h -> ~$140-160; d26/fp8: ~6.4 h -> ~$165)
#
# ---------------------------------------------------------------------------
# USAGE
#   bash runs/oracle_runs/baseline.sh                      # full run (from anywhere)
#   screen -L -Logfile runs/oracle_runs/baseline.log -S baseline bash runs/oracle_runs/baseline.sh
#   SMOKE=1 bash runs/oracle_runs/baseline.sh             # quick OOM/pipeline check, nothing saved
#   bash runs/oracle_runs/no_value_embeds.sh             # IDENTICAL run but value embeddings disabled (sets NO_VALUE_EMBEDS=1)
#
# CONTROLLABLE KNOBS (all env-overridable, e.g. `DEPTH=24 DEVICE_BATCH_SIZE=8 bash ...`)
#   Model:        DEPTH=26  RATIO=12  MAX_SEQ_LEN=2048  DEVICE_BATCH_SIZE=16  NPROC=8
#   Precision:    PRECISION=fp8          # fp8 (bf16 base + fp8 GEMM, fastest, default) | bf16 | fp16 | fp32
#                 FP8_RECIPE=tensorwise  # (advanced) fp8 scaling recipe when PRECISION=fp8: tensorwise | rowwise
#   Repro:        SEED=1337
#   Checkpoints:  SAVE_EVERY=2000        # model-checkpoint cadence in steps (-1=final only, 0=never)
#                 SAVE_OPTIMIZER=every   # every | final | never  (optimizer state policy; see note)
#                 COMPRESS_CHECKPOINTS=1 # gzip checkpoints (load auto-detects)
#                 COMPRESS_LEVEL=4       # gzip level 1-9
#   Eval/log:     EVAL_EVERY=250  CORE_METRIC_EVERY=2000  SAMPLE_EVERY=2000
#   Ablation:     NO_VALUE_EMBEDS=0      # 1 = zero+freeze value embeddings (or just run no_value_embeds.sh)
#   Data:         TRAIN_SHARDS=<n>       # override the auto-sized shard count if you want
#   Output:       MODEL_TAG=oracle_baseline_d${DEPTH}_${PRECISION}   (auto-includes value-embeds + precision variant)
#   Mode:         SMOKE=1                # run a 3-step validation at the real config, then exit
#
# CHECKPOINT DISK NOTE: at d26, a full checkpoint is ~14 GB compressed (~6 GB model
# + ~8 GB optimizer). SAVE_OPTIMIZER controls the optimizer half:
#   every : optimizer saved with every checkpoint  -> resume from any step (~85 GB total)
#   final : optimizer only at the last step        -> resume from end; intermediate
#           snapshots are weights-only (~5-6 GB each) -> good for trajectory analysis (~40 GB)
#   never : no optimizer ever                       -> smallest, NOT resumable (~34 GB)
#
# REPRODUCIBILITY NOTE: --seed pins weight init (and the dataloader is already
# deterministic), so control and oracle-treatment runs sharing a seed get identical
# initialization + data order. Bit-exactness is NOT guaranteed (flash-attn / GPU
# atomics are nondeterministic in the backward pass), but the init lottery is fixed.
# =============================================================================

set -euo pipefail

# --- locate repo root so `python -m ...` and `import nanochat` work from anywhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# --- progress banner with running elapsed time (lots of feedback while it runs)
banner () {
    local mins=$((SECONDS / 60)) secs=$((SECONDS % 60))
    echo ""
    echo "######################################################################"
    printf "### %-56s [%3dm%02ds]\n" "$*" "$mins" "$secs"
    echo "######################################################################"
}

# --- config (all env-overridable) -------------------------------------------
DEPTH=${DEPTH:-24}                              # the single complexity dial
RATIO=${RATIO:-12}                              # data:param ratio (12 = nanochat compute-optimal)
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}                # context length
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-16}      # per-GPU microbatch; REDUCE to 8/4/2 if you OOM (larger DEPTH is memory-heavy)
NPROC=${NPROC:-8}                               # GPUs / processes
PRECISION=${PRECISION:-fp8}                     # ONE precision slider: fp8 (bf16 base + fp8 GEMM, fastest, default) | bf16 | fp16 | fp32
FP8_RECIPE=${FP8_RECIPE:-tensorwise}           # (advanced) fp8 scaling recipe when PRECISION=fp8: tensorwise | rowwise
SEED=${SEED:-1337}                              # RNG seed for weight init (share with treatment runs)
SAVE_EVERY=${SAVE_EVERY:-2000}                  # model-checkpoint cadence (steps); -1=final only, 0=never
SAVE_OPTIMIZER=${SAVE_OPTIMIZER:-every}         # optimizer save policy: every | final | never
COMPRESS_CHECKPOINTS=${COMPRESS_CHECKPOINTS:-1} # gzip checkpoints (1=on, 0=off)
COMPRESS_LEVEL=${COMPRESS_LEVEL:-4}             # gzip level 1-9
EVAL_EVERY=${EVAL_EVERY:-250}                   # val bpb cadence
CORE_METRIC_EVERY=${CORE_METRIC_EVERY:-2000}    # CORE metric cadence
SAMPLE_EVERY=${SAMPLE_EVERY:-2000}              # in-training sampling cadence
NO_VALUE_EMBEDS=${NO_VALUE_EMBEDS:-0}           # 1 = zero+freeze value embeddings (no_value_embeds.sh sets this); the ONLY difference vs the default run
TRAIN_SHARDS=${TRAIN_SHARDS:-}                  # override the auto-sized shard count if set
SMOKE=${SMOKE:-0}                               # 1 = quick validation run (no full training, nothing saved)
# Derive the value-embeds ablation flag + a tag/run suffix so the two variants never collide
if [ "$NO_VALUE_EMBEDS" = "1" ]; then NO_VE_FLAG="--no-value-embeds"; TAGSUFFIX="_noVE"; VE_STATUS="DISABLED (zeroed+frozen)"; else NO_VE_FLAG=""; TAGSUFFIX=""; VE_STATUS="learned"; fi
# Precision: one slider -> base compute dtype (NANOCHAT_DTYPE) + fp8 toggle. fp8 keeps a bf16
# base and only accelerates the big matmuls (master weights stay fp32, eval runs in bf16).
case "$PRECISION" in
    fp8)  FP8_FLAG="--fp8 --fp8-recipe=${FP8_RECIPE}"; PREC_DESC="bf16 base + fp8 GEMM (${FP8_RECIPE})"; PREC_TAG="fp8" ;;   # NANOCHAT_DTYPE left auto => bf16
    bf16) FP8_FLAG=""; PREC_DESC="bf16"; PREC_TAG="bf16" ;;
    fp16) export NANOCHAT_DTYPE="float16"; FP8_FLAG=""; PREC_DESC="fp16"; PREC_TAG="fp16" ;;
    fp32) export NANOCHAT_DTYPE="float32"; FP8_FLAG=""; PREC_DESC="fp32"; PREC_TAG="fp32" ;;
    *)    echo "ERROR: PRECISION must be one of: fp8 | bf16 | fp16 | fp32 (got '$PRECISION')" >&2; exit 1 ;;
esac
MODEL_TAG=${MODEL_TAG:-oracle_baseline${TAGSUFFIX}_d${DEPTH}_${PREC_TAG}}   # checkpoint dir name (incl. value-embeds + precision variant so runs never collide)

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
TOK_DIR="$NANOCHAT_BASE_DIR/tokenizer"

banner "ORACLE BASELINE (negative control): d${DEPTH}, ratio ${RATIO}, ${PREC_DESC}, seed ${SEED} | value_embeds=${VE_STATUS}"

# --- python venv setup with uv ----------------------------------------------
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# --- compute the training plan (params, token budget, shards, epochs) -------
# Sizes the dataset download to the compute-optimal token budget plus a margin,
# so the run stays strictly under 1 epoch (no token repeats). Prints a
# human-readable summary to stderr for feedback.
banner "Computing compute-optimal training plan"
read -r _ _ TARGET_TOKENS PLAN_SHARDS TOTAL_PARAMS SCALING_PARAMS <<< "$(python3 - "$DEPTH" "$RATIO" <<'PY'
import sys, math
depth = int(sys.argv[1]); ratio = float(sys.argv[2])
aspect, hd, vocab, vpad = 64, 128, 32768, 64
md = ((depth*aspect + hd - 1)//hd)*hd; nh = md//hd; pv = ((vocab + vpad - 1)//vpad)*vpad
ve = len([i for i in range(depth) if i % 2 == (depth-1) % 2])               # value-embed layers
matrices = depth*(4*md*md + 2*(md*4*md)) + ve*12*nh
total = matrices + pv*md + pv*md + ve*pv*md + (2*depth + 26)                # incl. value-embeds + scalars
scaling = depth*(4*md*md + 2*(md*4*md)) + pv*md                             # transformer matrices + lm_head
target = int(ratio*scaling)                                                # compute-optimal token horizon
TOK_PER_SHARD = 40_000_000     # conservative (speedrun: ~150 shards / 5.84B tok => ~39M/shard)
MARGIN = 1.25                  # download ~25% extra so we always stay < 1 epoch (no data repeats)
shards = min(math.ceil(MARGIN*target/TOK_PER_SHARD), 6542)
avail = shards*TOK_PER_SHARD
print(depth, int(ratio), target, shards, total, scaling)                   # stdout: machine-readable
print(f"  model_dim={md}  heads={nh}  value-embed layers={ve}", file=sys.stderr)
print(f"  scaling params={scaling/1e6:.0f}M   total params={total/1e6:.0f}M", file=sys.stderr)
print(f"  compute-optimal token budget = {target/1e9:.2f}B  (ratio {ratio:g} x {scaling/1e6:.0f}M scaling params)", file=sys.stderr)
print(f"  auto-sized download: {shards} train shards (~{avail/1e9:.1f}B tok avail) + 1 val shard", file=sys.stderr)
print(f"  => ~{target/avail:.2f} epochs worst-case (< 1 by design; real value lower since shards hold >40M tok)", file=sys.stderr)
PY
)"
TRAIN_SHARDS=${TRAIN_SHARDS:-$PLAN_SHARDS}
echo ""
echo "  >> tag=${MODEL_TAG}  device_batch_size=${DEVICE_BATCH_SIZE}  gpus=${NPROC}  seed=${SEED}  value_embeds=${VE_STATUS}"
echo "  >> save_every=${SAVE_EVERY}  save_optimizer=${SAVE_OPTIMIZER}  compress=${COMPRESS_CHECKPOINTS}(lvl ${COMPRESS_LEVEL})  shards=${TRAIN_SHARDS}"

# =============================================================================
# SMOKE TEST: validate the pipeline + peak memory at the REAL config, then exit.
# Runs 3 optimization steps at the real depth/precision/device-batch-size (so the
# per-GPU memory footprint is faithful) but with a tiny total batch and everything
# else disabled. Saves nothing. Use it to catch OOM before committing the full run.
# =============================================================================
if [ "$SMOKE" = "1" ]; then
    banner "SMOKE TEST — validating pipeline + peak memory at real config (nothing saved)"
    # tokenizer: reuse if present, else train one on the first 8 shards (one-time)
    if [ -f "$TOK_DIR/tokenizer.pkl" ] && [ -f "$TOK_DIR/token_bytes.pt" ]; then
        echo "  >> reusing existing tokenizer"
    else
        echo "  >> no tokenizer found -> training one on the first 8 shards (one-time)"
        python -m nanochat.dataset -n 8
        python -m scripts.tok_train
        python -m scripts.tok_eval
    fi
    python -m nanochat.dataset -n 8     # a little training data (a few steps' worth)
    SMOKE_TBS=$(( DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NPROC ))   # one grad-accum step -> fast but real per-GPU memory
    echo "  >> 3 steps @ depth=$DEPTH device_batch_size=$DEVICE_BATCH_SIZE seq_len=$MAX_SEQ_LEN bf16 (total_batch=$SMOKE_TBS)"
    torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth="$DEPTH" \
        --max-seq-len="$MAX_SEQ_LEN" \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --total-batch-size="$SMOKE_TBS" \
        --num-iterations=3 \
        --seed="$SEED" \
        $NO_VE_FLAG \
        $FP8_FLAG \
        --save-every=0 \
        --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
        --run=dummy
    banner "SMOKE PASSED — model builds + fwd/bwd + optimizer step ran without OOM at this config"
    echo "  >> launch the full run with:  bash runs/oracle_runs/baseline.sh   (omit SMOKE=1)"
    exit 0
fi

# --- wandb: seamless auto-detect (auto-on if authenticated, else disabled) ---
DEFAULT_RUN="baseline${TAGSUFFIX}_d${DEPTH}_${PREC_TAG}_r${RATIO}"
if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_AUTHED=$(python3 -c "
import os
try:
    import nanochat.common  # loads nanochat/.env + mirrors WANDB_TOKEN->WANDB_API_KEY (exactly as training does)
except Exception:
    pass
ok = bool(os.environ.get('WANDB_API_KEY'))
if not ok:
    try:
        import netrc; ok = bool(netrc.netrc().authenticators('api.wandb.ai'))
    except Exception:
        ok = False
print(int(ok))
" 2>/dev/null || echo 0)
    if [ "$WANDB_AUTHED" = "1" ]; then
        WANDB_RUN="$DEFAULT_RUN"
        echo "  >> wandb authenticated -> logging to project 'nanochat' as run '$WANDB_RUN'"
    else
        WANDB_RUN=dummy
        echo "  >> wandb NOT authenticated -> logging disabled."
        echo "     To enable: run 'wandb login' (or set WANDB_API_KEY / pass WANDB_RUN=name), then re-run."
    fi
else
    echo "  >> using WANDB_RUN='$WANDB_RUN' from environment"
fi

# --- start the run report (system info + timestamp header) ------------------
python -m nanochat.report reset

# --- tokenizer: REUSE if present, only train when missing -------------------
banner "Tokenizer"
if [ -f "$TOK_DIR/tokenizer.pkl" ] && [ -f "$TOK_DIR/token_bytes.pt" ]; then
    HAVE_TOKENIZER=1
    echo "  >> Found existing tokenizer at $TOK_DIR -> REUSING it (skipping tok_train/tok_eval)."
else
    HAVE_TOKENIZER=0
    echo "  >> No tokenizer found -> will train a fresh one (vocab 32768) after the first 8 shards download."
    python -m nanochat.dataset -n 8     # tokenizer trains on ~2B chars = first 8 shards (synchronous)
fi

# --- dataset: kick off the full compute-optimal download in the background --
banner "Downloading dataset (${TRAIN_SHARDS} train shards) in background"
python -m nanochat.dataset -n "$TRAIN_SHARDS" &
DATASET_DOWNLOAD_PID=$!

if [ "$HAVE_TOKENIZER" = "0" ]; then
    echo "  >> Training tokenizer while the rest of the dataset downloads..."
    python -m scripts.tok_train
    python -m scripts.tok_eval
fi

echo "  >> Waiting for dataset download to finish before pretraining..."
wait "$DATASET_DOWNLOAD_PID"
echo "  >> Dataset ready."

# --- base model pretraining (NO oracle = negative control; bf16; compute-opt)
banner "Pretraining base model — d${DEPTH}, ratio ${RATIO}, ${PREC_DESC}, NO oracle"
# Key choices:
#   * precision via the PRECISION knob (default fp8 = bf16 base + fp8 GEMM for speed; set PRECISION=bf16 for cleanest numerics)
#   * --target-param-data-ratio=12   => compute-optimal token horizon
#   * --total-batch-size omitted     => auto-computed optimal batch from scaling laws
#   * --seed                         => reproducible init (share with treatment runs)
#   * --save-every / --save-optimizer / --compress-checkpoints => checkpoint controls
torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
    --depth="$DEPTH" \
    --target-param-data-ratio="$RATIO" \
    --max-seq-len="$MAX_SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --model-tag="$MODEL_TAG" \
    --seed="$SEED" \
    $NO_VE_FLAG \
    $FP8_FLAG \
    --save-every="$SAVE_EVERY" \
    --save-optimizer="$SAVE_OPTIMIZER" \
    --compress-checkpoints="$COMPRESS_CHECKPOINTS" \
    --checkpoint-compress-level="$COMPRESS_LEVEL" \
    --eval-every="$EVAL_EVERY" \
    --core-metric-every="$CORE_METRIC_EVERY" \
    --sample-every="$SAMPLE_EVERY" \
    --run="$WANDB_RUN"

# --- base model evaluation: CORE, bpb (train/val), samples ------------------
banner "Evaluating base model (CORE, bpb, samples)"
torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_eval -- \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --model-tag="$MODEL_TAG"

# =============================================================================
# OPTIONAL: SFT (chat finetuning) — COMMENTED OUT.
# The baseline is a *base* model (the negative control). Uncomment the block
# below to also produce a chat model. SFT loads the base checkpoint BY TAG, so
# --model-tag must match the pretraining tag above.
# =============================================================================
# banner "SFT (optional chat finetuning)"
# # 2.3MB of synthetic identity conversations (gives nanochat a personality)
# curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
#     https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
# torchrun --standalone --nproc_per_node="$NPROC" -m scripts.chat_sft -- \
#     --device-batch-size="$DEVICE_BATCH_SIZE" \
#     --model-tag="$MODEL_TAG" \
#     --run="$WANDB_RUN"
# torchrun --standalone --nproc_per_node="$NPROC" -m scripts.chat_eval -- -i sft
#
# # talk to it over CLI:   python -m scripts.chat_cli -p "Why is the sky blue?"
# # or the ChatGPT-style web UI:   python -m scripts.chat_web

# --- assemble the final markdown report -------------------------------------
banner "Generating report"
python -m nanochat.report generate

banner "DONE — baseline d${DEPTH} (bf16, compute-optimal) complete"
echo "  checkpoints: $NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG/"
echo "  report:      $REPO_ROOT/report.md"
