#!/usr/bin/env bash
# Fit baseline-nanochat DoM concept probes on a GPU pod and push to HF. Run from the
# nanochat repo root (branch weekday-geometry). Idempotent: tokenizer + downloads are
# skipped if already present, so a disconnect/re-run resumes cheaply.
#   HF_TOKEN=... bash runs/lib/dom_pod_run.sh [families]     (default families=all)
set -euo pipefail

FAMILIES="${1:-all}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export HF_HUB_DISABLE_XET=1
BASELINE_REPO="kaushikreddyxyz/nanochat-d12-injections"
DATA_REPO="kaushikreddyxyz/probe-train-data"
CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/baseline_dom"
DATA_DIR="/workspace/probe-train-data"
OUT_DIR="runs/dom_baseline"

command -v uv &>/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"
[ -d .venv ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -n "${HF_TOKEN:-}" ]; then
  python -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])"
fi

# 1) tokenizer (deterministic, must match the baseline's vocab-32768 tokenizer)
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
  echo ">> reproducing tokenizer (nanochat.dataset -n 8 + tok_train)"
  python -m nanochat.dataset -n 8
  python -m scripts.tok_train
fi

# 2) baseline checkpoint (model + meta only; optimizer shards not needed)
mkdir -p "$CKPT_DIR"
CKPT_DIR="$CKPT_DIR" BASELINE_REPO="$BASELINE_REPO" python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
d, repo = os.environ["CKPT_DIR"], os.environ["BASELINE_REPO"]
for f in ("baseline/model_002520.pt.gz", "baseline/meta_002520.json"):
    dst = os.path.join(d, os.path.basename(f))
    if not os.path.exists(dst):
        shutil.copy(hf_hub_download(repo, f), dst)
print("ckpt ready:", os.listdir(d))
PY

# 3) span-labeled text
DATA_DIR="$DATA_DIR" DATA_REPO="$DATA_REPO" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["DATA_REPO"], repo_type="dataset",
                  local_dir=os.environ["DATA_DIR"],
                  allow_patterns=["data/*/final/mixed/*.jsonl"])
print("data ready")
PY

# 4) fit + eval + push (one gold_probes-style stacked npz per layer, into baseline/probes/)
python runs/lib/build_dom.py \
  --ckpt-dir "$CKPT_DIR" --step 2520 --data-root "$DATA_DIR" \
  --out "$OUT_DIR" --device cuda \
  --push-repo "$BASELINE_REPO" --push-subdir baseline/probes

echo ">> DONE. results in $OUT_DIR (pushed to $BASELINE_REPO/baseline/probes/)"
