#!/usr/bin/env bash
# Baseline 3-arm reference eval:
#   arms  = { baseline(=off, plain), seasonal-ablate @ MOST_SALIENT_LAYER (L6), seasonal-ablate ALL layers }
#   evals = { CORE (general), seasons_v2 (narrow), colors_v2 (control) [+ seasons_v1] }
# Ablation projects out ONLY the 4 seasonal DoM directions (from baseline/probes/, L6 =
# most-salient layer for the seasons). Baseline has no injection site -> no gemma, no
# injection; the ablation arms are the only interventions. Run from nanochat repo root on 1xH100.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${PY:-.venv/bin/python}"
RUN=runs/lib/eval/run_evals.py
DEVICE="${DEVICE:-cuda}"
HF_REPO=kaushikreddyxyz/nanochat-d12-injections
STEP=2520
EVAL=runs/seasons/eval
NPZ="${NPZ:-/workspace/hfbaseline/baseline/probes/dom_54_probes_difference_of_means_layer06.npz}"
SALIENT="${SALIENT:-6}"
OUT="${OUT:-$EVAL/results_baseline_3arm}"
CORE_MAX="${CORE_MAX:-500}"
mkdir -p "$OUT"

common=( --arms baseline --injection off ablate_L0 ablate_all
         --hf-repo "$HF_REPO" --step "$STEP" --device "$DEVICE"
         --baseline-ablate-npz "$NPZ" --baseline-ablate-layer "$SALIENT"
         --baseline-ablate-concepts autumn spring summer winter )

echo ">> [1/3] seasons_v2 (narrow) + CORE (general)"
$PY "$RUN" "${common[@]}" --family seasons \
  --evalset "$EVAL/evalsets/seasons_v2" --metrics completion core \
  --core-max-per-task "$CORE_MAX" --out-dir "$OUT/seasons"

echo ">> [2/3] colors_v2 (control, completion)"
$PY "$RUN" "${common[@]}" --family color_wheel \
  --evalset "$EVAL/evalsets/colors_v2" --metrics completion --out-dir "$OUT/colors"

echo ">> [3/3] seasons_v1 (simple MC, completion)"
$PY "$RUN" "${common[@]}" --family seasons \
  --evalset "$EVAL/evalsets/seasons_v1.jsonl" --metrics completion --out-dir "$OUT/seasons_v1"

echo ">> DONE. summaries: $OUT/{seasons,colors,seasons_v1}/summary.json"
echo "   arm map: off=baseline plain | ablate_L0=seasonal ablation @L$SALIENT | ablate_all=seasonal ablation @all-layers"