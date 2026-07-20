#!/usr/bin/env bash
# One-GPU seasons injection-on-vs-off eval driver. NO seasons code is duplicated: every
# step invokes the SHARED, family-parameterized drivers in runs/lib/eval with seasons
# arguments (--family/--site-name/--items-module/--opengen-module/--hf-repo). Steps 2-4
# need the seasons checkpoint on HF (kaushikreddyxyz/nanochat-d12-injections/
# seasons_trainable_L0), i.e. train it first via the shared launcher:
#   MODEL_TAG=seasons_trainable_L0 RUN_NAME=seasons-trainable-L0 WANDB_PROJECT=seasons-geometry \
#     CONFIG=runs/seasons/exp_trainable_L0.json bash runs/lib/launch_arm.sh
# All commands from the nanochat repo root:
#   0. .venv/bin/python -m pytest runs/tests -q            # all runs/ tests live there
#   1. .venv/bin/python runs/seasons/eval/pod_smoke.py            # gate: stop if it fails
#   2. runs/lib/eval/run_evals.py  --family seasons ...           # completion + val-bpb + CORE
#   3. runs/lib/eval/causal.py     --items-module ...             # causal/counterfactual
#   4. runs/lib/eval/open_gen.py   --items-module ... --opengen-module ...
# Regenerate the eval set (CPU, deterministic) after editing seasons_evalset.py:
#   python runs/seasons/eval/seasons_evalset.py
set -euo pipefail

cd "$(dirname "$0")/../../.."          # nanochat repo root
PY="${PY:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
SHARED=runs/lib/eval                    # the shared drivers live here
SEVAL=runs/seasons/eval
OUT="$SEVAL/results"
mkdir -p "$OUT"

# Seasons pins (STORE order autumn/spring/summer/winter == cols 43..46; layer-0 injection;
# the unified injections repo). Env-overridable. LOUDNESS is a DIAL in units of the trained
# arm's OWN calibrated channel_scale and applies only to the baseline control — there is no
# absolute loudness constant here any more.
HF_REPO="${HF_REPO:-kaushikreddyxyz/nanochat-d12-injections}"
STEP="${STEP:-2520}"
LOUDNESS="${LOUDNESS:-1.0}"
TRAINABLE_ARM="${TRAINABLE_ARM:-seasons_trainable_L0}"

echo ">> [0/4] local test suite (CPU, fast — catches a broken checkout)"
$PY -m pytest runs/tests -q

echo ">> [1/4] pod smoke gate (tokenizer round-trip, season first tokens, on/off"
echo ">>        bit-identity, baseline site-free, gemma probe channel check)"
$PY "$SEVAL/pod_smoke.py"

echo ">> [2/4] run_evals (shared): seasons_v1 completion + val-bpb (gemma) + CORE, {on,off}"
$PY "$SHARED/run_evals.py" \
  --family seasons \
  --arms baseline "$TRAINABLE_ARM" \
  --site-name seasons --layer 8 \
  --evalset "$SEVAL/evalsets/seasons_v1.jsonl" \
  --valbpb-source gemma \
  --hf-repo "$HF_REPO" --step "$STEP" \
  --out-dir "$OUT" --device "$DEVICE" "$@"

echo ">> [3/4] causal/counterfactual (shared): trainable arm + its baseline control"
$PY "$SHARED/causal.py" \
  --family seasons \
  --items-module "$SEVAL/seasons_items.py" \
  --site-name seasons --real-arms "$TRAINABLE_ARM" --arms all \
  --loudness "$LOUDNESS" --layer 8 \
  --hf-repo "$HF_REPO" --step "$STEP" \
  --out-dir "$OUT" --device "$DEVICE"

echo ">> [4/4] open-generation (shared): season-free prompts, injected season-Y"
$PY "$SHARED/open_gen.py" \
  --family seasons \
  --items-module "$SEVAL/seasons_items.py" \
  --opengen-module "$SEVAL/seasons_opengen_items.py" \
  --site-name seasons --real-arms "$TRAINABLE_ARM" --arms all \
  --loudness "$LOUDNESS" \
  --hf-repo "$HF_REPO" --step "$STEP" \
  --out-dir "$OUT" --device "$DEVICE"

echo ">> done — artifacts:"
ls -la "$OUT"
echo ">> DONE — seasons results/summary.json + causal_summary.json + opengen_summary.json"
