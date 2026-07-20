#!/usr/bin/env bash
# One-GPU weekday injection-on-vs-off eval driver (steps 0-3 below run automatically).
# NO weekday code is duplicated: every step invokes the SHARED, family-parameterized
# drivers in runs/lib/eval with weekday arguments.
#   printf 'GH_TOKEN=%s\nHF_TOKEN=%s\n' "$gh" "$hf" | bash runs/weekdays/pod_bootstrap.sh runs/weekdays/eval/run_all.sh
# Full order (all from the nanochat repo root):
#   0. .venv/bin/python -m pytest runs/tests -q            # all runs/ tests live there
#   1. .venv/bin/python runs/weekdays/eval/pod_smoke.py            # gate: stop if it fails
#   2. runs/lib/eval/run_evals.py  --family weekdays ...
#   3. runs/lib/eval/causal.py     --items-module ...
#   4. runs/lib/eval/open_gen.py   --items-module ... --opengen-module ...  # manually after 3
#   5. python runs/weekdays/eval/make_figures.py                   # local, from results/*.json
set -euo pipefail

cd "$(dirname "$0")/../../.."          # nanochat repo root
PY="${PY:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
SHARED=runs/lib/eval                    # the shared drivers live here
EVAL=runs/weekdays/eval
OUT="$EVAL/results"
mkdir -p "$OUT"

# Weekday pins (STORE order == cols 47..53; block-3 injection). Env-overridable. LOUDNESS
# is a DIAL in units of each trained arm's OWN calibrated channel_scale and applies only to
# the baseline controls — there is no absolute loudness constant here any more.
HF_REPO="${HF_REPO:-kaushikreddyxyz/weekday-geometry-d12}"
STEP="${STEP:-2520}"
LOUDNESS="${LOUDNESS:-1.0}"
REAL_ARMS="${REAL_ARMS:-trainable sphere orthogonal}"

echo ">> [0/4] local test suite (CPU, fast — catches a broken checkout)"
$PY -m pytest runs/tests -q

echo ">> [1/4] pod smoke gate (tokenizer round-trip, on/off bit-identity on CUDA,"
echo ">>        baseline+attach_site, gated gemma + probe constants)"
$PY "$EVAL/pod_smoke.py"

echo ">> [2/4] run_evals (shared): weekday_v1 + val-bpb (held-out shards) + CORE,"
echo ">>        4 arms x {on, off}"
$PY "$SHARED/run_evals.py" \
  --family weekdays \
  --arms baseline trainable sphere orthogonal \
  --site-name weekdays --layer 8 \
  --evalset "$EVAL/evalsets/weekday_v1.jsonl" \
  --hf-repo "$HF_REPO" --step "$STEP" \
  --out-dir "$OUT" --device "$DEVICE"

echo ">> [3/4] causal/counterfactual (shared): 3 real arms + 3 baseline controls"
$PY "$SHARED/causal.py" \
  --family weekdays \
  --items-module "$EVAL/weekday_items.py" \
  --site-name weekdays --real-arms $REAL_ARMS --arms all \
  --loudness "$LOUDNESS" --layer 8 \
  --empirical-json "$EVAL/empirical_patterns.json" \
  --hf-repo "$HF_REPO" --step "$STEP" \
  --out-dir "$OUT" --device "$DEVICE"

echo ">> [4/4] done — artifacts:"
ls -la "$OUT"
echo ">> DONE — results/summary.json + results/causal_summary.json complete."
echo ">> Commit $OUT/*.json to the weekday-geometry branch (small JSONs)."
