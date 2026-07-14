#!/usr/bin/env bash
# run_all.sh — one-GPU pod driver for the injection-on-vs-off eval suite.
# Launch via pod_bootstrap.sh (which clones, uv-syncs, and nohups this script):
#   printf 'GH_TOKEN=%s\nHF_TOKEN=%s\n' "$gh" "$hf" \
#     | bash pod_bootstrap.sh runs/weekdays/eval/run_all.sh
# or by hand from the nanochat repo root. See RUNBOOK.md for the full sequence,
# expected wall time, and the definition of done.
set -euo pipefail

cd "$(dirname "$0")/../../.."          # nanochat repo root
PY="${PY:-.venv/bin/python}"
EVAL=runs/weekdays/eval
OUT="$EVAL/results"
mkdir -p "$OUT"

echo ">> [0/4] local test suite (CPU, fast — catches a broken checkout)"
$PY -m pytest "$EVAL" -q

echo ">> [1/4] pod smoke gate (tokenizer round-trip, on/off bit-identity on CUDA,"
echo ">>        baseline+attach_site, gated gemma + probe constants)"
$PY "$EVAL/pod_smoke.py"

echo ">> [2/4] run_evals: weekday_v1 + val-bpb (held-out shards) + CORE,"
echo ">>        4 arms x {on, off}"
$PY "$EVAL/run_evals.py" --device cuda --out-dir "$OUT"

echo ">> [3/4] causal/counterfactual protocol: 3 real arms + 3 baseline controls"
$PY "$EVAL/causal.py" --device cuda --arms all

echo ">> [4/4] done — artifacts:"
ls -la "$OUT"
echo ">> DONE — results/summary.json + results/causal_summary.json complete."
echo ">> Commit $OUT/*.json to the weekday-geometry branch (small JSONs)."
