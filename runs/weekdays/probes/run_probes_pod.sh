#!/usr/bin/env bash
# run_probes_pod.sh — one-GPU pod driver for the weekday PROBE experiment.
# Launch via pod_bootstrap.sh (clones weekday-geometry, uv-syncs, nohups this):
#   printf 'GH_TOKEN=%s\nHF_TOKEN=%s\n' "$gh" "$hf" \
#     | ssh <pod> 'bash -s' -- < runs/weekdays/pod_bootstrap.sh runs/weekdays/probes/run_probes_pod.sh
#
# Sequence: CPU tests (gate) -> build probe data ONCE (prescored held-out shard,
# no gemma model) -> 4 arm trainers IN PARALLEL on the one GPU (124M models) ->
# list artifacts. Results land in runs/weekdays/probes/results/.
set -euo pipefail

cd "$(dirname "$0")/../../.."          # nanochat repo root
PY="${PY:-.venv/bin/python}"
PROBES=runs/weekdays/probes
OUT="$PROBES/results"
mkdir -p "$OUT"

echo ">> [0/3] CPU test gate"
$PY "$PROBES/test_probes.py"

echo ">> [1/3] build probe data cache (held-out shard 100, prescored)"
$PY "$PROBES/build_probe_data.py" --max-docs 2500

echo ">> [2/3] train probes: 4 arms in parallel (one GPU, small models)"
pids=()
for arm in baseline trainable sphere orthogonal; do
  $PY "$PROBES/train_probes.py" --arm "$arm" --device cuda \
      > "$OUT/train_$arm.log" 2>&1 &
  pids+=($!)
done
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "!! arm process $i FAILED (see $OUT/train_*.log)"
    fail=1
  fi
done
[ "$fail" -eq 0 ] || { tail -n 30 "$OUT"/train_*.log; exit 1; }

echo ">> [3/3] done — artifacts:"
ls -la "$OUT"
echo ">> DONE — pull $OUT/*.npz *.json direction_*.npy to the laptop for analysis."
