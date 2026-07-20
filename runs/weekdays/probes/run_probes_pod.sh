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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">> [0/3] CPU test gate"
$PY -m pytest runs/tests/test_probes.py -q

echo ">> [1/3] build probe data cache (held-out shard 100, prescored)"
if [ -f "$PROBES/probe_data.pt" ]; then
  echo ">> probe_data.pt exists — skipping build (idempotent rerun)"
else
  $PY "$PROBES/build_probe_data.py" --max-docs 2500
fi

echo ">> [2/3] train probes: 4 arms in parallel (one GPU, small models)"
_done_marker() {  # last artifact an arm writes -> arm complete, skip on rerun
  case "$1" in baseline) echo "$OUT/probe_baseline_off.json" ;;
               *) echo "$OUT/probe_bolt_$1_on.json" ;; esac
}
pids=()
for arm in baseline trainable sphere orthogonal; do
  if [ -f "$(_done_marker "$arm")" ]; then
    echo ">> $arm already complete — skipping"
    continue
  fi
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
