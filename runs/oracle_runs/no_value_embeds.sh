#!/bin/bash
#
# no_value_embeds.sh — oracle-encodings negative control, NO value embeddings
# =============================================================================
# IDENTICAL to baseline.sh in every respect EXCEPT that the ResFormer value
# embeddings are zeroed and frozen (--no-value-embeds), so they neither
# contribute to the forward nor learn. This is the matched control for studying
# what the learned value embeddings do: baseline.sh learns them, this one does not.
#
# It does NOT duplicate any logic -- it simply sets NO_VALUE_EMBEDS=1 and execs
# baseline.sh, so the two runs can never drift apart. Given the same SEED (default
# 1337), every other parameter is bit-identical at init between the two runs; the
# value-embedding tables are zeroed AFTER init_weights so RNG order is preserved.
#
# All of baseline.sh's knobs still apply, e.g.:
#   bash runs/oracle_runs/no_value_embeds.sh
#   DEPTH=24 DEVICE_BATCH_SIZE=8 bash runs/oracle_runs/no_value_embeds.sh
#   SMOKE=1 bash runs/oracle_runs/no_value_embeds.sh
#
# Checkpoints/wandb get a "_noVE" suffix automatically so they never collide with
# the baseline (tag: oracle_baseline_noVE_d<DEPTH>).
# =============================================================================

set -euo pipefail
export NO_VALUE_EMBEDS=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/baseline.sh" "$@"
