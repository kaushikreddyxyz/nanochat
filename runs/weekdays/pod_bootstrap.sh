#!/usr/bin/env bash
# pod_bootstrap.sh — shared pod-side bootstrap for ALL 4 weekday-geometry runs.
# =============================================================================
# Clones the oracle-encodings superproject + the nanochat submodule at branch
# weekday-geometry, installs the uv env, wires HF / WANDB tokens WITHOUT ever
# putting a secret on argv, and launches a given run script under nohup.
#
# SECRETS — piped as a KEY=VALUE block on STDIN (never argv / never shell history),
# following the runpod-spinup token-over-stdin pattern. Recognized keys:
#     GH_TOKEN        private-repo clone (one-shot git credential helper; env-only)
#     HF_TOKEN        huggingface_hub auth (written to nanochat/.env, gitignored)
#     WANDB_API_KEY   (or WANDB_TOKEN) wandb auth (written to nanochat/.env)
# Keys already exported in the environment are used if not supplied on stdin.
# If a repo is public and no GH_TOKEN is given, SSH-agent forwarding (`ssh -A`)
# or an anonymous clone still works.
#
# USAGE (on the pod):
#   printf 'GH_TOKEN=%s\nHF_TOKEN=%s\nWANDB_API_KEY=%s\n' "$gh" "$hf" "$wb" \
#     | bash pod_bootstrap.sh exp1_baseline.sh
#
#   or via one ssh hop from the laptop (token still only crosses the pipe):
#   printf 'GH_TOKEN=%s\nHF_TOKEN=%s\nWANDB_API_KEY=%s\n' "$gh" "$hf" "$wb" \
#     | ssh runpod-<pod> 'bash -s' -- \
#         < <(curl -s .../pod_bootstrap.sh) exp1_baseline.sh
#
# ARGS:
#   $1 run_script : e.g. exp1_baseline.sh  (bare name -> runs/weekdays/<name>)
#                   or a full repo-relative path runs/weekdays/exp1_baseline.sh
#   $2 branch     : default weekday-geometry
#   $3 workdir    : default /workspace
# =============================================================================
set -euo pipefail

RUN_SCRIPT="${1:?usage: pod_bootstrap.sh <run_script> [branch] [workdir]}"
BRANCH="${2:-weekday-geometry}"
WORKDIR="${3:-/workspace}"
REPO_URL="https://github.com/kaushikreddyxyz/oracle-encodings.git"
REPO_DIR="$WORKDIR/oracle-encodings"

# bare filename -> runs/weekdays/<name>
case "$RUN_SCRIPT" in */*) : ;; *) RUN_SCRIPT="runs/weekdays/$RUN_SCRIPT" ;; esac

# --- read secrets from stdin (KEY=VALUE lines); never argv ------------------
if [ ! -t 0 ]; then
  while IFS='=' read -r k v; do
    [ -z "${k:-}" ] && continue
    case "$k" in
      GH_TOKEN|HF_TOKEN|WANDB_API_KEY|WANDB_TOKEN) export "$k=$v" ;;
    esac
  done
fi
# bridge WANDB_TOKEN -> WANDB_API_KEY (nanochat/common.py also does this)
if [ -n "${WANDB_TOKEN:-}" ] && [ -z "${WANDB_API_KEY:-}" ]; then export WANDB_API_KEY="$WANDB_TOKEN"; fi

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# --- clone superproject + nanochat submodule @ branch -----------------------
# IMPORTANT: the branch lives in the NANOCHAT SUBMODULE. The superproject's
# submodule pointer still points at nanochat main (the branch is unmerged, no
# pointer bump), so the superproject checkout is best-effort (it typically has
# no weekday-geometry branch -> stay on its default) while the SUBMODULE
# checkout is a HARD requirement, verified below.
if [ ! -d "$REPO_DIR/.git" ]; then
  if [ -n "${GH_TOKEN:-}" ]; then
    # one-shot credential helper: token stays in env, never in URL / argv / history
    git config --global credential.helper '!f() { echo username=x-access-token; echo "password=$GH_TOKEN"; }; f'
  fi
  echo ">> cloning $REPO_URL -> $REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
  git checkout "$BRANCH" 2>/dev/null \
    || echo ">> superproject has no branch $BRANCH (expected) — staying on its default branch"
  git submodule update --init --recursive
  # pin the nanochat submodule to the working branch (HARD: the run scripts,
  # configs, and framework hooks only exist on this branch)
  git -C nanochat fetch origin "$BRANCH"
  git -C nanochat checkout "$BRANCH"
  if [ -n "${GH_TOKEN:-}" ]; then git config --global --unset credential.helper || true; fi
else
  echo ">> repo present -> updating $REPO_DIR"
  cd "$REPO_DIR"
  git fetch --all --prune || true
  git checkout "$BRANCH" 2>/dev/null \
    || echo ">> superproject has no branch $BRANCH (expected) — staying on its current branch"
  git pull --ff-only || true
  git submodule update --init --recursive
  git -C nanochat fetch origin "$BRANCH"
  git -C nanochat checkout "$BRANCH"
  git -C nanochat merge --ff-only "origin/$BRANCH" || true
fi

# HARD verification: the submodule must be on the working branch, or nothing
# below (runs/weekdays/, the injection hooks) exists.
ACTUAL_BRANCH="$(git -C "$REPO_DIR/nanochat" rev-parse --abbrev-ref HEAD)"
if [ "$ACTUAL_BRANCH" != "$BRANCH" ]; then
  echo "ERROR: nanochat submodule is on '$ACTUAL_BRANCH', expected '$BRANCH'" >&2
  exit 1
fi
echo ">> nanochat submodule @ $BRANCH ($(git -C "$REPO_DIR/nanochat" rev-parse --short HEAD))"

NANO="$REPO_DIR/nanochat"

# --- write nanochat/.env (gitignored) so common.py auto-loads HF/WANDB ------
ENVF="$NANO/.env"
: > "$ENVF"
[ -n "${HF_TOKEN:-}" ]      && printf 'HF_TOKEN=%s\n'     "$HF_TOKEN"      >> "$ENVF"
[ -n "${WANDB_API_KEY:-}" ] && printf 'WANDB_TOKEN=%s\n'  "$WANDB_API_KEY" >> "$ENVF"
chmod 600 "$ENVF" || true

# --- uv env (nanochat convention) -------------------------------------------
cd "$NANO"
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ".venv" ] || uv venv
uv sync --extra gpu

# --- launch the run under nohup (survives the ssh session) ------------------
RUN_TAG="$(basename "$RUN_SCRIPT" .sh)"
LOG="$NANO/${RUN_TAG}.log"
PIDF="$NANO/${RUN_TAG}.pid"
echo ">> launching $RUN_SCRIPT (branch $BRANCH) under nohup"
# HF_TOKEN / WANDB_API_KEY are already exported -> inherited by the run + base_train
nohup bash "$RUN_SCRIPT" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDF"
echo ">> PID=$PID"
echo ">> log:  $LOG    (tail -f $LOG)"
echo ">> pid:  $PIDF"
