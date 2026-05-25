#!/usr/bin/env bash
# Full pipeline: run AIDE on all 3 tasks, then assemble submission/.
#
# Steps:
#   1. (optional) regenerate methodology.pdf from build_methodology_pdf.py
#   2. run aide_task1_claude_one_click.sh
#   3. run aide_task2_claude_one_click.sh
#   4. run aide_task3_claude_one_click.sh
#   5. run build_final_submission.py --zip
#
# Each AIDE run uses its own port (default 8080/8081/8082) so they cannot
# collide. By default the three runs are SEQUENTIAL; set PARALLEL=1 to
# launch them as background processes.
#
# Skip individual tasks with SKIP_TASK1=1 / SKIP_TASK2=1 / SKIP_TASK3=1.
# Skip the AIDE step entirely (only re-pack existing runs) with SKIP_RUN=1.
#
# Usage:
#   bash scripts/run_all_and_submit.sh                 # sequential, all 3
#   PARALLEL=1 bash scripts/run_all_and_submit.sh      # parallel
#   SKIP_RUN=1 bash scripts/run_all_and_submit.sh      # just repackage
#   SKIP_TASK3=1 bash scripts/run_all_and_submit.sh    # tasks 1+2 only

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PY:-$ROOT_DIR/.venv/bin/python}"
SKIP_RUN="${SKIP_RUN:-0}"
SKIP_TASK1="${SKIP_TASK1:-0}"
SKIP_TASK2="${SKIP_TASK2:-0}"
SKIP_TASK3="${SKIP_TASK3:-0}"
PARALLEL="${PARALLEL:-0}"
DO_ZIP="${DO_ZIP:-1}"

PORT_T1="${AI4S_PROXY_PORT_T1:-8080}"
PORT_T2="${AI4S_PROXY_PORT_T2:-8081}"
PORT_T3="${AI4S_PROXY_PORT_T3:-8082}"

# Sequential timestamps separate per-task logs.
STAMP="$(date +%Y%m%d_%H%M%S)"
PIPELINE_LOG="$ROOT_DIR/outputs/pipeline_${STAMP}.log"
mkdir -p "$ROOT_DIR/outputs"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$PIPELINE_LOG"; }

# ---------- 1. methodology.pdf ----------------------------------------------

if [[ -f "$ROOT_DIR/scripts/build_methodology_pdf.py" ]]; then
  log "(re)generating methodology.pdf"
  "$PY" "$ROOT_DIR/scripts/build_methodology_pdf.py" 2>&1 | tee -a "$PIPELINE_LOG"
else
  log "warning: scripts/build_methodology_pdf.py not found"
fi

# ---------- 2-4. run AIDE on each task -------------------------------------

run_one_task() {
  local task=$1
  local port=$2
  local skip_var=$3
  local skip_val=${!skip_var}
  local script="$ROOT_DIR/scripts/aide_task${task}_claude_one_click.sh"

  if [[ "$skip_val" == "1" ]]; then
    log "task ${task}: SKIPPED (${skip_var}=1)"
    return 0
  fi
  if [[ ! -x "$script" ]]; then
    log "task ${task}: script not found or not executable: $script"
    return 1
  fi
  log "task ${task}: starting (port $port)"
  AI4S_PROXY_PORT=$port bash "$script" 2>&1 | tee -a "$PIPELINE_LOG"
  log "task ${task}: finished"
}

if [[ "$SKIP_RUN" == "1" ]]; then
  log "SKIP_RUN=1 — skipping all AIDE runs; will only repackage existing outputs"
else
  if [[ "$PARALLEL" == "1" ]]; then
    log "PARALLEL=1 — running all 3 tasks concurrently in background"
    run_one_task 1 "$PORT_T1" SKIP_TASK1 &
    P1=$!
    run_one_task 2 "$PORT_T2" SKIP_TASK2 &
    P2=$!
    run_one_task 3 "$PORT_T3" SKIP_TASK3 &
    P3=$!
    wait "$P1" "$P2" "$P3" || true
  else
    run_one_task 1 "$PORT_T1" SKIP_TASK1 || log "task 1 failed (continuing)"
    run_one_task 2 "$PORT_T2" SKIP_TASK2 || log "task 2 failed (continuing)"
    run_one_task 3 "$PORT_T3" SKIP_TASK3 || log "task 3 failed (continuing)"
  fi
fi

# ---------- 5. assemble submission/ ----------------------------------------

log "assembling submission/"
TASKS_ARG=()
[[ "$SKIP_TASK1" != "1" ]] && TASKS_ARG+=(1)
[[ "$SKIP_TASK2" != "1" ]] && TASKS_ARG+=(2)
[[ "$SKIP_TASK3" != "1" ]] && TASKS_ARG+=(3)

if [[ ${#TASKS_ARG[@]} -eq 0 ]]; then
  log "no tasks selected; nothing to package"
  exit 0
fi

ZIP_FLAG=""
[[ "$DO_ZIP" == "1" ]] && ZIP_FLAG="--zip"

"$PY" "$ROOT_DIR/scripts/build_final_submission.py" \
  --tasks "${TASKS_ARG[@]}" \
  $ZIP_FLAG \
  2>&1 | tee -a "$PIPELINE_LOG"

log "pipeline complete. submission at: $ROOT_DIR/submission/"
[[ "$DO_ZIP" == "1" ]] && log "zipped:                $ROOT_DIR/submission.zip"
log "full log:              $PIPELINE_LOG"
