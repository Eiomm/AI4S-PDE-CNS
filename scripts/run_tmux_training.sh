#!/usr/bin/env bash
# Launch the chemistry harness in tmux. This is intentionally small: one
# background session runs the same tests -> generate -> inspect loop.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TMUX_BIN="${TMUX_BIN:-tmux}"
OUTPUTS_DIR="${AI4S_OUTPUTS_DIR:-$ROOT_DIR/outputs}"
CONDA_ENV="${AI4S_CONDA_ENV:-ai4s-chem-evolve}"
SKIP_TESTS="${AI4S_SKIP_TESTS:-0}"
RUN_NAME="${AI4S_RUN_NAME:-target_minimal}"
TARGET="${AI4S_TARGET:-$ROOT_DIR/target.pdb}"
ROUNDS="${AGENT_ROUNDS:-4}"
PER_ROUND="${AGENT_PER_ROUND:-12}"
TOP_K="${AGENT_TOP_K:-20}"
MODE="${AGENT_MODE:-docking}"
DOCKING_LIMIT="${AGENT_DOCKING_LIMIT:-6}"
TMUX_PREFIX="${AI4S_TMUX_PREFIX:-ai4s-chem}"
SESSION="${TMUX_PREFIX}-${RUN_NAME}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_tmux_training.sh [name]       Start one tmux harness run.
  bash scripts/run_tmux_training.sh --status     Show tmux and latest-run status.
  bash scripts/run_tmux_training.sh --kill       Kill the tmux session.

Examples:
  bash scripts/run_tmux_training.sh official
  AGENT_ROUNDS=8 AGENT_PER_ROUND=64 bash scripts/run_tmux_training.sh final
  bash scripts/run_tmux_training.sh final --status
  bash scripts/run_tmux_training.sh final --kill

Environment:
  AI4S_OUTPUTS_DIR=outputs            Root directory for run artifacts.
  AI4S_TMUX_PREFIX=ai4s-chem          Session prefix.
  AI4S_TMUX_HOLD=0                    Let finished tmux panes close.
  AI4S_CONDA_ENV=ai4s-chem-evolve     Conda env used by run_harness_once.sh.
  PY=/path/to/python                  Python executable override.
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

have_session() {
  "$TMUX_BIN" has-session -t "$SESSION" >/dev/null 2>&1
}

latest_link() {
  local path="$OUTPUTS_DIR/$RUN_NAME/latest"
  if [[ -L "$path" || -e "$path" ]]; then
    readlink "$path" 2>/dev/null || printf '%s\n' "$path"
  fi
}

latest_tmux_link() {
  local path="$OUTPUTS_DIR/$RUN_NAME/tmux_latest"
  if [[ -L "$path" || -e "$path" ]]; then
    readlink "$path" 2>/dev/null || printf '%s\n' "$path"
  fi
}

show_status() {
  local state="stopped"
  if command -v "$TMUX_BIN" >/dev/null 2>&1 && have_session; then
    state="running"
  fi

  echo "root:        $ROOT_DIR"
  echo "outputs:     $OUTPUTS_DIR"
  echo "run name:    $RUN_NAME"
  echo "session:     $SESSION"
  echo "state:       $state"
  echo "latest run:  $(latest_link || true)"
  echo "tmux launch: $(latest_tmux_link || true)"
  echo

  local launch_dir
  launch_dir="$(latest_tmux_link || true)"
  if [[ -n "$launch_dir" && -f "$launch_dir/training.done" ]]; then
    echo "done:"
    sed 's/^/  /' "$launch_dir/training.done"
    echo
  fi

  local run_dir
  run_dir="$(latest_link || true)"
  if [[ -n "$run_dir" && -f "$run_dir/inspect.log" ]]; then
    echo "latest inspect:"
    tail -n 3 "$run_dir/inspect.log" | sed 's/^/  /'
    echo
  fi

  if [[ "$state" == "running" ]]; then
    echo "pane tail:"
    "$TMUX_BIN" capture-pane -t "$SESSION" -p 2>/dev/null | tail -n 12 | sed 's/^/  /'
  fi
}

kill_session() {
  if command -v "$TMUX_BIN" >/dev/null 2>&1 && have_session; then
    echo "killing tmux session: $SESSION"
    "$TMUX_BIN" kill-session -t "$SESSION" || true
  else
    echo "tmux session is not running: $SESSION"
  fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  RUN_NAME="$1"
  SESSION="${TMUX_PREFIX}-${RUN_NAME}"
  shift
fi

if [[ $# -gt 0 ]]; then
  case "$1" in
    --status)
      show_status
      exit 0
      ;;
    --kill)
      kill_session
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
fi

[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "name must use only letters, numbers, dot, underscore, or dash: $RUN_NAME"
command -v "$TMUX_BIN" >/dev/null 2>&1 || die "tmux is not installed or not in PATH"
have_session && die "tmux session already exists: $SESSION"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_PARENT="$OUTPUTS_DIR/$RUN_NAME"
LAUNCH_DIR="$RUN_PARENT/tmux_$STAMP"
RUNNER="$LAUNCH_DIR/training_runner.sh"
if [[ -n "${PY:-}" ]]; then
  PY_EXPORT_LINE="export PY=$(printf '%q' "$PY")"
else
  PY_EXPORT_LINE="unset PY"
fi

mkdir -p "$LAUNCH_DIR"
ln -sfn "$LAUNCH_DIR" "$RUN_PARENT/tmux_latest"

cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -u
cd "$(printf '%q' "$ROOT_DIR")"
started_at="\$(date -Is)"
started_epoch="\$(date +%s)"

echo "[tmux-training] root: \$(pwd)"
echo "[tmux-training] session: $(printf '%q' "$SESSION")"
echo "[tmux-training] launch: $(printf '%q' "$LAUNCH_DIR")"
echo

export AI4S_OUTPUTS_DIR="$(printf '%q' "$OUTPUTS_DIR")"
export AI4S_CONDA_ENV="$(printf '%q' "$CONDA_ENV")"
export AI4S_SKIP_TESTS="$(printf '%q' "$SKIP_TESTS")"
export AI4S_TMUX_HOLD="$(printf '%q' "${AI4S_TMUX_HOLD:-1}")"
$PY_EXPORT_LINE

bash scripts/run_harness_once.sh \
  --name "$(printf '%q' "$RUN_NAME")" \
  --target "$(printf '%q' "$TARGET")" \
  --rounds "$(printf '%q' "$ROUNDS")" \
  --per-round "$(printf '%q' "$PER_ROUND")" \
  --top-k "$(printf '%q' "$TOP_K")" \
  --mode "$(printf '%q' "$MODE")" \
  --docking-limit "$(printf '%q' "$DOCKING_LIMIT")"
rc=\$?
finished_at="\$(date -Is)"
finished_epoch="\$(date +%s)"

{
  echo "exit_code=\$rc"
  echo "session=$(printf '%q' "$SESSION")"
  echo "run_name=$(printf '%q' "$RUN_NAME")"
  echo "started_at=\$started_at"
  echo "finished_at=\$finished_at"
  echo "duration_seconds=\$((finished_epoch - started_epoch))"
} > "$(printf '%q' "$LAUNCH_DIR")/training.done.tmp"
mv "$(printf '%q' "$LAUNCH_DIR")/training.done.tmp" "$(printf '%q' "$LAUNCH_DIR")/training.done"

echo
echo "[tmux-training] exit=\$rc at \$finished_at"
echo "[tmux-training] done file: $(printf '%q' "$LAUNCH_DIR")/training.done"
if [[ "\${AI4S_TMUX_HOLD:-1}" != "0" ]]; then
  echo "[tmux-training] press Enter to close this pane, or detach with Ctrl-b d"
  read -r _ || true
fi
exit "\$rc"
EOF
chmod +x "$RUNNER"

"$TMUX_BIN" new-session -d -s "$SESSION" "$RUNNER"

echo "started tmux session: $SESSION"
echo "launch dir: $LAUNCH_DIR"
echo "status: bash scripts/run_tmux_training.sh $RUN_NAME --status"
echo "attach: tmux attach -t $SESSION"
