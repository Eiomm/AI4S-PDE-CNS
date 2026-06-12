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
用法：
  bash scripts/run_tmux_training.sh [name]       启动一次 tmux harness run。
  bash scripts/run_tmux_training.sh --status     查看 tmux 和最新 run 状态。
  bash scripts/run_tmux_training.sh --kill       结束 tmux 会话。

示例：
  bash scripts/run_tmux_training.sh official
  AGENT_ROUNDS=8 AGENT_PER_ROUND=64 bash scripts/run_tmux_training.sh final
  bash scripts/run_tmux_training.sh final --status
  bash scripts/run_tmux_training.sh final --kill

环境变量：
  AI4S_OUTPUTS_DIR=outputs            run 产物根目录。
  AI4S_TMUX_PREFIX=ai4s-chem          tmux 会话前缀。
  AI4S_TMUX_HOLD=0                    完成后允许 tmux pane 自动关闭。
  AI4S_CONDA_ENV=ai4s-chem-evolve     run_harness_once.sh 使用的 Conda 环境。
  PY=/path/to/python                  指定 Python 可执行文件。
EOF
}

die() {
  echo "错误：$*" >&2
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
  local state="已停止"
  if command -v "$TMUX_BIN" >/dev/null 2>&1 && have_session; then
    state="运行中"
  fi

  echo "项目根目录：  $ROOT_DIR"
  echo "输出根目录：  $OUTPUTS_DIR"
  echo "run 名称：    $RUN_NAME"
  echo "tmux 会话：   $SESSION"
  echo "状态：        $state"
  echo "最新 run：    $(latest_link || true)"
  echo "tmux 启动目录：$(latest_tmux_link || true)"
  echo

  local launch_dir
  launch_dir="$(latest_tmux_link || true)"
  if [[ -n "$launch_dir" && -f "$launch_dir/training.done" ]]; then
    echo "完成记录："
    sed 's/^/  /' "$launch_dir/training.done"
    echo
  fi

  local run_dir
  run_dir="$(latest_link || true)"
  if [[ -n "$run_dir" && -f "$run_dir/inspect.log" ]]; then
    echo "最新 inspect："
    tail -n 3 "$run_dir/inspect.log" | sed 's/^/  /'
    echo
  fi

  if [[ "$state" == "运行中" ]]; then
    echo "pane 最新输出："
    "$TMUX_BIN" capture-pane -t "$SESSION" -p 2>/dev/null | tail -n 12 | sed 's/^/  /'
  fi
}

kill_session() {
  if command -v "$TMUX_BIN" >/dev/null 2>&1 && have_session; then
    echo "正在结束 tmux 会话：$SESSION"
    "$TMUX_BIN" kill-session -t "$SESSION" || true
  else
    echo "tmux 会话未运行：$SESSION"
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
      die "未知参数：$1"
      ;;
  esac
fi

[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "名称只能使用字母、数字、点、下划线或短横线：$RUN_NAME"
command -v "$TMUX_BIN" >/dev/null 2>&1 || die "tmux 未安装或不在 PATH 中"
have_session && die "tmux 会话已存在：$SESSION"

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

echo "[tmux-training] 项目根目录：\$(pwd)"
echo "[tmux-training] tmux 会话：$(printf '%q' "$SESSION")"
echo "[tmux-training] 启动目录：$(printf '%q' "$LAUNCH_DIR")"
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
echo "[tmux-training] 退出码=\$rc，完成时间=\$finished_at"
echo "[tmux-training] 完成记录：$(printf '%q' "$LAUNCH_DIR")/training.done"
if [[ "\${AI4S_TMUX_HOLD:-1}" != "0" ]]; then
  echo "[tmux-training] 按 Enter 关闭这个 pane，或用 Ctrl-b d 分离会话"
  read -r _ || true
fi
exit "\$rc"
EOF
chmod +x "$RUNNER"

"$TMUX_BIN" new-session -d -s "$SESSION" "$RUNNER"

echo "已启动 tmux 会话：$SESSION"
echo "启动目录：$LAUNCH_DIR"
echo "查看状态：bash scripts/run_tmux_training.sh $RUN_NAME --status"
echo "进入会话：tmux attach -t $SESSION"
