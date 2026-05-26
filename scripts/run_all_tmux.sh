#!/usr/bin/env bash
# Launch the three AI4S AIDE tasks in tmux and package the final submission
# after all task sessions have completed.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TMUX_BIN="${TMUX_BIN:-tmux}"
TMUX_PREFIX="${AI4S_TMUX_PREFIX:-ai4s}"
TASK_SESSIONS=("${TMUX_PREFIX}-task1" "${TMUX_PREFIX}-task2" "${TMUX_PREFIX}-task3")
ASSEMBLE_SESSION="${TMUX_PREFIX}-assemble"
ALL_SESSIONS=("${TASK_SESSIONS[@]}" "$ASSEMBLE_SESSION")
OUTPUTS_DIR="${AI4S_OUTPUTS_DIR:-$ROOT_DIR/outputs}"

PORT_T1="${AI4S_PROXY_PORT_T1:-8080}"
PORT_T2="${AI4S_PROXY_PORT_T2:-8081}"
PORT_T3="${AI4S_PROXY_PORT_T3:-8082}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_all_tmux.sh           Start task1/task2/task3 + assembler
  bash scripts/run_all_tmux.sh --status  Show tmux/task status
  bash scripts/run_all_tmux.sh --kill    Kill AI4S tmux sessions

Environment:
  PY=/path/to/python                     Python executable for task scripts
  AI4S_CONDA_ENV=ai4s-pde-cns            Conda env name to use when PY is unset
  AI4S_PROXY_PORT_T1=8080                Proxy port for task 1
  AI4S_PROXY_PORT_T2=8081                Proxy port for task 2
  AI4S_PROXY_PORT_T3=8082                Proxy port for task 3
  AI4S_OUTPUTS_DIR=outputs/output1       Output root for this full run
  AI4S_TMUX_PREFIX=ai4s-output1          Tmux session prefix for parallel runs
  AI4S_TMUX_HOLD=0                       Let finished panes close instead of waiting
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

have_session() {
  "$TMUX_BIN" has-session -t "$1" >/dev/null 2>&1
}

resolve_python() {
  if [[ -n "${PY:-}" ]]; then
    [[ -x "$PY" ]] || die "PY is not executable: $PY"
    printf '%s\n' "$PY"
    return 0
  fi

  local env_name="${AI4S_CONDA_ENV:-}"
  if [[ -z "$env_name" ]] && command -v conda >/dev/null 2>&1; then
    env_name="$(conda env list 2>/dev/null | awk '$1 ~ /^ai4/ {print $1; exit}')"
  fi
  env_name="${env_name:-ai4s-pde-cns}"

  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "$conda_base" && -x "$conda_base/envs/$env_name/bin/python" ]]; then
      printf '%s\n' "$conda_base/envs/$env_name/bin/python"
      return 0
    fi
  fi

  if [[ -x "/root/miniconda3/envs/$env_name/bin/python" ]]; then
    printf '%s\n' "/root/miniconda3/envs/$env_name/bin/python"
    return 0
  fi

  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    printf '%s\n' "$ROOT_DIR/.venv/bin/python"
    return 0
  fi

  command -v python3 >/dev/null 2>&1 || die "no Python found; set PY=/path/to/python"
  command -v python3
}

check_launch_preconditions() {
  command -v "$TMUX_BIN" >/dev/null 2>&1 || die "tmux is not installed or not in PATH"

  if [[ -f "$ROOT_DIR/.env" ]] && grep -Eq '^[[:space:]]*AI4S_PROXY_PORT[[:space:]]*=' "$ROOT_DIR/.env"; then
    cat >&2 <<'EOF'
error: .env defines AI4S_PROXY_PORT.

scripts/run_all_tmux.sh assigns separate ports to each task (8080/8081/8082).
Remove or comment out AI4S_PROXY_PORT from .env before launching tmux,
otherwise all task scripts will source .env and collide on the same port.
EOF
    exit 1
  fi

  for script in \
    "$ROOT_DIR/scripts/aide_task1_claude_one_click.sh" \
    "$ROOT_DIR/scripts/aide_task2_claude_one_click.sh" \
    "$ROOT_DIR/scripts/aide_task3_claude_one_click.sh" \
    "$ROOT_DIR/scripts/build_final_submission.py"; do
    [[ -e "$script" ]] || die "missing required file: $script"
  done
}

kill_sessions() {
  for session in "${ALL_SESSIONS[@]}"; do
    if have_session "$session"; then
      echo "killing tmux session: $session"
      "$TMUX_BIN" kill-session -t "$session" || true
    fi
  done
}

latest_launch_dir() {
  if [[ -L "$OUTPUTS_DIR/tmux_run_latest" || -e "$OUTPUTS_DIR/tmux_run_latest" ]]; then
    readlink "$OUTPUTS_DIR/tmux_run_latest" 2>/dev/null || true
  fi
}

done_exit_code() {
  local done_file="$1"
  [[ -f "$done_file" ]] || return 1
  awk -F= '$1 == "exit_code" {print $2; found=1} END {exit found ? 0 : 1}' "$done_file"
}

count_llm_calls() {
  local task="$1"
  local latest="$OUTPUTS_DIR/aide_task${task}_claude/latest"
  local log_file=""
  if [[ -d "$latest/llm_io" ]]; then
    log_file="$(find "$latest/llm_io" -maxdepth 1 -name 'llm-*.jsonl' -type f -print | sort | tail -n 1 || true)"
  fi
  if [[ -n "$log_file" && -f "$log_file" ]]; then
    wc -l < "$log_file" | tr -d ' '
  else
    printf '0'
  fi
}

show_status() {
  local python_bin
  python_bin="$(resolve_python)"

  echo "root:        $ROOT_DIR"
  echo "outputs:     $OUTPUTS_DIR"
  echo "tmux prefix: $TMUX_PREFIX"
  echo "tmux:        $(command -v "$TMUX_BIN" 2>/dev/null || printf 'missing')"
  echo "python:      $python_bin"
  if [[ -f "$ROOT_DIR/.env" ]] && grep -Eq '^[[:space:]]*AI4S_PROXY_PORT[[:space:]]*=' "$ROOT_DIR/.env"; then
    echo "env warning: .env contains AI4S_PROXY_PORT; launch will fail until it is removed/commented."
  fi
  echo "latest run:  $(latest_launch_dir || true)"
  echo

  printf '%-14s %-9s %-7s %-10s %-9s %-s\n' "session" "state" "port" "done" "llm_calls" "aide_process"
  printf '%-14s %-9s %-7s %-10s %-9s %-s\n' "-------" "-----" "----" "----" "---------" "------------"

  local launch_dir
  launch_dir="$(latest_launch_dir || true)"
  for task in 1 2 3; do
    local session="${TASK_SESSIONS[$((task - 1))]}"
    local port="$PORT_T1"
    [[ "$task" == "2" ]] && port="$PORT_T2"
    [[ "$task" == "3" ]] && port="$PORT_T3"

    local state="stopped"
    have_session "$session" && state="running"

    local done_state="-"
    if [[ -n "$launch_dir" && -f "$launch_dir/task${task}.done" ]]; then
      local code
      code="$(done_exit_code "$launch_dir/task${task}.done" || printf '?')"
      done_state="exit=$code"
    fi

    local proc="-"
    if pgrep -af "[r]un_ai4s_aide_task.py ${task} .*--output-dir ${OUTPUTS_DIR}/aide_task${task}_claude" >/dev/null 2>&1; then
      proc="yes"
    fi

    printf '%-14s %-9s %-7s %-10s %-9s %-s\n' \
      "$session" "$state" "$port" "$done_state" "$(count_llm_calls "$task")" "$proc"
  done

  local assemble_state="stopped"
  have_session "$ASSEMBLE_SESSION" && assemble_state="running"
  local assemble_done="-"
  if [[ -n "$launch_dir" && -f "$launch_dir/assemble.done" ]]; then
    local code
    code="$(done_exit_code "$launch_dir/assemble.done" || printf '?')"
    assemble_done="exit=$code"
  fi
  printf '%-14s %-9s %-7s %-10s %-9s %-s\n' "$ASSEMBLE_SESSION" "$assemble_state" "-" "$assemble_done" "-" "-"

  echo
  for session in "${ALL_SESSIONS[@]}"; do
    if have_session "$session"; then
      echo "[$session] latest pane lines:"
      "$TMUX_BIN" capture-pane -t "$session" -p 2>/dev/null | tail -n 3 | sed 's/^/  /'
    fi
  done
}

write_task_runner() {
  local task="$1"
  local port="$2"
  local python_bin="$3"
  local launch_dir="$4"
  local script="$ROOT_DIR/scripts/aide_task${task}_claude_one_click.sh"
  local runner="$launch_dir/task${task}_runner.sh"

  cat > "$runner" <<EOF
#!/usr/bin/env bash
set -u
cd "$(printf '%q' "$ROOT_DIR")"
task="$task"
port="$port"
python_bin="$(printf '%q' "$python_bin")"
task_script="$(printf '%q' "$script")"
launch_dir="$(printf '%q' "$launch_dir")"
started_at="\$(date -Is)"
started_epoch="\$(date +%s)"

echo "[tmux-task\${task}] root: \$(pwd)"
echo "[tmux-task\${task}] python: \$python_bin"
echo "[tmux-task\${task}] proxy port: \$port"
echo "[tmux-task\${task}] script: \$task_script"
echo

PY="\$python_bin" AI4S_PROXY_PORT="\$port" AI4S_OUTPUTS_DIR="$(printf '%q' "$OUTPUTS_DIR")" bash "\$task_script"
rc=\$?
finished_at="\$(date -Is)"
finished_epoch="\$(date +%s)"

{
  echo "task=\$task"
  echo "exit_code=\$rc"
  echo "started_at=\$started_at"
  echo "finished_at=\$finished_at"
  echo "duration_seconds=\$((finished_epoch - started_epoch))"
} > "\$launch_dir/task\${task}.done.tmp"
mv "\$launch_dir/task\${task}.done.tmp" "\$launch_dir/task\${task}.done"

echo
echo "[tmux-task\${task}] exit=\$rc at \$finished_at"
echo "[tmux-task\${task}] done file: \$launch_dir/task\${task}.done"
if [[ "\${AI4S_TMUX_HOLD:-1}" != "0" ]]; then
  echo "[tmux-task\${task}] press Enter to close this pane, or detach with Ctrl-b d"
  read -r _ || true
fi
exit "\$rc"
EOF
  chmod +x "$runner"
}

write_assemble_runner() {
  local python_bin="$1"
  local launch_dir="$2"
  local runner="$launch_dir/assemble_runner.sh"

  cat > "$runner" <<EOF
#!/usr/bin/env bash
set -u
cd "$(printf '%q' "$ROOT_DIR")"
python_bin="$(printf '%q' "$python_bin")"
launch_dir="$(printf '%q' "$launch_dir")"
outputs_dir="$(printf '%q' "$OUTPUTS_DIR")"
started_at="\$(date -Is)"
started_epoch="\$(date +%s)"

echo "[assemble] waiting for task done files under \$launch_dir"
while true; do
  missing=0
  for task in 1 2 3; do
    [[ -f "\$launch_dir/task\${task}.done" ]] || missing=\$((missing + 1))
  done
  [[ "\$missing" -eq 0 ]] && break
  echo "[assemble] waiting: \$missing task(s) still running at \$(date -Is)"
  sleep 10
done

failed=0
for task in 1 2 3; do
  code="\$(awk -F= '\$1 == "exit_code" {print \$2}' "\$launch_dir/task\${task}.done")"
  echo "[assemble] task\${task} exit=\$code"
  [[ "\$code" == "0" ]] || failed=1
done

if [[ "\$failed" -ne 0 ]]; then
  rc=1
  echo "[assemble] one or more tasks failed; skipping final packaging"
else
  if [[ -f scripts/build_methodology_pdf.py ]]; then
    "\$python_bin" scripts/build_methodology_pdf.py
  fi
  AI4S_OUTPUTS_DIR="\$outputs_dir" "\$python_bin" scripts/build_final_submission.py --out "\$outputs_dir/submission" --zip
  rc=\$?
fi

finished_at="\$(date -Is)"
finished_epoch="\$(date +%s)"
{
  echo "task=assemble"
  echo "exit_code=\$rc"
  echo "started_at=\$started_at"
  echo "finished_at=\$finished_at"
  echo "duration_seconds=\$((finished_epoch - started_epoch))"
} > "\$launch_dir/assemble.done.tmp"
mv "\$launch_dir/assemble.done.tmp" "\$launch_dir/assemble.done"

echo
echo "[assemble] exit=\$rc at \$finished_at"
echo "[assemble] done file: \$launch_dir/assemble.done"
if [[ "\${AI4S_TMUX_HOLD:-1}" != "0" ]]; then
  echo "[assemble] press Enter to close this pane, or detach with Ctrl-b d"
  read -r _ || true
fi
exit "\$rc"
EOF
  chmod +x "$runner"
}

start_all() {
  check_launch_preconditions

  for session in "${ALL_SESSIONS[@]}"; do
    if have_session "$session"; then
      die "tmux session already exists: $session (run --kill first or attach to inspect)"
    fi
  done

  local python_bin
  python_bin="$(resolve_python)"

  local stamp launch_dir
  stamp="$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$OUTPUTS_DIR"
  launch_dir="$OUTPUTS_DIR/tmux_run_$stamp"
  mkdir -p "$launch_dir"
  ln -sfn "$launch_dir" "$OUTPUTS_DIR/tmux_run_latest"

  write_task_runner 1 "$PORT_T1" "$python_bin" "$launch_dir"
  write_task_runner 2 "$PORT_T2" "$python_bin" "$launch_dir"
  write_task_runner 3 "$PORT_T3" "$python_bin" "$launch_dir"
  write_assemble_runner "$python_bin" "$launch_dir"

  echo "launch dir: $launch_dir"
  echo "outputs:    $OUTPUTS_DIR"
  echo "sessions:   ${TASK_SESSIONS[*]} $ASSEMBLE_SESSION"
  echo "python:     $python_bin"

  "$TMUX_BIN" new-session -d -s "${TASK_SESSIONS[0]}" -c "$ROOT_DIR" "bash '$launch_dir/task1_runner.sh'"
  "$TMUX_BIN" new-session -d -s "${TASK_SESSIONS[1]}" -c "$ROOT_DIR" "bash '$launch_dir/task2_runner.sh'"
  "$TMUX_BIN" new-session -d -s "${TASK_SESSIONS[2]}" -c "$ROOT_DIR" "bash '$launch_dir/task3_runner.sh'"
  "$TMUX_BIN" new-session -d -s "$ASSEMBLE_SESSION" -c "$ROOT_DIR" "bash '$launch_dir/assemble_runner.sh'"

  echo
  show_status
}

case "${1:-}" in
  "")
    start_all
    ;;
  --status)
    show_status
    ;;
  --kill)
    kill_sessions
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
