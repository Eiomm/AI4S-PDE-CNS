#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/ai4s-pde-cns/bin/python"

cd "$ROOT_DIR"

log() {
  printf '[one-click] %s\n' "$*" >&2
}

if [[ -f ".env" ]]; then
  log "加载 .env（不会打印密钥）"
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

PROXY_PORT="${AI4S_PROXY_PORT:-8080}"
PROXY_TARGET="${AI4S_PROXY_TARGET:-https://api.gpt.ge}"
PROXY_LOG_DIR="${AI4S_PROXY_LOG_DIR:-./logs}"
PROXY_PID_FILE="$ROOT_DIR/.proxy_${PROXY_PORT}.pid"

proxy_ready() {
  "$PYTHON_BIN" - <<PY
import socket
s = socket.socket()
s.settimeout(0.5)
try:
    s.connect(("127.0.0.1", int("$PROXY_PORT")))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PY
}

if proxy_ready; then
  log "proxy 已在 127.0.0.1:${PROXY_PORT} 运行"
else
  log "启动 proxy：127.0.0.1:${PROXY_PORT} -> ${PROXY_TARGET}"
  mkdir -p "$PROXY_LOG_DIR"
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid "$PYTHON_BIN" task_log_sample/openai-log/proxy.py \
      --port "$PROXY_PORT" \
      --target "$PROXY_TARGET" \
      --log-dir "$PROXY_LOG_DIR" \
      > "$PROXY_LOG_DIR/proxy_stdout.log" 2>&1 &
  else
    nohup "$PYTHON_BIN" task_log_sample/openai-log/proxy.py \
      --port "$PROXY_PORT" \
      --target "$PROXY_TARGET" \
      --log-dir "$PROXY_LOG_DIR" \
      > "$PROXY_LOG_DIR/proxy_stdout.log" 2>&1 &
  fi
  echo "$!" > "$PROXY_PID_FILE"
  for _ in {1..20}; do
    if proxy_ready; then
      log "proxy 已就绪，pid=$(cat "$PROXY_PID_FILE")"
      break
    fi
    sleep 0.5
  done
  if ! proxy_ready; then
    log "proxy 启动失败，最近日志如下："
    tail -n 80 "$PROXY_LOG_DIR/proxy_stdout.log" >&2 || true
    exit 1
  fi
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  log "执行 Agent dry-run，只生成请求上下文"
  exec bash scripts/run_in_env.sh scripts/task1_agent_runner.py --config "${AI4S_AGENT_CONFIG:-configs/agent_gpt55.yaml}" --dry-run
fi

GOAL="${AI4S_AGENT_GOAL:-生成 Task1 可运行代码；必须在终端打印训练/搜索进展，包括 epoch、trial、score、best checkpoint、耗时；Optuna 应在 Agent 生成的 Python 代码中直接 import optuna 使用。}"
AGENT_CONFIG="${AI4S_AGENT_CONFIG:-configs/agent_gpt55.yaml}"

log "启动 Agent runner"
log "goal: ${GOAL}"
log "config: ${AGENT_CONFIG}"
bash scripts/run_in_env.sh scripts/task1_agent_runner.py --config "$AGENT_CONFIG" --goal "$GOAL"

LATEST_SUMMARY="$(ls -1t agent_workspace/logs/agent_*/summary.json 2>/dev/null | head -1 || true)"
if [[ -z "$LATEST_SUMMARY" ]]; then
  log "没有找到 Agent summary，停止。"
  exit 1
fi

log "执行 Agent tool_requests：${LATEST_SUMMARY}"
bash scripts/run_in_env.sh scripts/task1_tool_executor.py --summary "$LATEST_SUMMARY"

log "完成。常用查看命令："
log "  tail -n 80 ${LATEST_SUMMARY}"
log "  find agent_workspace/code -maxdepth 3 -type f -print"
log "  tail -n 80 logs/proxy_stdout.log"
