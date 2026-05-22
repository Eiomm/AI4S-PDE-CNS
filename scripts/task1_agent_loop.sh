#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/ai4s-pde-cns/bin/python"

cd "$ROOT_DIR"

log() {
  printf '[agent-loop] %s\n' "$*" >&2
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

exec bash scripts/run_in_env.sh scripts/task1_agent_loop.py "$@"
