#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

PY="${PY:-$ROOT_DIR/.venv/bin/python}"
MODEL="${AI4S_AIDE_MODEL:-claude-opus-4-7}"
PROXY_PORT="${AI4S_PROXY_PORT:-8080}"
PROXY_TARGET="${AI4S_PROXY_TARGET:-https://api.gpt.ge}"
DSLIGHTING_DEBUG="${AI4S_DSLIGHTING_DEBUG:-1}"
DSLIGHTING_DEBUG_CONSOLE="${AI4S_DSLIGHTING_DEBUG_CONSOLE:-1}"
PROVIDER_RAW_DEBUG="${AI4S_PROVIDER_RAW_DEBUG:-0}"
DSLIGHTING_DEBUG_NORMALIZED="$(printf '%s' "$DSLIGHTING_DEBUG" | tr '[:upper:]' '[:lower:]')"
case "$DSLIGHTING_DEBUG_NORMALIZED" in
  1|true|yes|on|debug)
    DSLIGHTING_DEBUG_FLAG="--llm-debug"
    ;;
  *)
    DSLIGHTING_DEBUG_FLAG=""
    ;;
esac
DSLIGHTING_DEBUG_CONSOLE_NORMALIZED="$(printf '%s' "$DSLIGHTING_DEBUG_CONSOLE" | tr '[:upper:]' '[:lower:]')"
case "$DSLIGHTING_DEBUG_CONSOLE_NORMALIZED" in
  1|true|yes|on)
    DSLIGHTING_DEBUG_CONSOLE_FLAG="--debug-console"
    ;;
  *)
    DSLIGHTING_DEBUG_CONSOLE_FLAG=""
    ;;
esac
PROVIDER_RAW_DEBUG_NORMALIZED="$(printf '%s' "$PROVIDER_RAW_DEBUG" | tr '[:upper:]' '[:lower:]')"
case "$PROVIDER_RAW_DEBUG_NORMALIZED" in
  1|true|yes|on|debug)
    PROVIDER_RAW_DEBUG_FLAG="--provider-raw-debug"
    LITELLM_LOG_LEVEL="DEBUG"
    LITELLM_SET_VERBOSE_VALUE="True"
    ;;
  *)
    PROVIDER_RAW_DEBUG_FLAG=""
    LITELLM_LOG_LEVEL="ERROR"
    LITELLM_SET_VERBOSE_VALUE="False"
    ;;
esac
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$ROOT_DIR/outputs/aide_task1_claude/$STAMP"
LOG_DIR="$RUN_ROOT/llm_io"
OUTPUT_DIR="$RUN_ROOT/prediction"
WORKSPACE_DIR="$RUN_ROOT/workspace"
DEBUG_DIR="$RUN_ROOT/dslighting_debug"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$WORKSPACE_DIR" "$DEBUG_DIR"
ln -sfn "$RUN_ROOT" "$ROOT_DIR/outputs/aide_task1_claude/latest"

if [[ ! -x "$PY" ]]; then
  echo "Python not found: $PY" >&2
  echo "Create the env first: python3.12 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
  exit 1
fi

"$PY" - <<PYCODE
import socket
port = int("$PROXY_PORT")
sock = socket.socket()
try:
    sock.bind(("127.0.0.1", port))
except OSError:
    raise SystemExit(f"Port {port} is already in use. Stop the old proxy or set AI4S_PROXY_PORT.")
finally:
    sock.close()
PYCODE

echo "[aide-task1] run root: $RUN_ROOT"
echo "[aide-task1] starting LLM I/O proxy on port $PROXY_PORT -> $PROXY_TARGET"
echo "[aide-task1] DSLighting structured debug: $DSLIGHTING_DEBUG"
echo "[aide-task1] DSLighting structured debug console: $DSLIGHTING_DEBUG_CONSOLE"
echo "[aide-task1] provider raw debug: $PROVIDER_RAW_DEBUG"

"$PY" task_log_sample/openai-log/proxy.py \
  --port "$PROXY_PORT" \
  --target "$PROXY_TARGET" \
  --log-dir "$LOG_DIR" \
  --log-level INFO \
  > "$RUN_ROOT/proxy_stdout_stderr.log" 2>&1 &

PROXY_PID=$!
cleanup() {
  if kill -0 "$PROXY_PID" >/dev/null 2>&1; then
    kill "$PROXY_PID" >/dev/null 2>&1 || true
    wait "$PROXY_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

"$PY" - <<PYCODE
import socket, time
port = int("$PROXY_PORT")
deadline = time.time() + 20
while time.time() < deadline:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            raise SystemExit(0)
    except OSError:
        time.sleep(0.25)
raise SystemExit("Proxy did not become ready within 20 seconds")
PYCODE

echo "[aide-task1] running AIDE with model=$MODEL"

AI4S_AGENT_MODEL="$MODEL" \
AI4S_AGENT_BASE_URL="http://127.0.0.1:${PROXY_PORT}/v1" \
AI4S_DSLIGHTING_DEBUG="$DSLIGHTING_DEBUG" \
AI4S_DSLIGHTING_DEBUG_CONSOLE="$DSLIGHTING_DEBUG_CONSOLE" \
AI4S_PROVIDER_RAW_DEBUG="$PROVIDER_RAW_DEBUG" \
LITELLM_LOG="$LITELLM_LOG_LEVEL" \
LITELLM_SET_VERBOSE="$LITELLM_SET_VERBOSE_VALUE" \
OPENAI_API_BASE="http://127.0.0.1:${PROXY_PORT}/v1" \
PYTHONUNBUFFERED=1 \
"$PY" scripts/run_ai4s_aide_task.py 1 \
  --model "$MODEL" \
  --api-base "http://127.0.0.1:${PROXY_PORT}/v1" \
  --provider openai \
  --output-dir "$OUTPUT_DIR" \
  --workspace-dir "$WORKSPACE_DIR" \
  --keep-workspace \
  --debug-dir "$DEBUG_DIR" \
  ${DSLIGHTING_DEBUG_FLAG:+"$DSLIGHTING_DEBUG_FLAG"} \
  ${DSLIGHTING_DEBUG_CONSOLE_FLAG:+"$DSLIGHTING_DEBUG_CONSOLE_FLAG"} \
  ${PROVIDER_RAW_DEBUG_FLAG:+"$PROVIDER_RAW_DEBUG_FLAG"} \
  "$@" \
  2>&1 | tee "$RUN_ROOT/aide_stdout_stderr.log"

if compgen -G "$DEBUG_DIR/debug_session_*" >/dev/null; then
  "$PY" scripts/export_dslighting_llm_io.py "$DEBUG_DIR" \
    --output "$RUN_ROOT/llm_io_normalized.jsonl" \
    | tee "$RUN_ROOT/llm_io_normalized_export.log"
fi

echo
echo "[aide-task1] prediction:"
echo "$OUTPUT_DIR/ai4s-pde-task1-burgers-fixed/task1_pred.hdf5"
echo
echo "[aide-task1] LLM input/output JSONL:"
ls "$LOG_DIR"/llm-*.jsonl 2>/dev/null || true
echo
echo "[aide-task1] normalized DSLighting LLM input/output JSONL:"
echo "$RUN_ROOT/llm_io_normalized.jsonl"
echo
echo "[aide-task1] DSLighting debug archive:"
ls -d "$DEBUG_DIR"/debug_session_* 2>/dev/null || true
echo
echo "[aide-task1] proxy log:"
echo "$RUN_ROOT/proxy_stdout_stderr.log"
echo
echo "[aide-task1] AIDE log:"
echo "$RUN_ROOT/aide_stdout_stderr.log"
echo
echo "[aide-task1] Use llm_io_normalized.jsonl for human-readable structured input/output. Enable raw provider logs only with AI4S_PROVIDER_RAW_DEBUG=1."
