#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/ai4s-pde-cns/bin/python"

cd "$ROOT_DIR"

# 自动加载 Task1 根目录下的 .env。这里不打印任何密钥，避免泄露。
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

exec "$PYTHON_BIN" task_log_sample/openai-log/proxy.py \
  --port "${AI4S_PROXY_PORT:-8080}" \
  --target "${AI4S_PROXY_TARGET:-https://api.gpt.ge}" \
  --log-dir "${AI4S_PROXY_LOG_DIR:-./logs}"
