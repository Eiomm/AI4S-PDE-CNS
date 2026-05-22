#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/root/miniconda3/envs/ai4s-pde-cns/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing fixed Python environment: $PYTHON_BIN" >&2
  exit 1
fi

exec "$PYTHON_BIN" "$@"
