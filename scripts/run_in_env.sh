#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/root/miniconda3/envs/ai4s-pde-cns/bin/python"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing fixed Python environment: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$@"
