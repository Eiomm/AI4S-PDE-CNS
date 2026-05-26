#!/usr/bin/env bash
# Watch the isolated output1 full-run status on ports 8090/8091/8092.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INTERVAL="${1:-10}"
CMD=(bash scripts/run_all_tmux_experiment.sh output1 8090 --status)

if command -v watch >/dev/null 2>&1; then
  exec watch -n "$INTERVAL" "${CMD[*]}"
fi

while true; do
  clear
  "${CMD[@]}"
  sleep "$INTERVAL"
done
