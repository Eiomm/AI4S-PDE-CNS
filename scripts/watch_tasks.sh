#!/usr/bin/env bash
# Lightweight dashboard for scripts/run_all_tmux.sh.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INTERVAL="${WATCH_INTERVAL:-10}"
ONCE=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/watch_tasks.sh              Refresh every 10 seconds
  bash scripts/watch_tasks.sh --once       Print one status snapshot
  bash scripts/watch_tasks.sh --interval N Refresh every N seconds
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once)
      ONCE=1
      shift
      ;;
    --interval)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      INTERVAL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

print_snapshot() {
  echo "AI4S tmux dashboard - $(date -Is)"
  echo
  bash scripts/run_all_tmux.sh --status
}

if [[ "$ONCE" == "1" ]]; then
  print_snapshot
  exit 0
fi

while true; do
  clear || true
  print_snapshot
  sleep "$INTERVAL"
done
