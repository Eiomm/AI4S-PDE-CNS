#!/usr/bin/env bash
# Launch an isolated full AI4S tmux experiment under outputs/<name>.
#
# Examples:
#   bash scripts/run_all_tmux_experiment.sh output1 8090
#   bash scripts/run_all_tmux_experiment.sh output1 --status
#   bash scripts/run_all_tmux_experiment.sh output1 --kill
#   bash scripts/run_all_tmux_experiment.sh output2 8100

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_all_tmux_experiment.sh [name] [port_base] [run_all_arg]

Examples:
  bash scripts/run_all_tmux_experiment.sh output1 8090
  bash scripts/run_all_tmux_experiment.sh output1 --status
  bash scripts/run_all_tmux_experiment.sh output1 --kill
  bash scripts/run_all_tmux_experiment.sh output2 8100

Defaults:
  name=output1
  port_base=8090

This writes to outputs/<name>/ and uses tmux sessions:
  ai4s-<name>-task1
  ai4s-<name>-task2
  ai4s-<name>-task3
  ai4s-<name>-assemble
EOF
}

name="output1"
port_base="${AI4S_PORT_BASE:-8090}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  name="$1"
  shift
fi

if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  port_base="$1"
  shift
fi

if [[ ! "$name" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: experiment name must use only letters, numbers, dot, underscore, or dash: $name" >&2
  exit 2
fi

if (( port_base < 1024 || port_base > 65533 )); then
  echo "error: port_base must be between 1024 and 65533: $port_base" >&2
  exit 2
fi

export AI4S_OUTPUTS_DIR="$ROOT_DIR/outputs/$name"
export AI4S_TMUX_PREFIX="ai4s-$name"
export AI4S_PROXY_PORT_T1="${AI4S_PROXY_PORT_T1:-$port_base}"
export AI4S_PROXY_PORT_T2="${AI4S_PROXY_PORT_T2:-$((port_base + 1))}"
export AI4S_PROXY_PORT_T3="${AI4S_PROXY_PORT_T3:-$((port_base + 2))}"

mkdir -p "$AI4S_OUTPUTS_DIR"

echo "experiment: $name"
echo "outputs:    $AI4S_OUTPUTS_DIR"
echo "sessions:   ${AI4S_TMUX_PREFIX}-task1 ${AI4S_TMUX_PREFIX}-task2 ${AI4S_TMUX_PREFIX}-task3 ${AI4S_TMUX_PREFIX}-assemble"
echo "ports:      $AI4S_PROXY_PORT_T1 $AI4S_PROXY_PORT_T2 $AI4S_PROXY_PORT_T3"
echo

exec bash scripts/run_all_tmux.sh "$@"
