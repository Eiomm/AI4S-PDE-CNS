#!/usr/bin/env bash
set -euo pipefail

cd /workspace
if [[ -f /workspace/configs/docker_llm.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /workspace/configs/docker_llm.env
  set +a
fi
exec conda run --no-capture-output -n ai4s-chem-evolve python Code/main.py
