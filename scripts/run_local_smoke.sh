#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PY:-/data/wangjunao/miniconda3/envs/ai4s-chem-evolve/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
ENV_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CHEM_EVOLVE_LLM_ENABLED="${CHEM_EVOLVE_LLM_ENABLED:-1}"
mkdir -p examples runs/smoke
if [ ! -f examples/target.pdb ]; then
  echo "错误：需要 examples/target.pdb" >&2
  exit 2
fi
"$PYTHON_BIN" -m chem_evolve_agent.cli --targets examples/target.pdb --out runs/smoke --rounds 1 --per-round 8 --top-k 5 --mode proxy --runner agent
"$PYTHON_BIN" scripts/inspect_result_zip.py runs/smoke/result.zip
