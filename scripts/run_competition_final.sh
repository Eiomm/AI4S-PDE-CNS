#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAISDATA_DIR="${SAISDATA_DIR:-/saisdata/37}"
SAISRESULT_DIR="${SAISRESULT_DIR:-/saisresult}"
PYTHON_BIN="${PY:-/data/wangjunao/miniconda3/envs/ai4s-chem-evolve/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
ENV_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${SAISRESULT_DIR}"
cd "${ROOT_DIR}"
"$PYTHON_BIN" -m chem_evolve_agent.cli \
  --targets "${SAISDATA_DIR}/target1.pdb" "${SAISDATA_DIR}/target2.pdb" "${SAISDATA_DIR}/target3.pdb" \
  --out "${SAISRESULT_DIR}" \
  --rounds "${AGENT_ROUNDS:-8}" \
  --per-round "${AGENT_PER_ROUND:-64}" \
  --top-k "${AGENT_TOP_K:-20}" \
  --mode "${AGENT_MODE:-competition}" \
  --docking-limit "${AGENT_DOCKING_LIMIT:-8}"
"$PYTHON_BIN" scripts/inspect_result_zip.py "${SAISRESULT_DIR}/result.zip" result1.csv result2.csv result3.csv
