#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAISDATA_DIR="${SAISDATA_DIR:-/saisdata}"
SAISRESULT_DIR="${SAISRESULT_DIR:-/saisresult}"
PYTHON_BIN="${PY:-/data/wangjunao/miniconda3/envs/ai4s-chem-evolve/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
ENV_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export AI4S_ROUTE_ENGINE="${AI4S_ROUTE_ENGINE:-aizynthfinder}"
export AIZYNTHFINDER_CONFIG="${AIZYNTHFINDER_CONFIG:-${ROOT_DIR}/data/aizynthfinder/config.yml}"
export AI4S_ROUTE_LIMIT_PER_ROUND="${AI4S_ROUTE_LIMIT_PER_ROUND:-10}"
export AI4S_VINA_FEEDBACK_PER_ROUND="${AI4S_VINA_FEEDBACK_PER_ROUND:-1}"
export CHEM_EVOLVE_LLM_ENABLED="${CHEM_EVOLVE_LLM_ENABLED:-1}"
export AI4S_AGENT_BYPASS_PROXY="${AI4S_AGENT_BYPASS_PROXY:-1}"
export LITELLM_LOCAL_MODEL_COST_MAP="${LITELLM_LOCAL_MODEL_COST_MAP:-True}"
mkdir -p "${SAISRESULT_DIR}"
cd "${ROOT_DIR}"
TARGET_DIR="${SAISDATA_DIR}"
if [[ -f "${SAISDATA_DIR}/target1.pdb" && -f "${SAISDATA_DIR}/target2.pdb" && -f "${SAISDATA_DIR}/target3.pdb" ]]; then
  TARGET_DIR="${SAISDATA_DIR}"
elif [[ -f "${SAISDATA_DIR}/37/target1.pdb" && -f "${SAISDATA_DIR}/37/target2.pdb" && -f "${SAISDATA_DIR}/37/target3.pdb" ]]; then
  TARGET_DIR="${SAISDATA_DIR}/37"
fi
"$PYTHON_BIN" -m chem_evolve_agent.cli \
  --targets "${TARGET_DIR}/target1.pdb" "${TARGET_DIR}/target2.pdb" "${TARGET_DIR}/target3.pdb" \
  --out "${SAISRESULT_DIR}" \
  --rounds "${AGENT_ROUNDS:-8}" \
  --per-round "${AGENT_PER_ROUND:-32}" \
  --top-k "${AGENT_TOP_K:-10}" \
  --mode "${AGENT_MODE:-competition}" \
  --docking-limit "${AGENT_DOCKING_LIMIT:-${AGENT_TOP_K:-10}}" \
  --runner agent \
  --run-seed "${CHEM_EVOLVE_RUN_SEED:-0}"
"$PYTHON_BIN" scripts/inspect_result_zip.py "${SAISRESULT_DIR}/result.zip" result1.csv result2.csv result3.csv
