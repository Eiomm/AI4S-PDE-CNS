#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PY:-/data/wangjunao/miniconda3/envs/ai4s-chem-evolve/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

ENV_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"

TARGET="${AI4S_REAL_SMOKE_TARGET:-examples/target.pdb}"
OUT_DIR="${AI4S_REAL_SMOKE_OUT:-/tmp/ai4s_real_competition_smoke}"

export CHEM_EVOLVE_LLM_ENABLED="${CHEM_EVOLVE_LLM_ENABLED:-1}"
export AI4S_ROUTE_ENGINE="${AI4S_ROUTE_ENGINE:-aizynthfinder}"
export AIZYNTHFINDER_CONFIG="${AIZYNTHFINDER_CONFIG:-${ROOT_DIR}/data/aizynthfinder/config.yml}"
export AI4S_ROUTE_LIMIT_PER_ROUND="${AI4S_ROUTE_LIMIT_PER_ROUND:-2}"
export AI4S_VINA_FEEDBACK_PER_ROUND="${AI4S_VINA_FEEDBACK_PER_ROUND:-1}"

if [[ ! -f "$TARGET" ]]; then
  echo "错误：找不到靶点 PDB：$TARGET" >&2
  exit 2
fi

"$PYTHON_BIN" -m chem_evolve_agent.cli \
  --targets "$TARGET" \
  --out "$OUT_DIR" \
  --rounds "${AGENT_ROUNDS:-1}" \
  --per-round "${AGENT_PER_ROUND:-4}" \
  --top-k "${AGENT_TOP_K:-1}" \
  --mode competition \
  --docking-limit "${AGENT_DOCKING_LIMIT:-1}" \
  --runner agent \
  --run-seed "${CHEM_EVOLVE_RUN_SEED:-0}"

"$PYTHON_BIN" scripts/inspect_result_zip.py "$OUT_DIR/result.zip"

if ! grep -q '"event": "competition_dock"' "$OUT_DIR/result.log"; then
  echo "错误：competition smoke 没有记录 competition_dock" >&2
  exit 2
fi

if ! grep -q '"route_source": "aizynthfinder"' "$OUT_DIR/result.log"; then
  echo "错误：competition smoke 没有使用 AiZynthFinder 路线" >&2
  exit 2
fi

echo "REAL_COMPETITION_SMOKE_OK 真实竞赛 smoke 通过：$OUT_DIR/result.zip"
