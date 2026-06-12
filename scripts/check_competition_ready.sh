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

"$PYTHON_BIN" scripts/check_data.py

if [[ "${AI4S_REQUIRE_SBDD:-0}" == "1" ]]; then
  "$PYTHON_BIN" scripts/check_tools.py --require-sbdd
else
  "$PYTHON_BIN" scripts/check_tools.py
fi

if [[ "${AI4S_SKIP_LLM_CHECK:-0}" != "1" ]]; then
  "$PYTHON_BIN" scripts/check_llm_connectivity.py
fi

SMOKE_OUT="${AI4S_READY_SMOKE_OUT:-/tmp/ai4s_competition_ready_smoke}"
rm -rf "$SMOKE_OUT"
CHEM_EVOLVE_LLM_ENABLED=0 \
AI4S_ROUTE_ENGINE=aizynthfinder \
AIZYNTHFINDER_CONFIG="${AIZYNTHFINDER_CONFIG:-data/aizynthfinder/config.yml}" \
AI4S_ROUTE_LIMIT_PER_ROUND="${AI4S_ROUTE_LIMIT_PER_ROUND:-8}" \
"$PYTHON_BIN" -m chem_evolve_agent.cli \
  --targets examples/target.pdb \
  --out "$SMOKE_OUT" \
  --rounds 1 \
  --per-round 8 \
  --top-k 5 \
  --mode proxy \
  --runner agent
"$PYTHON_BIN" scripts/inspect_result_zip.py "$SMOKE_OUT/result.zip"

if [[ "${AI4S_RUN_REAL_COMPETITION_SMOKE:-0}" == "1" ]]; then
  bash scripts/run_real_competition_smoke.sh
fi

echo "COMPETITION_READY_CHECK_OK 竞赛就绪检查通过"
