#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Semifinal profile. It keeps one clean path:
# LLM/RDKit exploration -> AiZynthFinder route planning -> per-round Vina feedback
# -> final Vina reranking -> result.zip.
export AGENT_MODE="${AGENT_MODE:-competition}"
export CHEM_EVOLVE_LLM_ENABLED="${CHEM_EVOLVE_LLM_ENABLED:-1}"
export AGENT_ROUNDS="${AGENT_ROUNDS:-8}"
export AGENT_PER_ROUND="${AGENT_PER_ROUND:-32}"
export AGENT_TOP_K="${AGENT_TOP_K:-10}"
export AGENT_DOCKING_LIMIT="${AGENT_DOCKING_LIMIT:-10}"
export AI4S_ROUTE_ENGINE="aizynthfinder"
export AIZYNTHFINDER_CONFIG="${AIZYNTHFINDER_CONFIG:-${ROOT_DIR}/data/aizynthfinder/config.yml}"
export AI4S_ROUTE_LIMIT_PER_ROUND="${AI4S_ROUTE_LIMIT_PER_ROUND:-10}"
export AI4S_VINA_FEEDBACK_PER_ROUND="${AI4S_VINA_FEEDBACK_PER_ROUND:-1}"
export AI4S_AGENT_MEMORY_LIMIT="${AI4S_AGENT_MEMORY_LIMIT:-20}"
export AIZYNTHFINDER_TIMEOUT="${AIZYNTHFINDER_TIMEOUT:-900}"

echo "[4h] 模式：          $AGENT_MODE"
echo "[4h] 轮数：          $AGENT_ROUNDS"
echo "[4h] 每轮候选数：    $AGENT_PER_ROUND"
echo "[4h] 路线预算：      $AI4S_ROUTE_LIMIT_PER_ROUND"
echo "[4h] Vina 反馈预算： $AI4S_VINA_FEEDBACK_PER_ROUND"
echo "[4h] docking 预算：  $AGENT_DOCKING_LIMIT"
echo "[4h] 保留 top_k：    $AGENT_TOP_K"
echo "[4h] LLM 是否启用：  $CHEM_EVOLVE_LLM_ENABLED"

exec bash scripts/run_competition_final.sh
