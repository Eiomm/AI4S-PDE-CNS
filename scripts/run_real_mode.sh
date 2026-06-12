#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${AGENT_MODE:-competition}"
TARGET="${AI4S_TARGET:-examples/target.pdb}"
OUTPUTS_DIR="${AI4S_OUTPUTS_DIR:-$ROOT_DIR/outputs}"
RUN_NAME="${AI4S_RUN_NAME:-}"
OUT_DIR="${AI4S_OUT_DIR:-}"
ROUNDS="${AGENT_ROUNDS:-10}"
PER_ROUND="${AGENT_PER_ROUND:-15}"
TOP_K="${AGENT_TOP_K:-5}"
DOCKING_LIMIT="${AGENT_DOCKING_LIMIT:-5}"
CONDA_ENV="${AI4S_CONDA_ENV:-ai4s-chem-evolve}"
RUN_SEED="${CHEM_EVOLVE_RUN_SEED:-0}"

usage() {
  cat <<'EOF'
用法：
  bash scripts/run_real_mode.sh --mode competition --target examples/target.pdb
  bash scripts/run_real_mode.sh --mode docking --target examples/target.pdb

参数：
  --mode competition|docking  要运行的真实评分模式。默认：competition
  --target PATH               靶点 PDB。默认：examples/target.pdb
  --out DIR                   输出目录。默认：outputs/<mode>_real/<timestamp>
  --name NAME                 输出 run 分组名。默认：<mode>_real
  --rounds N                  agent 轮数。默认：1
  --per-round N               每轮候选池大小。默认：8
  --top-k N                   排名后 CSV 保留行数。默认：3
  --docking-limit N           Vina 预算。默认：3
  --run-seed N                随机种子偏移。默认：0
  -h, --help                  显示帮助。

环境变量：
  PY=/path/to/python
  CHEM_EVOLVE_LLM_ENABLED=1
  AI4S_AGENT_API_KEY_ENVS=APIFOX_GPT_GE_API_KEY,OPENAI_API_KEY
  AI4S_AGENT_BASE_URL=https://...
  AI4S_AGENT_MODEL=openai/...
  AIZYNTHFINDER_CONFIG=data/aizynthfinder/config.yml
  AI4S_ROUTE_LIMIT_PER_ROUND=3
  AI4S_VINA_FEEDBACK_PER_ROUND=1

说明：
  这个脚本不会自动退回 proxy。路线规划必须使用 AiZynthFinder。
  competition 模式探索阶段使用 proxy_search，结束时用真实 Vina 复排。
  docking 模式在候选评估阶段直接使用真实 Vina。
EOF
}

die() {
  echo "错误：$*" >&2
  exit 2
}

load_dotenv_preserve_env() {
  local env_file="$1"
  [[ -f "$env_file" ]] || return 0
  local line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "${line:0:1}" == "#" || "$line" != *"="* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ -z "${!key+x}" ]]; then
      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%"${value##*[![:space:]]}"}"
      value="${value%\"}"
      value="${value#\"}"
      value="${value%\'}"
      value="${value#\'}"
      export "$key=$value"
    fi
  done < "$env_file"
}

resolve_python() {
  if [[ -n "${PY:-}" ]]; then
    [[ -x "$PY" ]] || die "PY is not executable: $PY"
    printf '%s\n' "$PY"
    return 0
  fi

  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "$conda_base" && -x "$conda_base/envs/$CONDA_ENV/bin/python" ]]; then
      printf '%s\n' "$conda_base/envs/$CONDA_ENV/bin/python"
      return 0
    fi
  fi

  for candidate in \
    "/data/wangjunao/miniconda3/envs/$CONDA_ENV/bin/python" \
    "$HOME/miniconda3/envs/$CONDA_ENV/bin/python" \
    "$ROOT_DIR/.venv/bin/python"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  command -v python3 >/dev/null 2>&1 || die "没有找到 Python；请设置 PY=/path/to/python"
  command -v python3
}

configure_python_runtime() {
  local python_bin="$1"
  local env_prefix
  env_prefix="$(cd "$(dirname "$python_bin")/.." && pwd)"
  export PATH="$env_prefix/bin:$PATH"
  export LD_LIBRARY_PATH="$env_prefix/lib:${LD_LIBRARY_PATH:-}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --target)
      TARGET="${2:-}"
      shift 2
      ;;
    --out)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --name)
      RUN_NAME="${2:-}"
      shift 2
      ;;
    --rounds)
      ROUNDS="${2:-}"
      shift 2
      ;;
    --per-round)
      PER_ROUND="${2:-}"
      shift 2
      ;;
    --top-k)
      TOP_K="${2:-}"
      shift 2
      ;;
    --docking-limit)
      DOCKING_LIMIT="${2:-}"
      shift 2
      ;;
    --run-seed)
      RUN_SEED="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

load_dotenv_preserve_env ".env"

[[ "$MODE" =~ ^(competition|docking)$ ]] || die "--mode must be competition or docking: $MODE"
[[ -f "$TARGET" ]] || die "target PDB not found: $TARGET"
[[ "$ROUNDS" =~ ^[0-9]+$ && "$ROUNDS" -gt 0 ]] || die "--rounds must be a positive integer"
[[ "$PER_ROUND" =~ ^[0-9]+$ && "$PER_ROUND" -gt 0 ]] || die "--per-round must be a positive integer"
[[ "$TOP_K" =~ ^[0-9]+$ && "$TOP_K" -gt 0 ]] || die "--top-k must be a positive integer"
[[ "$DOCKING_LIMIT" =~ ^[0-9]+$ && "$DOCKING_LIMIT" -gt 0 ]] || die "--docking-limit must be a positive integer"

PYTHON_BIN="$(resolve_python)"
configure_python_runtime "$PYTHON_BIN"

export PYTHONUNBUFFERED=1
export AI4S_ROUTE_ENGINE="aizynthfinder"
export AIZYNTHFINDER_CONFIG="${AIZYNTHFINDER_CONFIG:-data/aizynthfinder/config.yml}"
export CHEM_EVOLVE_LLM_ENABLED="${CHEM_EVOLVE_LLM_ENABLED:-1}"
export AI4S_ROUTE_LIMIT_PER_ROUND="${AI4S_ROUTE_LIMIT_PER_ROUND:-$DOCKING_LIMIT}"
if [[ "$MODE" == "competition" ]]; then
  export AI4S_VINA_FEEDBACK_PER_ROUND="${AI4S_VINA_FEEDBACK_PER_ROUND:-1}"
else
  export AI4S_VINA_FEEDBACK_PER_ROUND="${AI4S_VINA_FEEDBACK_PER_ROUND:-0}"
fi

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="${MODE}_real"
fi
if [[ -z "$OUT_DIR" ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  OUT_DIR="$OUTPUTS_DIR/$RUN_NAME/$STAMP"
  mkdir -p "$OUTPUTS_DIR/$RUN_NAME"
  ln -sfn "$OUT_DIR" "$OUTPUTS_DIR/$RUN_NAME/latest"
fi

mkdir -p "$OUT_DIR"
export AI4S_AGENT_LLM_LOG_DIR="$OUT_DIR/llm_io"
export CHEM_EVOLVE_LLM_LOG_DIR="$OUT_DIR/llm_io"

echo "[run] 项目根目录：        $ROOT_DIR"
echo "[run] Python：            $PYTHON_BIN"
echo "[run] 模式：              $MODE"
echo "[run] 靶点：              $TARGET"
echo "[run] 输出目录：          $OUT_DIR"
echo "[run] 轮数：              $ROUNDS"
echo "[run] 每轮候选数：        $PER_ROUND"
echo "[run] 保留 top_k：        $TOP_K"
echo "[run] docking 预算：      $DOCKING_LIMIT"
echo "[run] LLM 是否启用：      $CHEM_EVOLVE_LLM_ENABLED"
echo "[run] 路线引擎：          $AI4S_ROUTE_ENGINE"
echo "[run] 每轮路线预算：      $AI4S_ROUTE_LIMIT_PER_ROUND"
echo "[run] Vina 反馈预算：     $AI4S_VINA_FEEDBACK_PER_ROUND"

"$PYTHON_BIN" -m chem_evolve_agent.cli \
  --targets "$TARGET" \
  --out "$OUT_DIR" \
  --rounds "$ROUNDS" \
  --per-round "$PER_ROUND" \
  --top-k "$TOP_K" \
  --mode "$MODE" \
  --docking-limit "$DOCKING_LIMIT" \
  --runner agent \
  --run-seed "$RUN_SEED" \
  2>&1 | tee "$OUT_DIR/stdout.log"

"$PYTHON_BIN" scripts/inspect_result_zip.py "$OUT_DIR/result.zip" | tee "$OUT_DIR/inspect.log"

grep -q '"route_source": "aizynthfinder"' "$OUT_DIR/result.log" || die "result.log did not record AiZynthFinder routes"
if [[ "$MODE" == "competition" ]]; then
  grep -q '"event": "competition_dock"' "$OUT_DIR/result.log" || die "competition mode did not record final Vina reranking"
else
  grep -q '"binding_source": "vina"' "$OUT_DIR/result.log" || die "docking mode did not record Vina binding scores"
fi

echo "[run] OK：提交包已生成 $OUT_DIR/result.zip"
