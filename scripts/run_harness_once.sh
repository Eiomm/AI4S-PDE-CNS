#!/usr/bin/env bash
# Run the smallest engineering loop for the AI4S chemistry pipeline:
# tests -> target generation -> submission inspection -> lightweight memory note.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

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

load_dotenv_preserve_env ".env"

OUTPUTS_DIR="${AI4S_OUTPUTS_DIR:-$ROOT_DIR/outputs}"
RUN_NAME="${AI4S_RUN_NAME:-target_minimal}"
TARGET="${AI4S_TARGET:-$ROOT_DIR/target.pdb}"
ROUNDS="${AGENT_ROUNDS:-4}"
PER_ROUND="${AGENT_PER_ROUND:-12}"
TOP_K="${AGENT_TOP_K:-20}"
MODE="${AGENT_MODE:-docking}"
DOCKING_LIMIT="${AGENT_DOCKING_LIMIT:-6}"
RUNNER="${AGENT_RUNNER:-agent}"
SKIP_TESTS="${AI4S_SKIP_TESTS:-0}"
CONDA_ENV="${AI4S_CONDA_ENV:-ai4s-chem-evolve}"
RUN_SEED="${CHEM_EVOLVE_RUN_SEED:-0}"

usage() {
  cat <<'EOF'
用法：
  bash scripts/run_harness_once.sh [options]

参数：
  --name NAME             输出实验名。默认：target_minimal
  --target PATH           PDB 靶点路径。默认：target.pdb
  --rounds N              生成轮数。默认：4
  --per-round N           每轮请求的候选数。默认：12
  --top-k N               result csv 保留行数。默认：20
  --mode MODE             proxy、docking 或 competition。默认：docking
  --docking-limit N       docking 模式或竞赛终局复排的 Vina 预算。默认：6
  --runner RUNNER         agent。默认：agent
  --skip-tests            本次 harness 跳过 pytest。
  -h, --help              显示帮助。

环境变量：
  PY=/path/to/python                  指定 Python 可执行文件。
  AI4S_CONDA_ENV=ai4s-chem-evolve     未设置 PY 时使用的 Conda 环境。
  AI4S_OUTPUTS_DIR=outputs            带时间戳 run 的根目录。
  CHEM_EVOLVE_LLM_ENABLED=1           启用 LiteLLM 推理生成器。默认：1。
  CHEM_EVOLVE_RUN_SEED=0              重复实验的随机种子偏移。

输出：
  outputs/<name>/<timestamp>/result.csv       竞赛兼容 csv。
  outputs/<name>/<timestamp>/result.log       竞赛兼容事件日志。
  outputs/<name>/<timestamp>/result.zip       竞赛兼容 zip。
  outputs/<name>/<timestamp>/candidates.csv   方便阅读的别名。
  outputs/<name>/<timestamp>/pipeline.log     方便阅读的别名。
  outputs/<name>/<timestamp>/submission.zip   方便阅读的别名。
  outputs/<name>/latest                       指向最新 run 的软链接。
EOF
}

die() {
  echo "错误：$*" >&2
  exit 2
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
  if [[ -d "$env_prefix/bin" ]]; then
    export PATH="$env_prefix/bin:$PATH"
  fi
  if [[ -d "$env_prefix/lib" ]]; then
    export LD_LIBRARY_PATH="$env_prefix/lib:${LD_LIBRARY_PATH:-}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      RUN_NAME="${2:-}"
      shift 2
      ;;
    --target)
      TARGET="${2:-}"
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
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --docking-limit)
      DOCKING_LIMIT="${2:-}"
      shift 2
      ;;
    --runner)
      RUNNER="${2:-}"
      shift 2
      ;;
    --skip-tests)
      SKIP_TESTS=1
      shift
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

[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "name must use only letters, numbers, dot, underscore, or dash: $RUN_NAME"
[[ -f "$TARGET" ]] || die "target PDB not found: $TARGET"
[[ "$MODE" =~ ^(proxy|docking|competition)$ ]] || die "mode must be proxy, docking, or competition: $MODE"
[[ "$RUNNER" == "agent" ]] || die "legacy runner has been removed; runner must be agent"

PYTHON_BIN="$(resolve_python)"
configure_python_runtime "$PYTHON_BIN"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_PARENT="$OUTPUTS_DIR/$RUN_NAME"
RUN_ROOT="$RUN_PARENT/$STAMP"
LLM_LOG_DIR="$RUN_ROOT/llm_io"
MEMORY_FILE="$OUTPUTS_DIR/engineering_memory.md"

mkdir -p "$RUN_ROOT" "$LLM_LOG_DIR"
ln -sfn "$RUN_ROOT" "$RUN_PARENT/latest"

export AI4S_AGENT_LLM_LOG_DIR="$LLM_LOG_DIR"
export CHEM_EVOLVE_LLM_LOG_DIR="$LLM_LOG_DIR"
export PYTHONUNBUFFERED=1

write_aliases() {
  [[ -f "$RUN_ROOT/result.csv" ]] && cp -f "$RUN_ROOT/result.csv" "$RUN_ROOT/candidates.csv"
  [[ -f "$RUN_ROOT/result.log" ]] && cp -f "$RUN_ROOT/result.log" "$RUN_ROOT/pipeline.log"
  [[ -f "$RUN_ROOT/result.zip" ]] && cp -f "$RUN_ROOT/result.zip" "$RUN_ROOT/submission.zip"
}

submission_artifacts_complete() {
  [[ -s "$RUN_ROOT/result.csv" ]] &&
    [[ -s "$RUN_ROOT/result.log" ]] &&
    [[ -s "$RUN_ROOT/result.zip" ]] &&
    [[ -s "$RUN_ROOT/submission.zip" ]] &&
    [[ -s "$RUN_ROOT/inspect.log" ]] &&
    grep -q '^OK ' "$RUN_ROOT/inspect.log"
}

finish() {
  local rc=$?
  local finished_at
  local status
  local artifact_status
  local effective_rc
  finished_at="$(date -Is)"
  effective_rc="$rc"
  status="fail"
  artifact_status="incomplete"
  write_aliases || true
  if submission_artifacts_complete; then
    artifact_status="complete"
  fi
  if [[ "$rc" == "0" && "$artifact_status" == "complete" ]]; then
    status="pass"
  elif [[ "$rc" == "0" ]]; then
    status="incomplete"
    effective_rc=3
  fi
  {
    echo "name=$RUN_NAME"
    echo "exit_code=$effective_rc"
    echo "original_exit_code=$rc"
    echo "artifact_status=$artifact_status"
    echo "target=$TARGET"
    echo "run_root=$RUN_ROOT"
    echo "python=$PYTHON_BIN"
    echo "rounds=$ROUNDS"
    echo "per_round=$PER_ROUND"
    echo "top_k=$TOP_K"
    echo "mode=$MODE"
    echo "docking_limit=$DOCKING_LIMIT"
    echo "runner=$RUNNER"
    echo "run_seed=$RUN_SEED"
    echo "finished_at=$finished_at"
  } > "$RUN_ROOT/harness.done"
  {
    echo
    echo "## $finished_at - $RUN_NAME"
    echo "- status: $status"
    echo "- artifact_status: $artifact_status"
    echo "- target: $TARGET"
    echo "- run: $RUN_ROOT"
    echo "- settings: rounds=$ROUNDS, per_round=$PER_ROUND, top_k=$TOP_K, mode=$MODE, docking_limit=$DOCKING_LIMIT, runner=$RUNNER, run_seed=$RUN_SEED"
    if [[ "$status" == "incomplete" ]]; then
      echo "- warning: 进程退出码为 0，但最终结果文件或 inspect OK 缺失"
    fi
    if [[ -f "$RUN_ROOT/submission.zip" ]]; then
      echo "- submission: $RUN_ROOT/submission.zip"
    fi
    if [[ -f "$RUN_ROOT/inspect.log" ]]; then
      echo "- inspect: $(tail -n 1 "$RUN_ROOT/inspect.log")"
    fi
  } >> "$MEMORY_FILE"
  echo
  echo "[harness] 退出码：$effective_rc"
  echo "[harness] 本次输出：$RUN_ROOT"
  echo "[harness] 最新链接：$RUN_PARENT/latest"
  [[ -f "$RUN_ROOT/submission.zip" ]] && echo "[harness] 提交包：$RUN_ROOT/submission.zip"
  exit "$effective_rc"
}
trap finish EXIT

cat > "$RUN_ROOT/run.env" <<EOF
ROOT_DIR=$ROOT_DIR
PYTHON=$PYTHON_BIN
TARGET=$TARGET
RUN_NAME=$RUN_NAME
RUN_ROOT=$RUN_ROOT
ROUNDS=$ROUNDS
PER_ROUND=$PER_ROUND
TOP_K=$TOP_K
MODE=$MODE
DOCKING_LIMIT=$DOCKING_LIMIT
RUNNER=$RUNNER
RUN_SEED=$RUN_SEED
SKIP_TESTS=$SKIP_TESTS
EOF

echo "[harness] 项目根目录：$ROOT_DIR"
echo "[harness] Python：    $PYTHON_BIN"
echo "[harness] 靶点：      $TARGET"
echo "[harness] 输出目录：  $RUN_ROOT"
echo "[harness] 模式：      $MODE"
echo "[harness] runner：    $RUNNER"
echo "[harness] 随机种子：  $RUN_SEED"
echo

if [[ "$SKIP_TESTS" != "1" ]]; then
  echo "[harness] 步骤 1/3：运行 pytest"
  "$PYTHON_BIN" -m pytest -v 2>&1 | tee "$RUN_ROOT/test.log"
else
  echo "[harness] 步骤 1/3：已跳过 pytest"
  printf 'pytest 已通过 --skip-tests 跳过\n' > "$RUN_ROOT/test.log"
fi

echo
echo "[harness] 步骤 2/3：生成候选分子"
"$PYTHON_BIN" -m chem_evolve_agent.cli \
  --targets "$TARGET" \
  --out "$RUN_ROOT" \
  --rounds "$ROUNDS" \
  --per-round "$PER_ROUND" \
  --top-k "$TOP_K" \
  --mode "$MODE" \
  --docking-limit "$DOCKING_LIMIT" \
  --runner "$RUNNER" \
  --run-seed "$RUN_SEED" \
  2>&1 | tee "$RUN_ROOT/pipeline.stdout.log"

write_aliases

echo
echo "[harness] 步骤 3/3：检查提交包"
"$PYTHON_BIN" scripts/inspect_result_zip.py "$RUN_ROOT/result.zip" 2>&1 | tee "$RUN_ROOT/inspect.log"

echo
echo "[harness] 产物："
echo "  候选分子：$RUN_ROOT/candidates.csv"
echo "  流水线日志：$RUN_ROOT/pipeline.log"
echo "  提交包：$RUN_ROOT/submission.zip"
