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
RUNNER="${AGENT_RUNNER:-legacy}"
SKIP_TESTS="${AI4S_SKIP_TESTS:-0}"
CONDA_ENV="${AI4S_CONDA_ENV:-ai4s-chem-evolve}"
RUN_SEED="${CHEM_EVOLVE_RUN_SEED:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_harness_once.sh [options]

Options:
  --name NAME             Output experiment name. Default: target_minimal
  --target PATH           PDB target path. Default: target.pdb
  --rounds N              Generation rounds. Default: 4
  --per-round N           Candidates requested per round. Default: 12
  --top-k N               Rows kept in result csv. Default: 20
  --mode MODE             proxy, docking, or competition. Default: docking
  --docking-limit N       Number of top candidates to try in docking mode. Default: 6
  --runner RUNNER         legacy or agent. Default: legacy
  --skip-tests            Skip pytest in this harness run.
  -h, --help              Show this help.

Environment:
  PY=/path/to/python                  Python executable override.
  AI4S_CONDA_ENV=ai4s-chem-evolve     Conda env used when PY is unset.
  AI4S_OUTPUTS_DIR=outputs            Root directory for timestamped runs.
  CHEM_EVOLVE_LLM_ENABLED=1           Enable optional LiteLLM generator.
  CHEM_EVOLVE_RUN_SEED=0              Seed offset used to vary repeated experiment runs.

Outputs:
  outputs/<name>/<timestamp>/result.csv       Competition-compatible csv.
  outputs/<name>/<timestamp>/result.log       Competition-compatible event log.
  outputs/<name>/<timestamp>/result.zip       Competition-compatible zip.
  outputs/<name>/<timestamp>/candidates.csv   Human-friendly alias.
  outputs/<name>/<timestamp>/pipeline.log     Human-friendly alias.
  outputs/<name>/<timestamp>/submission.zip   Human-friendly alias.
  outputs/<name>/latest                       Symlink to the newest run.
EOF
}

die() {
  echo "error: $*" >&2
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

  command -v python3 >/dev/null 2>&1 || die "no Python found; set PY=/path/to/python"
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
[[ "$RUNNER" =~ ^(legacy|agent)$ ]] || die "runner must be legacy or agent: $RUNNER"

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

finish() {
  local rc=$?
  local finished_at
  local status
  finished_at="$(date -Is)"
  status="fail"
  if [[ "$rc" == "0" ]]; then
    status="pass"
  fi
  write_aliases || true
  {
    echo "name=$RUN_NAME"
    echo "exit_code=$rc"
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
    echo "- target: $TARGET"
    echo "- run: $RUN_ROOT"
    echo "- settings: rounds=$ROUNDS, per_round=$PER_ROUND, top_k=$TOP_K, mode=$MODE, docking_limit=$DOCKING_LIMIT, runner=$RUNNER, run_seed=$RUN_SEED"
    if [[ -f "$RUN_ROOT/submission.zip" ]]; then
      echo "- submission: $RUN_ROOT/submission.zip"
    fi
    if [[ -f "$RUN_ROOT/inspect.log" ]]; then
      echo "- inspect: $(tail -n 1 "$RUN_ROOT/inspect.log")"
    fi
  } >> "$MEMORY_FILE"
  echo
  echo "[harness] exit=$rc"
  echo "[harness] run root: $RUN_ROOT"
  echo "[harness] latest:   $RUN_PARENT/latest"
  [[ -f "$RUN_ROOT/submission.zip" ]] && echo "[harness] result:   $RUN_ROOT/submission.zip"
  exit "$rc"
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

echo "[harness] root:    $ROOT_DIR"
echo "[harness] python:  $PYTHON_BIN"
echo "[harness] target:  $TARGET"
echo "[harness] output:  $RUN_ROOT"
echo "[harness] mode:    $MODE"
echo "[harness] runner:  $RUNNER"
echo "[harness] seed:    $RUN_SEED"
echo

if [[ "$SKIP_TESTS" != "1" ]]; then
  echo "[harness] step 1/3: pytest"
  "$PYTHON_BIN" -m pytest -v 2>&1 | tee "$RUN_ROOT/test.log"
else
  echo "[harness] step 1/3: pytest skipped"
  printf 'pytest skipped by --skip-tests\n' > "$RUN_ROOT/test.log"
fi

echo
echo "[harness] step 2/3: generate candidates"
"$PYTHON_BIN" -m chem_evolve_agent.cli \
  --targets "$TARGET" \
  --out "$RUN_ROOT" \
  --rounds "$ROUNDS" \
  --per-round "$PER_ROUND" \
  --top-k "$TOP_K" \
  --mode "$MODE" \
  --docking-limit "$DOCKING_LIMIT" \
  --runner "$RUNNER" \
  2>&1 | tee "$RUN_ROOT/pipeline.stdout.log"

write_aliases

echo
echo "[harness] step 3/3: inspect submission"
"$PYTHON_BIN" scripts/inspect_result_zip.py "$RUN_ROOT/result.zip" 2>&1 | tee "$RUN_ROOT/inspect.log"

echo
echo "[harness] artifacts:"
echo "  candidates: $RUN_ROOT/candidates.csv"
echo "  pipeline:   $RUN_ROOT/pipeline.log"
echo "  submission: $RUN_ROOT/submission.zip"
