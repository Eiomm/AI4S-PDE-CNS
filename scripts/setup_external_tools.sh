#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PY:-/data/wangjunao/miniconda3/envs/ai4s-chem-evolve/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

ENV_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"

DATA_DIR="${AIZYNTHFINDER_DATA_DIR:-$ROOT_DIR/data/aizynthfinder}"
CONFIG_PATH="$DATA_DIR/config.yml"

echo "[tools] Python：$PYTHON_BIN"

"$PYTHON_BIN" - <<'PY'
import importlib.util
missing = [name for name in ("rdkit", "vina", "openbabel", "aizynthfinder") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("缺少 Python 模块：" + ", ".join(missing))
print("[tools] Python 模块检查通过：rdkit, vina, openbabel, aizynthfinder")
PY

command -v obabel >/dev/null || {
  echo "错误：PATH 中找不到 obabel" >&2
  exit 2
}
command -v aizynthcli >/dev/null || {
  echo "错误：PATH 中找不到 aizynthcli" >&2
  exit 2
}
command -v download_public_data >/dev/null || {
  echo "错误：PATH 中找不到 download_public_data" >&2
  exit 2
}

if [[ ! -f "$CONFIG_PATH" ]]; then
  mkdir -p "$DATA_DIR"
  echo "[tools] 下载 AiZynthFinder 公共数据到 $DATA_DIR"
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    download_public_data "$DATA_DIR"
fi

[[ -f "$CONFIG_PATH" ]] || {
  echo "错误：下载后仍缺少 AiZynthFinder 配置：$CONFIG_PATH" >&2
  exit 2
}

if [[ "$DATA_DIR" == "$ROOT_DIR/data/aizynthfinder" ]]; then
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import yaml

config_path = Path("data/aizynthfinder/config.yml")
payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
payload["expansion"]["uspto"] = [
    "data/aizynthfinder/uspto_model.onnx",
    "data/aizynthfinder/uspto_templates.csv.gz",
]
payload["expansion"]["ringbreaker"] = [
    "data/aizynthfinder/uspto_ringbreaker_model.onnx",
    "data/aizynthfinder/uspto_ringbreaker_templates.csv.gz",
]
payload["filter"]["uspto"] = "data/aizynthfinder/uspto_filter_model.onnx"
payload["stock"]["zinc"] = "data/aizynthfinder/zinc_stock.hdf5"
config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
print("[tools] 已把 AiZynthFinder 配置规范为项目相对路径")
PY
fi

echo "[tools] AIZYNTHFINDER_CONFIG=$CONFIG_PATH"
echo "[tools] 完成"
