#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai4sv2_task1.config import load_yaml
from ai4sv2_task1.predict import run_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Task1 predictions from official checkpoints.")
    parser.add_argument("--config", default="configs/official_fno.yaml")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--run-name", default=None, help="只作为 metadata 备注；实际 run 目录统一使用 UTC timestamp。")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for prefix validation.")
    args = parser.parse_args()
    result = run_prediction(load_yaml(args.config), split=args.split, run_name=args.run_name, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
