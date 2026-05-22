#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai4sv2_task1.memory import add_record_to_registry, export_run_memory


def main() -> None:
    """把一次 run 的结果压缩成长期 memory 记录。

    用法分两步：
    1. 从 `runs/task1/<UTC timestamp>/metadata.json` 生成同目录的
       `memory_export.json`。
    2. 如果带 `--append-registry`，再把这条摘要追加到
       `memory/episodic/runs.jsonl`。

    这样设计是为了让 Agent 或人类可以先检查 `memory_export.json`，
    再决定是否写入长期记忆，避免坏记录污染后续检索。

    日常实验优先使用 `scripts/task1_run_workflow.py`，它会在预测结束后自动
    生成并追加 compact memory；本脚本主要保留给补录历史 run 使用。
    """

    parser = argparse.ArgumentParser(description="Export compact Task1 run memory.")
    parser.add_argument("--run-dir", required=True, help="例如 runs/task1/official_fno_val")
    parser.add_argument("--hypothesis", required=True, help="本次实验要验证的假设，一句话即可。")
    parser.add_argument("--decision", required=True, help="baseline / reject / keep / promote_candidate / do_not_repeat")
    parser.add_argument("--tag", action="append", default=[], help="可重复传入，例如 --tag official --tag fno")
    parser.add_argument("--append-registry", action="store_true", help="确认无误后追加到长期 registry.jsonl。")
    args = parser.parse_args()

    export_path = export_run_memory(args.run_dir, hypothesis=args.hypothesis, decision=args.decision, tags=args.tag)
    print(export_path)
    if args.append_registry:
        print(add_record_to_registry(export_path))


if __name__ == "__main__":
    main()
