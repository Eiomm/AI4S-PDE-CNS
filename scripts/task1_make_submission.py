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

from ai4sv2_task1.submission import make_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Task1 submission bundle from a run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--submission-name", required=True)
    parser.add_argument("--submission-id", default="AI4S-PDE-CNS")
    parser.add_argument("--code-dir", default=None, help="Agent 生成的代码目录；默认读取最新 summary.generated_code_root。")
    parser.add_argument("--llm-log", default=None, help="由 proxy JSONL 转换得到的合规 LLM 调用日志。")
    args = parser.parse_args()
    print(
        json.dumps(
            make_submission(
                args.run_dir,
                submission_name=args.submission_name,
                submission_id=args.submission_id,
                code_dir=args.code_dir,
                llm_log_path=args.llm_log,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
