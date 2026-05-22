#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai4sv2_task1.memory import promote_candidate


def main() -> None:
    """把某条 memory 记录写入候选板。

    注意：这里的 promote 只是“进入候选板”，不是“可以提交”。
    真正 submit_ready 仍需要重新 replay、校验 HDF5、确认前 10 帧一致、
    检查官方 LLM proxy log 和 submission 目录。
    """

    parser = argparse.ArgumentParser(description="Promote a Task1 memory record to candidate board.")
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--slot", required=True, help="例如 best_official_baseline / best_clean_finetune")
    parser.add_argument("--metric", required=True, help="例如 competition_score_proxy")
    parser.add_argument("--value", type=float, required=True)
    parser.add_argument("--blocker", action="append", default=[], help="候选仍需解决的问题，可重复传入。")
    args = parser.parse_args()
    print(promote_candidate(args.record_id, slot=args.slot, metric=args.metric, value=args.value, blockers=args.blocker))


if __name__ == "__main__":
    main()
