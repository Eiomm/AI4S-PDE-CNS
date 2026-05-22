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

from ai4sv2_task1.memory import query_memory


def main() -> None:
    """检索长期 memory，并输出给 LLM 使用的小型 retrieval packet。

    当前为了简单可行，只读取 4 个 memory 入口：
    - `memory/contract/task1_rules.yaml`
    - `memory/episodic/runs.jsonl`
    - `memory/findings/metric_leaderboard.csv`
    - `memory/wisdom/strategy_summary.md`

    后续 Agent runner 应把这个 JSON 放进 prompt，而不是把整个
    `memory/` 或 `runs/` 目录塞给模型。
    """

    parser = argparse.ArgumentParser(description="Query compact Task1 memory.")
    parser.add_argument("--route", default=None, help="例如 official_fno / official_ensemble / finetune_fno")
    parser.add_argument("--tag", action="append", default=[], help="可重复传入。")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=6000)
    args = parser.parse_args()
    packet = query_memory(route=args.route, tags=args.tag, limit=args.limit, max_chars=args.max_chars)
    print(json.dumps(packet, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
