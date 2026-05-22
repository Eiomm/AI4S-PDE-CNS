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
from ai4sv2_task1.memory import add_record_to_registry, export_run_memory, promote_candidate
from ai4sv2_task1.predict import run_prediction


def infer_default_decision(split: str, metrics: dict | None) -> str:
    """根据是否有验证指标给出默认 memory 决策。

    - validation run 有 metrics，默认 `keep`，表示这条记录可以供下轮检索；
    - test run 没有公开 target，默认 `baseline`，表示它主要用于产物生成。
    """

    if split == "val" and metrics:
        return "keep"
    return "baseline"


def main() -> None:
    """执行 Task1 的最小实验闭环。

    这个脚本把过去需要手动串起来的步骤合并为一条命令：

    1. 读取 YAML 配置；
    2. 用 neural checkpoint 生成预测；
    3. 自动完成 shape / finite / 前 10 帧校验；
    4. 如果是 validation run，自动写出 metrics；
    5. 自动生成 `memory_export.json`；
    6. 默认追加到 compact long-term memory；
    7. 可选把结果写入候选 leaderboard。

    它仍然是 harness 工作流，不代表最终 submission `code/` 来源。最终
    `code/` 仍应由 `task1_agent_runner.py` 通过 GPT-5.5 proxy 生成。
    """

    parser = argparse.ArgumentParser(description="Run Task1 prediction and automatically export compact memory.")
    parser.add_argument("--config", default="configs/official_fno.yaml")
    parser.add_argument("--split", choices=["test", "val"], default="val")
    parser.add_argument("--run-name", default=None, help="只作为 metadata 备注；实际 run 目录统一使用 UTC timestamp。")
    parser.add_argument("--limit", type=int, default=None, help="前缀样本数量限制；默认完整 split。")
    parser.add_argument("--hypothesis", default=None, help="写入 memory 的实验假设。")
    parser.add_argument("--decision", default=None, help="baseline / reject / keep / promote_candidate / do_not_repeat")
    parser.add_argument("--tag", action="append", default=[], help="可重复传入，例如 --tag official --tag ensemble")
    parser.add_argument("--no-append-memory", action="store_true", help="只写 run_dir/memory_export.json，不追加长期 memory。")
    parser.add_argument("--promote-slot", default=None, help="可选：写入 leaderboard 的 slot 名称。")
    parser.add_argument("--promote-metric", default="competition_score_proxy")
    parser.add_argument("--promote-blocker", action="append", default=[], help="候选仍需解决的问题，可重复传入。")
    args = parser.parse_args()

    config = load_yaml(args.config)
    result = run_prediction(config, split=args.split, run_name=args.run_name, limit=args.limit)
    run_dir = result["run_dir"]
    metrics = result.get("metrics")
    decision = args.decision or infer_default_decision(args.split, metrics)
    hypothesis = args.hypothesis or f"Replay {config.get('route', 'unknown')} on Task1 {args.split} split."
    tags = args.tag or [str(config.get("route") or "unknown"), args.split]

    memory_export = export_run_memory(run_dir, hypothesis=hypothesis, decision=decision, tags=tags)
    registry = None
    if not args.no_append_memory:
        registry = add_record_to_registry(memory_export)

    promoted = None
    if args.promote_slot:
        if not metrics or args.promote_metric not in metrics:
            raise ValueError(f"无法 promote：metrics 中不存在 {args.promote_metric!r}")
        promoted = promote_candidate(
            f"task1:{result['run_name']}",
            slot=args.promote_slot,
            metric=args.promote_metric,
            value=float(metrics[args.promote_metric]),
            blockers=args.promote_blocker,
        )

    print(
        json.dumps(
            {
                "run_dir": run_dir,
                "prediction_path": result["prediction_path"],
                "metrics": metrics,
                "validation": result["validation"],
                "memory_export": str(memory_export),
                "memory_registry": str(registry) if registry else None,
                "promoted_to": str(promoted) if promoted else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
