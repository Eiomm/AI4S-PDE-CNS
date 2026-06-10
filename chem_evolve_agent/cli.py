from __future__ import annotations

import argparse
from pathlib import Path

from chem_evolve_agent.logging_utils import json_event
from chem_evolve_agent.submitter import write_final_result_zip, write_single_target_result
from chem_evolve_agent.workflow.runner import run_target_agent_pipeline, run_target_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--per-round", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--mode", choices=["proxy", "docking", "competition"], default="proxy")
    parser.add_argument("--docking-limit", type=int, default=8)
    parser.add_argument("--runner", choices=["legacy", "agent"], default="legacy")
    args = parser.parse_args()

    out_dir = Path(args.out)
    stems: list[str] = []
    runner = run_target_agent_pipeline if args.runner == "agent" else run_target_pipeline
    for index, target in enumerate(args.targets, start=1):
        stem = "result" if len(args.targets) == 1 else f"result{index}"
        candidates, logs = runner(
            target_path=Path(target),
            out_dir=out_dir,
            rounds=args.rounds,
            per_round=args.per_round,
            mode=args.mode,
            docking_limit=args.docking_limit,
        )
        logs.append(json_event("submit", stem=stem, top_k=args.top_k))
        write_single_target_result(out_dir, stem, candidates[: args.top_k], logs)
        stems.append(stem)

    if len(stems) > 1:
        write_final_result_zip(out_dir, stems)


if __name__ == "__main__":
    main()
