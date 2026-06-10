#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chem_evolve_agent.experiments import analyze_run_directory


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan AI4S outputs and build a human-readable run index.")
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    args = parser.parse_args()

    records = scan_outputs(args.outputs)
    index_dir = args.outputs / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    csv_path = index_dir / "run_index.csv"
    jsonl_path = index_dir / "run_index.jsonl"
    readme_path = args.outputs / "README.md"

    write_csv(csv_path, records)
    write_jsonl(jsonl_path, records)
    write_readme(readme_path, args.outputs, records, csv_path)

    print(f"indexed_runs={len(records)}")
    print(f"csv={csv_path}")
    print(f"jsonl={jsonl_path}")
    print(f"readme={readme_path}")


def scan_outputs(outputs: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for done_path in sorted(outputs.glob("*/*/harness.done")):
        run_dir = done_path.parent
        if run_dir.name == "latest" or run_dir.is_symlink():
            continue
        metadata = parse_done(done_path)
        metrics = analyze_run_directory(run_dir)
        run_group = run_dir.parent.name
        records.append(
            {
                "category": categorize(run_group),
                "run_group": run_group,
                "run_id": run_dir.name,
                "status": metadata.get("exit_code", metrics.status),
                "mode": metadata.get("mode", ""),
                "target": metadata.get("target", ""),
                "rounds": metadata.get("rounds", ""),
                "per_round": metadata.get("per_round", ""),
                "top_k": metadata.get("top_k", ""),
                "docking_limit": metadata.get("docking_limit", ""),
                "objective": metrics.objective,
                "best_total": metrics.best_total,
                "avg_total": metrics.avg_total,
                "candidate_count": metrics.candidate_count,
                "valid_smiles": metrics.valid_smiles,
                "unique_smiles": metrics.unique_smiles,
                "scaffold_count": metrics.scaffold_count,
                "llm_calls": metrics.llm_calls,
                "best_smiles": metrics.best_smiles,
                "run_dir": str(run_dir),
            }
        )
    records.sort(key=lambda item: (str(item["run_group"]), str(item["run_id"])))
    return records


def parse_done(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def categorize(run_group: str) -> str:
    if run_group.startswith("goal_"):
        return "auto_iteration"
    if run_group.startswith("verify_"):
        return "verification"
    if "final" in run_group:
        return "final_packaging"
    return "manual_or_other"


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    keys = [
        "category",
        "run_group",
        "run_id",
        "status",
        "mode",
        "objective",
        "best_total",
        "avg_total",
        "candidate_count",
        "valid_smiles",
        "unique_smiles",
        "scaffold_count",
        "llm_calls",
        "target",
        "run_dir",
        "best_smiles",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in keys})


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_readme(path: Path, outputs: Path, records: list[dict[str, Any]], csv_path: Path) -> None:
    by_category: dict[str, int] = {}
    for record in records:
        by_category[record["category"]] = by_category.get(record["category"], 0) + 1
    lines = [
        "# AI4S Outputs",
        "",
        "This directory keeps generated experiment artifacts. Current historical runs are not moved; use the generated index to find them.",
        "",
        "## Recommended Layout",
        "",
        "- `strategy_memory/`: cross-run memory, best strategy notes, and failure notes.",
        "- `index/`: generated run index files for humans and scripts.",
        "- `<run_group>/<timestamp>/`: legacy harness run directories; each contains `candidates.csv`, `pipeline.log`, `submission.zip`, and `harness.done`.",
        "",
        "For new agent systems, follow this mental model from common experiment trackers: experiment -> run -> artifacts/logs/metrics/config.",
        "",
        "## Current Index",
        "",
        f"- CSV: `{csv_path.relative_to(outputs)}`",
        f"- Total indexed runs: {len(records)}",
        "",
        "## Categories",
        "",
    ]
    for category, count in sorted(by_category.items()):
        lines.append(f"- `{category}`: {count}")
    lines.extend(
        [
            "",
            "## How To Refresh",
            "",
            "```bash",
            "python scripts/index_outputs.py",
            "```",
            "",
            "The most important fields are `category`, `run_group`, `run_id`, `objective`, `best_total`, `llm_calls`, and `run_dir`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
