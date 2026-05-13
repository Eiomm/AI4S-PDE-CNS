from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .pde_registry import load_registry_records, rank_experiment_records


def _write_table(records: list[dict], *, stream) -> None:
    columns = [
        "study_name",
        "step",
        "status",
        "action_type",
        "metric_name",
        "metric_value",
        "best_candidate_name",
        "run_dir",
    ]
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for record in records:
        writer.writerow({column: record.get(column) for column in columns})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare recorded autonomous PDE experiments.")
    parser.add_argument("--registry", default="runs/experiment_registry.jsonl")
    parser.add_argument("--metric", default="competition_score_proxy")
    parser.add_argument("--maximize", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    records = load_registry_records(Path(args.registry))
    ranked = rank_experiment_records(records, metric=args.metric, maximize=args.maximize)
    _write_table(ranked[: args.top_k], stream=sys.stdout)


if __name__ == "__main__":
    main()
