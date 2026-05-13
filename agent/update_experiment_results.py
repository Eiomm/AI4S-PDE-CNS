from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment_results import cleanup_experiment_artifacts, collect_experiment_records, write_experiment_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Update the Task 1 experiment result ledger.")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output", default="docs/results/task1_experiment_results.md")
    parser.add_argument("--csv", default="runs/experiment_results_summary.csv")
    parser.add_argument("--json", default="runs/experiment_results_summary.json")
    parser.add_argument("--top-n", type=int, default=80)
    parser.add_argument("--cleanup-zips", action="store_true", help="Delete stale runs/**/pred.zip files.")
    parser.add_argument("--keep-zip", action="append", default=None, help="Zip path to preserve when --cleanup-zips is used.")
    parser.add_argument("--delete-partial", action="append", default=None, help="Run directory or file under runs/ to delete after recording.")
    parser.add_argument("--cleanup-manifest", default="runs/cleanup_manifest.json")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    records = collect_experiment_records(runs_root)
    output = write_experiment_results(records, output_path=args.output, csv_path=args.csv, json_path=args.json, top_n=args.top_n)
    cleanup_manifest = None
    if args.cleanup_zips or args.delete_partial:
        cleanup_manifest = cleanup_experiment_artifacts(
            runs_root,
            cleanup_zips=args.cleanup_zips,
            keep_zips=[Path(path) for path in args.keep_zip or []],
            delete_partials=[Path(path) for path in args.delete_partial or []],
        )
        manifest_path = Path(args.cleanup_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(cleanup_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    if cleanup_manifest is not None:
        print(f"cleanup_deleted={len(cleanup_manifest['deleted'])}")


if __name__ == "__main__":
    main()
