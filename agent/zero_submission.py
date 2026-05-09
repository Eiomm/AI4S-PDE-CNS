from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import time
from pathlib import Path
from typing import Any

from .logging import LLMCallLogger
from .submission import validate_initial_condition, validate_submission


def _load_baseline(code_dir: Path) -> Any:
    path = code_dir / "baseline_stub.py"
    spec = importlib.util.spec_from_file_location("baseline_stub", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load baseline from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_code_dir(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(source, target, ignore=ignore)


def _write_time_csv(path: Path, *, train_time: float, inference_time: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["train_time", "inference_time"])
        writer.writeheader()
        writer.writerow({"train_time": f"{train_time:.6f}", "inference_time": f"{inference_time:.6f}"})


def _write_submission_json(path: Path) -> None:
    payload = {
        "submission_id": "AI4S-PDE-CNS",
        "problem_id": "PDE_Burgers",
        "code_path": "code",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def create_task1_zero_submission(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    code_dir: str | Path,
) -> Path:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    code_dir = Path(code_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = output_dir / "task1_pred.hdf5"
    baseline = _load_baseline(code_dir)
    started = time.perf_counter()
    baseline.copy_initial_condition_baseline(input_path, prediction_path)
    inference_time = time.perf_counter() - started

    _write_time_csv(output_dir / "task1_time.csv", train_time=0.0, inference_time=inference_time)
    _write_submission_json(output_dir / "submission.json")
    _copy_code_dir(code_dir, output_dir / "code")

    logger = LLMCallLogger(output_dir / "task1_logs.log")
    logger.write_call(
        provider="local",
        model="zero-train-baseline",
        messages=[
            {
                "role": "system",
                "content": "Generate a Task 1 zero-train baseline by copying initial conditions.",
            }
        ],
        response={
            "action": "copy_initial_condition_baseline",
            "input_path": str(input_path),
            "prediction_path": str(prediction_path),
        },
        elapsed_seconds=inference_time,
    )

    validate_initial_condition(prediction_path, input_path)
    validate_submission(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a valid Task 1 zero-train submission bundle.")
    parser.add_argument("--input", required=True, help="Path to official task1_test.hdf5.")
    parser.add_argument("--output-dir", required=True, help="Output run directory.")
    parser.add_argument("--code-dir", default="code", help="Submission code directory to copy.")
    args = parser.parse_args()
    print(
        create_task1_zero_submission(
            input_path=args.input,
            output_dir=args.output_dir,
            code_dir=args.code_dir,
        )
    )


if __name__ == "__main__":
    main()
