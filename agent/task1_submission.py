from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from .submission import validate_initial_condition, validate_submission, write_official_prediction_file


def _copy_code_dir(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    shutil.copytree(source, target, ignore=ignore)


def _write_time_csv(path: Path, *, train_time: float, inference_time: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["train_time", "inference_time"])
        writer.writeheader()
        writer.writerow(
            {
                "train_time": f"{float(train_time):.6f}",
                "inference_time": f"{float(inference_time):.6f}",
            }
        )


def _write_submission_json(path: Path) -> None:
    payload = {
        "submission_id": "AI4S-PDE-CNS",
        "problem_id": "PDE_Burgers",
        "code_path": "code",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def create_task1_submission_bundle(
    *,
    prediction_path: str | Path,
    initial_path: str | Path,
    output_dir: str | Path,
    code_dir: str | Path,
    log_path: str | Path,
    methodology_path: str | Path = "docs/methodology.pdf",
    train_time: float,
    inference_time: float,
) -> Path:
    """Create and validate a Task 1-only AI4S submission directory."""

    prediction_path = Path(prediction_path)
    initial_path = Path(initial_path)
    output_dir = Path(output_dir)
    code_dir = Path(code_dir)
    log_path = Path(log_path)
    methodology_path = Path(methodology_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_prediction = output_dir / "task1_pred.hdf5"
    target_log = output_dir / "task1_logs.log"
    write_official_prediction_file(prediction_path, target_prediction)
    if log_path.resolve() != target_log.resolve():
        shutil.copy2(log_path, target_log)
    _write_time_csv(output_dir / "task1_time.csv", train_time=train_time, inference_time=inference_time)
    _write_submission_json(output_dir / "submission.json")
    _copy_code_dir(code_dir, output_dir / "code")
    shutil.copy2(methodology_path, output_dir / "methodology.pdf")

    validate_initial_condition(target_prediction, initial_path)
    validate_submission(output_dir)
    return output_dir
