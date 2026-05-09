from __future__ import annotations

import argparse
import csv
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import h5py

from .logging import read_jsonl


class SubmissionError(RuntimeError):
    pass


@dataclass
class ValidationReport:
    valid: bool
    tasks: list[str]
    messages: list[str]


def _prediction_shape(path: Path) -> tuple[int, ...]:
    with h5py.File(path, "r") as h5:
        if "prediction" in h5:
            data = h5["prediction"]
        elif len(h5.keys()) == 1:
            data = h5[next(iter(h5.keys()))]
        else:
            raise SubmissionError(f"{path.name} must contain a prediction dataset")
        return tuple(data.shape)


def _validate_time_csv(path: Path) -> None:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or {"train_time", "inference_time"} - set(reader.fieldnames):
            raise SubmissionError(f"{path.name} must contain train_time and inference_time columns")
        rows = list(reader)
        if not rows:
            raise SubmissionError(f"{path.name} must contain one timing row")
        float(rows[0]["train_time"])
        float(rows[0]["inference_time"])


def _validate_log(path: Path) -> None:
    records = read_jsonl(path)
    if not records:
        raise SubmissionError(f"{path.name} must contain at least one JSON record")
    for idx, record in enumerate(records, start=1):
        if "timestamp" not in record or "elapsed_seconds" not in record:
            raise SubmissionError(f"{path.name}:{idx} missing timestamp or elapsed_seconds")


def validate_submission(path: str | Path) -> ValidationReport:
    root = Path(path)
    if not root.exists():
        raise SubmissionError(f"Submission path does not exist: {root}")
    meta_path = root / "submission.json"
    if not meta_path.exists():
        raise SubmissionError("submission.json is required")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("problem_id") != "PDE_Burgers":
        raise SubmissionError("submission.json problem_id must be PDE_Burgers")
    code_path = root / str(meta.get("code_path", "code"))
    if not code_path.is_dir() or not any(code_path.iterdir()):
        raise SubmissionError("code_path must point to a non-empty directory")

    tasks: list[str] = []
    for task in ("task1", "task2"):
        pred_path = root / f"{task}_pred.hdf5"
        time_path = root / f"{task}_time.csv"
        log_path = root / f"{task}_logs.log"
        present = [pred_path.exists(), time_path.exists(), log_path.exists()]
        if any(present) and not all(present):
            raise SubmissionError(f"{task} must include pred, time, and logs together")
        if not any(present):
            continue
        shape = _prediction_shape(pred_path)
        if len(shape) != 3 or shape[1:] != (200, 256):
            raise SubmissionError(f"{pred_path.name} shape must be (N, 200, 256), got {shape}")
        _validate_time_csv(time_path)
        _validate_log(log_path)
        tasks.append(task)
    if not tasks:
        raise SubmissionError("At least one task bundle is required")
    return ValidationReport(valid=True, tasks=tasks, messages=["ok"])


def pack_submission(root: str | Path, output: str | Path) -> Path:
    source = Path(root)
    output_path = Path(output)
    validate_submission(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source.rglob("*"):
            if path == output_path or path.is_dir():
                continue
            zf.write(path, path.relative_to(source).as_posix())
    return output_path


def _validate_cli() -> None:
    parser = argparse.ArgumentParser(description="Validate an AI4S submission directory.")
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    report = validate_submission(args.path)
    print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))


def _pack_cli() -> None:
    parser = argparse.ArgumentParser(description="Pack an AI4S submission directory.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_path = Path(args.run)
    output = Path(args.output) if args.output else run_path / "submission.zip"
    print(pack_submission(run_path, output))
