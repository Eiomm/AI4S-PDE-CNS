from __future__ import annotations

import argparse
import csv
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .code_trace import validate_code_log_consistency
from .logging import read_jsonl


class SubmissionError(RuntimeError):
    pass


PREDICTION_DATASET_KEY = "tensor"


@dataclass
class ValidationReport:
    valid: bool
    tasks: list[str]
    messages: list[str]


def _prediction_shape(path: Path) -> tuple[int, ...]:
    with h5py.File(path, "r") as h5:
        if PREDICTION_DATASET_KEY not in h5:
            raise SubmissionError(f"{path.name} must contain a {PREDICTION_DATASET_KEY!r} dataset")
        data = h5[PREDICTION_DATASET_KEY]
        return tuple(data.shape)


def _read_prediction(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if PREDICTION_DATASET_KEY in h5:
            return h5[PREDICTION_DATASET_KEY][:]
        if "prediction" in h5:
            return h5["prediction"][:]
        if len(h5.keys()) == 1:
            return h5[next(iter(h5.keys()))][:]
        raise SubmissionError(f"{path.name} must contain a prediction dataset")


def write_official_prediction_file(source_path: str | Path, target_path: str | Path) -> None:
    """Write a submission HDF5 file using the official prediction dataset key."""
    source = Path(source_path)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f"{target.name}.tmp")
    if temp.exists():
        temp.unlink()

    with h5py.File(source, "r") as src, h5py.File(temp, "w") as dst:
        if PREDICTION_DATASET_KEY in src:
            data = src[PREDICTION_DATASET_KEY]
        elif "prediction" in src:
            data = src["prediction"]
        elif len(src.keys()) == 1:
            data = src[next(iter(src.keys()))]
        else:
            raise SubmissionError(f"{source.name} must contain a prediction dataset")
        output = dst.create_dataset(PREDICTION_DATASET_KEY, data=data, dtype=data.dtype)
        for key, value in data.attrs.items():
            output.attrs[key] = value

    temp.replace(target)


def _read_initial_tensor(path: Path, dataset_key: str = "tensor") -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if dataset_key in h5:
            return h5[dataset_key][:]
        if len(h5.keys()) == 1:
            return h5[next(iter(h5.keys()))][:]
        raise SubmissionError(f"{path.name} must contain a {dataset_key!r} dataset")


def validate_initial_condition(
    prediction_path: str | Path,
    initial_path: str | Path,
    *,
    dataset_key: str = "tensor",
    atol: float = 1e-3,
) -> None:
    """Verify the first 10 predicted frames match the official initial condition."""
    pred = _read_prediction(Path(prediction_path))
    init = _read_initial_tensor(Path(initial_path), dataset_key)
    if pred.ndim != 3 or pred.shape[1] < 10 or pred.shape[2] != 256:
        raise SubmissionError(f"prediction shape must be (N, >=10, 256), got {pred.shape}")
    if init.ndim != 3 or init.shape[1] != 10 or init.shape[2] != 256:
        raise SubmissionError(f"initial condition shape must be (N, 10, 256), got {init.shape}")
    if pred.shape[0] != init.shape[0]:
        raise SubmissionError(f"prediction sample count {pred.shape[0]} does not match initial {init.shape[0]}")
    if not np.allclose(pred[:, :10, :], init, atol=atol, rtol=0.0):
        raise SubmissionError("prediction first 10 frames do not match the initial condition")


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
    if not (root / "methodology.pdf").is_file():
        raise SubmissionError("methodology.pdf is required")

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
    try:
        validate_code_log_consistency(
            code_dir=code_path,
            log_paths=[root / f"{task}_logs.log" for task in tasks],
            code_root_name=code_path.relative_to(root).as_posix(),
        )
    except ValueError as exc:
        raise SubmissionError(f"Code-log consistency check failed: {exc}") from exc
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


def default_pack_path(run_path: str | Path) -> Path:
    return Path(run_path) / "pred.zip"


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
    output = Path(args.output) if args.output else default_pack_path(run_path)
    print(pack_submission(run_path, output))
