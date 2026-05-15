from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from .code_trace import append_code_trace_log
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


def _write_submission_json(path: Path, *, require_llm_code_trace: bool = False) -> None:
    payload = {
        "submission_id": "AI4S-PDE-CNS",
        "problem_id": "PDE_Burgers",
        "code_path": "code",
    }
    if require_llm_code_trace:
        payload["require_llm_code_trace"] = True
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
    require_llm_code_trace: bool = False,
    provenance_log_paths: list[str | Path] | None = None,
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
    for source_log in provenance_log_paths or []:
        source_log = Path(source_log)
        if source_log.resolve() == target_log.resolve():
            continue
        if not source_log.is_file():
            raise FileNotFoundError(f"provenance log not found: {source_log}")
        with target_log.open("a", encoding="utf-8") as out, source_log.open("r", encoding="utf-8") as src:
            for line in src:
                if line.strip():
                    out.write(line if line.endswith("\n") else line + "\n")
    _write_time_csv(output_dir / "task1_time.csv", train_time=train_time, inference_time=inference_time)
    _write_submission_json(output_dir / "submission.json", require_llm_code_trace=require_llm_code_trace)
    _copy_code_dir(code_dir, output_dir / "code")
    if not require_llm_code_trace:
        append_code_trace_log(target_log, output_dir / "code")
    shutil.copy2(methodology_path, output_dir / "methodology.pdf")

    validate_initial_condition(target_prediction, initial_path)
    validate_submission(output_dir)
    return output_dir
