from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Mapping

from .code_trace import append_code_trace_log


class SubmissionWorkspaceError(RuntimeError):
    pass


def _write_submission_json(path: Path, *, require_llm_code_trace: bool = False) -> None:
    payload = {
        "submission_id": "AI4S-PDE-CNS",
        "problem_id": "PDE_Burgers",
        "code_path": "code",
    }
    if require_llm_code_trace:
        payload["require_llm_code_trace"] = True
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _copy_required_task_files(*, task: str, source_run: Path, output_dir: Path) -> None:
    for suffix in ("pred.hdf5", "time.csv", "logs.log"):
        source = source_run / f"{task}_{suffix}"
        if not source.is_file():
            raise SubmissionWorkspaceError(f"Missing {task} artifact: {source}")
        shutil.copy2(source, output_dir / source.name)


def _merge_code_dir(source: Path, target: Path, *, task: str) -> None:
    if not source.is_dir():
        raise SubmissionWorkspaceError(f"{task} run has no code directory: {source}")
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(source)
        if relative.as_posix() == "code_manifest.json":
            continue
        destination = target / relative
        if destination.exists():
            if destination.read_bytes() != path.read_bytes():
                raise SubmissionWorkspaceError(
                    f"Shared code collision: code/{relative.as_posix()} differs while merging {task} run."
                )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def build_submission_workspace(
    *,
    output_dir: str | Path,
    task_runs: Mapping[str, str | Path],
    methodology_path: str | Path,
    require_llm_code_trace: bool = False,
    provenance_log_paths: list[str | Path] | None = None,
) -> Path:
    output = Path(output_dir)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    code_dir = output / "code"
    code_dir.mkdir()

    normalized_runs = {str(task): Path(run) for task, run in task_runs.items()}
    for task in sorted(normalized_runs):
        if task not in {"task1", "task2"}:
            raise SubmissionWorkspaceError(f"Unsupported task name: {task}")
        source_run = normalized_runs[task]
        _copy_required_task_files(task=task, source_run=source_run, output_dir=output)
        _merge_code_dir(source_run / "code", code_dir, task=task)

    methodology = Path(methodology_path)
    if not methodology.is_file():
        raise SubmissionWorkspaceError(f"methodology.pdf not found: {methodology}")
    shutil.copy2(methodology, output / "methodology.pdf")
    _write_submission_json(output / "submission.json", require_llm_code_trace=require_llm_code_trace)

    for log_path in provenance_log_paths or []:
        source = Path(log_path)
        if not source.is_file():
            raise SubmissionWorkspaceError(f"provenance log not found: {source}")
        for task in normalized_runs:
            target = output / f"{task}_logs.log"
            with target.open("a", encoding="utf-8") as out, source.open("r", encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")

    if not require_llm_code_trace:
        for task in normalized_runs:
            append_code_trace_log(output / f"{task}_logs.log", code_dir)
    return output
