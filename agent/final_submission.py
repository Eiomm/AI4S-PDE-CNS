from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from .autonomy_audit import AutonomyAuditError, audit_autonomous_study
from .pde_tasks import DEFAULT_TASK1_FNO_WEIGHTS, task1_spec, task2_spec
from .pde_workflow import PredictionProvider, Task1FNOWorkflow
from .submission import default_pack_path, pack_submission, validate_initial_condition, validate_submission
from .submission_workspace import build_submission_workspace
from .task2_workflow import Task2PersistenceWorkflow


def _read_dataset(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if "tensor" in h5:
            return h5["tensor"][:]
        if "prediction" in h5:
            return h5["prediction"][:]
        keys = list(h5.keys())
        if len(keys) == 1:
            return h5[keys[0]][:]
        raise KeyError(f"{path} must contain 'tensor', 'prediction', or exactly one dataset")


def _prediction_shape(path: Path) -> tuple[int, ...]:
    return tuple(_read_dataset(path).shape)


def _first10_max_abs_error(prediction_path: Path, initial_path: Path) -> float:
    pred = _read_dataset(prediction_path)[:, :10, :]
    init = _read_dataset(initial_path)
    return float(np.max(np.abs(pred - init)))


def _task_report(*, task: str, run_dir: Path, initial_path: Path) -> dict[str, object]:
    prediction_path = run_dir / f"{task}_pred.hdf5"
    validate_initial_condition(prediction_path, initial_path)
    return {
        "task": task,
        "prediction": prediction_path.as_posix(),
        "shape": list(_prediction_shape(prediction_path)),
        "first_10_frames_match": True,
        "first_10_max_abs_error": _first10_max_abs_error(prediction_path, initial_path),
    }


def _reset_run_dir(run_root: Path, run_dir: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_root = run_root.resolve()
    resolved_run = run_dir.resolve()
    if resolved_run == resolved_root or resolved_root not in resolved_run.parents:
        raise ValueError(f"refusing to reset run directory outside run_root: {run_dir}")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def create_final_submission(
    *,
    project_root: str | Path = ".",
    run_root: str | Path = "runs",
    run_name: str = "final-official-ensemble-task2-persistence",
    code_dir: str | Path = "code",
    methodology_path: str | Path = "docs/methodology.pdf",
    task1_weights: Mapping[str, float] | None = None,
    task1_extra_inference_args: list[str] | None = None,
    include_task2: bool = True,
    prediction_provider: PredictionProvider | None = None,
    require_llm_code_trace: bool = False,
    provenance_log_paths: list[str | Path] | None = None,
    task1_run: str | Path | None = None,
    task2_run: str | Path | None = None,
    require_autonomy_audit: bool = False,
    task1_study_dir: str | Path | None = None,
    task2_study_dir: str | Path | None = None,
    task1_audit_metric: str = "competition_score_proxy",
    task2_audit_metric: str = "forecast_mse",
) -> dict[str, object]:
    """Create the final official-format pred.zip.

    Task 1 uses the compliant official Nu0.001 FNO + Unet-PF checkpoint ensemble.
    Task 2 is intentionally only a persistence scaffold until a real Task 2 model
    is trained and selected.
    """

    project_root = Path(project_root)
    run_root = Path(run_root)
    run_dir = run_root / run_name
    provided_runs: dict[str, Path] = {}
    if task1_run is not None:
        provided_runs["task1"] = Path(task1_run)
    if task2_run is not None:
        provided_runs["task2"] = Path(task2_run)
    audit_reports: dict[str, dict[str, object]] = {}
    if require_autonomy_audit and not provided_runs:
        raise RuntimeError("autonomy audit failed: provide task run directories and matching study directories")
    if require_autonomy_audit:
        requested_studies = {"task1": task1_study_dir, "task2": task2_study_dir}
        requested_metrics = {"task1": task1_audit_metric, "task2": task2_audit_metric}
        for task in provided_runs:
            study_dir = requested_studies.get(task)
            if study_dir is None:
                raise RuntimeError(f"autonomy audit failed: missing {task}_study_dir")
            try:
                audit_reports[task] = audit_autonomous_study(
                    study_dir,
                    task=task,
                    metric=requested_metrics[task],
                )
            except AutonomyAuditError as exc:
                raise RuntimeError(f"autonomy audit failed for {task}: {exc}") from exc
    if provided_runs:
        output = build_submission_workspace(
            output_dir=run_dir,
            task_runs=provided_runs,
            methodology_path=methodology_path,
            require_llm_code_trace=require_llm_code_trace,
            provenance_log_paths=provenance_log_paths,
        )
        validation = validate_submission(output)
        zip_path = pack_submission(output, default_pack_path(output))
        task_reports = [
            {
                "task": task,
                "prediction": (output / f"{task}_pred.hdf5").as_posix(),
                "shape": list(_prediction_shape(output / f"{task}_pred.hdf5")),
            }
            for task in validation.tasks
        ]
        report = {
            "run_dir": output.as_posix(),
            "zip_path": zip_path.as_posix(),
            "tasks": validation.tasks,
            "task_reports": task_reports,
            "require_llm_code_trace": require_llm_code_trace,
            "require_autonomy_audit": require_autonomy_audit,
            "autonomy_audit_reports": audit_reports,
            "provenance_log_paths": [Path(path).as_posix() for path in provenance_log_paths or []],
            "source_task_runs": {task: path.as_posix() for task, path in provided_runs.items()},
        }
        (output / "final_submission_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    _reset_run_dir(run_root, run_dir)
    weights = dict(task1_weights or DEFAULT_TASK1_FNO_WEIGHTS)

    task1 = task1_spec(project_root)
    task1_result = Task1FNOWorkflow(
        spec=task1,
        run_root=run_root,
        code_dir=code_dir,
        methodology_path=methodology_path,
        project_root=project_root,
        prediction_provider=prediction_provider,
        extra_inference_args=task1_extra_inference_args,
        require_llm_code_trace=require_llm_code_trace,
        provenance_log_paths=provenance_log_paths,
    ).run_test_submission(weights=weights, run_name=run_name)
    if not task1_result.success:
        raise RuntimeError(f"Task 1 final submission failed: {task1_result.error}")

    task_reports = [_task_report(task="task1", run_dir=run_dir, initial_path=task1.initial_condition_path)]

    if include_task2:
        task2 = task2_spec(project_root)
        task2_result = Task2PersistenceWorkflow(
            spec=task2,
            run_root=run_root,
            code_dir=code_dir,
            methodology_path=methodology_path,
            project_root=project_root,
            require_llm_code_trace=require_llm_code_trace,
            provenance_log_paths=provenance_log_paths,
        ).run_test_submission(run_name=run_name)
        if not task2_result.success:
            raise RuntimeError(f"Task 2 persistence scaffold failed: {task2_result.error}")
        task_reports.append(_task_report(task="task2", run_dir=run_dir, initial_path=task2.initial_condition_path))

    validation = validate_submission(run_dir)
    zip_path = pack_submission(run_dir, default_pack_path(run_dir))
    report = {
        "run_dir": run_dir.as_posix(),
        "zip_path": zip_path.as_posix(),
        "tasks": validation.tasks,
        "task_reports": task_reports,
        "task1_weights": weights,
        "task1_extra_inference_args": task1_extra_inference_args or [],
        "require_llm_code_trace": require_llm_code_trace,
        "provenance_log_paths": [Path(path).as_posix() for path in provenance_log_paths or []],
    }
    (run_dir / "final_submission_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final AI4S PDE pred.zip.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--run-name", default="final-official-ensemble-task2-persistence")
    parser.add_argument("--code-dir", default="code")
    parser.add_argument("--methodology-path", default="docs/methodology.pdf")
    parser.add_argument("--task1-weights", nargs=2, type=float, default=None, metavar=("FNO", "UNET_PF20"))
    parser.add_argument("--task1-segment-fno-weights", nargs=3, type=float, default=None, metavar=("SEG1", "SEG2", "SEG3"))
    parser.add_argument("--task1-persistence-segment-alpha", nargs=3, type=float, default=None, metavar=("SEG1", "SEG2", "SEG3"))
    parser.add_argument("--task2", choices=["persistence", "none"], default="persistence")
    parser.add_argument("--task1-run", default=None, help="Use an existing Task 1 run directory instead of regenerating Task 1.")
    parser.add_argument("--task2-run", default=None, help="Use an existing Task 2 run directory instead of regenerating Task 2.")
    parser.add_argument("--require-llm-code-trace", action="store_true")
    parser.add_argument("--require-autonomy-audit", action="store_true")
    parser.add_argument("--task1-study-dir", default=None, help="Autonomous Task 1 study directory to audit before final packaging.")
    parser.add_argument("--task2-study-dir", default=None, help="Autonomous Task 2 study directory to audit before final packaging.")
    parser.add_argument("--task1-audit-metric", default="competition_score_proxy")
    parser.add_argument("--task2-audit-metric", default="forecast_mse")
    parser.add_argument(
        "--provenance-log",
        action="append",
        default=[],
        help="Append real LLM JSONL records, such as autonomous planner_logs.log, into task logs for code provenance.",
    )
    args = parser.parse_args()

    weights = None
    if args.task1_weights is not None:
        weights = {
            "nu0.001": args.task1_weights[0],
            "unet_pf20_nu0.001": args.task1_weights[1],
        }
    extra_inference_args: list[str] = []
    if args.task1_segment_fno_weights is not None:
        extra_inference_args.extend(["--segment-fno-weights", *(str(value) for value in args.task1_segment_fno_weights)])
    if args.task1_persistence_segment_alpha is not None:
        extra_inference_args.extend(["--persistence-segment-alpha", *(str(value) for value in args.task1_persistence_segment_alpha)])
    report = create_final_submission(
        project_root=args.project_root,
        run_root=args.run_root,
        run_name=args.run_name,
        code_dir=args.code_dir,
        methodology_path=args.methodology_path,
        task1_weights=weights,
        task1_extra_inference_args=extra_inference_args,
        include_task2=args.task2 == "persistence",
        require_llm_code_trace=args.require_llm_code_trace,
        provenance_log_paths=args.provenance_log,
        task1_run=args.task1_run,
        task2_run=args.task2_run,
        require_autonomy_audit=args.require_autonomy_audit,
        task1_study_dir=args.task1_study_dir,
        task2_study_dir=args.task2_study_dir,
        task1_audit_metric=args.task1_audit_metric,
        task2_audit_metric=args.task2_audit_metric,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
