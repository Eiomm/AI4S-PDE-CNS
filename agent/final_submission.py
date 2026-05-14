from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from .pde_tasks import DEFAULT_TASK1_FNO_WEIGHTS, task1_spec, task2_spec
from .pde_workflow import PredictionProvider, Task1FNOWorkflow
from .submission import default_pack_path, pack_submission, validate_initial_condition, validate_submission
from .task2_workflow import Task2PersistenceWorkflow


def _prediction_shape(path: Path) -> tuple[int, ...]:
    with h5py.File(path, "r") as h5:
        return tuple(h5["tensor"].shape)


def _first10_max_abs_error(prediction_path: Path, initial_path: Path) -> float:
    with h5py.File(prediction_path, "r") as pred_h5, h5py.File(initial_path, "r") as init_h5:
        pred = pred_h5["tensor"][:, :10, :]
        init = init_h5["tensor"][:]
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


def create_final_submission(
    *,
    project_root: str | Path = ".",
    run_root: str | Path = "runs",
    run_name: str = "final-official-ensemble-task2-persistence",
    code_dir: str | Path = "code",
    methodology_path: str | Path = "docs/methodology.pdf",
    task1_weights: Mapping[str, float] | None = None,
    include_task2: bool = True,
    prediction_provider: PredictionProvider | None = None,
) -> dict[str, object]:
    """Create the final official-format pred.zip.

    Task 1 uses the compliant official Nu0.001 FNO + Unet-PF checkpoint ensemble.
    Task 2 is intentionally only a persistence scaffold until a real Task 2 model
    is trained and selected.
    """

    project_root = Path(project_root)
    run_root = Path(run_root)
    run_dir = run_root / run_name
    weights = dict(task1_weights or DEFAULT_TASK1_FNO_WEIGHTS)

    task1 = task1_spec(project_root)
    task1_result = Task1FNOWorkflow(
        spec=task1,
        run_root=run_root,
        code_dir=code_dir,
        methodology_path=methodology_path,
        project_root=project_root,
        prediction_provider=prediction_provider,
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
    parser.add_argument("--task2", choices=["persistence", "none"], default="persistence")
    args = parser.parse_args()

    weights = None
    if args.task1_weights is not None:
        weights = {
            "nu0.001": args.task1_weights[0],
            "unet_pf20_nu0.001": args.task1_weights[1],
        }
    report = create_final_submission(
        project_root=args.project_root,
        run_root=args.run_root,
        run_name=args.run_name,
        code_dir=args.code_dir,
        methodology_path=args.methodology_path,
        task1_weights=weights,
        include_task2=args.task2 == "persistence",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
