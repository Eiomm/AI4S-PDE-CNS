from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from .logging import utc_now_iso
from .pde_metrics import compute_task1_metrics
from .pde_results import RunResult, write_run_result_json
from .pde_tasks import TaskSpec, task2_spec
from .submission import default_pack_path, pack_submission
from .task2_submission import create_task2_submission_bundle


def _read_tensor(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if "tensor" in h5:
            return h5["tensor"][:].astype(np.float32)
        if "prediction" in h5:
            return h5["prediction"][:].astype(np.float32)
        if len(h5.keys()) == 1:
            return h5[next(iter(h5.keys()))][:].astype(np.float32)
        raise KeyError(f"{path} must contain a tensor or prediction dataset")


def _load_task2_baseline(code_dir: Path):
    path = code_dir / "task2_persistence_baseline.py"
    spec = importlib.util.spec_from_file_location("task2_persistence_baseline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Task 2 baseline from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Task2PersistenceWorkflow:
    """Minimal Task 2 scaffold for validation and packaging.

    Task 2 is not optimized yet. This workflow only proves that the repository can
    read task2_test.h5, produce a valid (N, 200, 256) prediction, and package it.
    """

    def __init__(
        self,
        *,
        spec: TaskSpec | None = None,
        run_root: str | Path = "runs",
        code_dir: str | Path = "code",
        methodology_path: str | Path = "docs/methodology.pdf",
        project_root: str | Path = ".",
    ):
        self.project_root = Path(project_root)
        self.spec = spec or task2_spec(self.project_root)
        if self.spec.task_id != "task2":
            raise ValueError("Task2PersistenceWorkflow requires a task2 TaskSpec")
        self.run_root = Path(run_root)
        self.code_dir = Path(code_dir)
        self.methodology_path = Path(methodology_path)

    def run_validation(self, *, run_name: str = "task2-persistence-val") -> RunResult:
        if self.spec.validation_target_path is None:
            raise ValueError("Task 2 validation requires validation_target_path")
        run_dir = self.run_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = run_dir / "task2_val_pred.hdf5"
        start = time.perf_counter()
        try:
            baseline = _load_task2_baseline(self.project_root / self.code_dir)
            target_full = _read_tensor(self.spec.validation_target_path)
            initial = target_full[:, :10, :]
            prediction = baseline.persistence_prediction(initial, output_steps=self.spec.output_steps)
            target = target_full[:, : self.spec.output_steps, :]
            baseline.write_prediction(prediction_path, prediction)
            metrics = compute_task1_metrics(prediction, target)
            inference_time = time.perf_counter() - start
            (run_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            result = RunResult(
                task_id="task2",
                run_dir=run_dir,
                metrics=metrics,
                prediction_path=prediction_path,
                zip_path=None,
                train_time=0.0,
                inference_time=inference_time,
                success=True,
                command=[sys.executable, "code/task2_persistence_baseline.py", "--validation"],
            )
        except Exception as exc:
            result = RunResult(
                task_id="task2",
                run_dir=run_dir,
                metrics={},
                prediction_path=prediction_path if prediction_path.exists() else None,
                zip_path=None,
                train_time=0.0,
                inference_time=time.perf_counter() - start,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                command=[sys.executable, "code/task2_persistence_baseline.py", "--validation"],
            )
        write_run_result_json(run_dir, result)
        return result

    def run_test_submission(self, *, run_name: str = "task2-persistence-submission") -> RunResult:
        run_dir = self.run_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = run_dir / self.spec.prediction_name
        log_path = run_dir / "task2_logs.log"
        start = time.perf_counter()
        try:
            baseline = _load_task2_baseline(self.project_root / self.code_dir)
            baseline.run_task2_persistence(self.spec.test_input_path, prediction_path, output_steps=self.spec.output_steps)
            inference_time = time.perf_counter() - start
            log_record = {
                "timestamp": utc_now_iso(),
                "elapsed_seconds": inference_time,
                "provider": "Task2PersistenceWorkflow",
                "model": "task2_persistence_baseline",
                "messages": [{"role": "system", "content": "Generate Task 2 persistence baseline submission."}],
                "response": {"prediction_path": str(prediction_path)},
            }
            log_path.write_text(json.dumps(log_record, ensure_ascii=False) + "\n", encoding="utf-8")
            create_task2_submission_bundle(
                prediction_path=prediction_path,
                initial_path=self.spec.initial_condition_path,
                output_dir=run_dir,
                code_dir=self.project_root / self.code_dir,
                log_path=log_path,
                methodology_path=self.project_root / self.methodology_path,
                train_time=0.0,
                inference_time=inference_time,
            )
            zip_path = pack_submission(run_dir, default_pack_path(run_dir))
            result = RunResult(
                task_id="task2",
                run_dir=run_dir,
                metrics={},
                prediction_path=prediction_path,
                zip_path=zip_path,
                train_time=0.0,
                inference_time=inference_time,
                success=True,
                command=[sys.executable, "code/task2_persistence_baseline.py"],
            )
        except Exception as exc:
            result = RunResult(
                task_id="task2",
                run_dir=run_dir,
                metrics={},
                prediction_path=prediction_path if prediction_path.exists() else None,
                zip_path=None,
                train_time=0.0,
                inference_time=time.perf_counter() - start,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                command=[sys.executable, "code/task2_persistence_baseline.py"],
            )
        write_run_result_json(run_dir, result)
        return result
