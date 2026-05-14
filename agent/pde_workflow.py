from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping

import h5py
import numpy as np

from .logging import utc_now_iso
from .pde_metrics import compute_task1_metrics
from .pde_memory import ExperimentMemory
from .pde_results import RunResult, write_run_result_json
from .pde_tasks import DEFAULT_TASK1_FNO_WEIGHTS, TaskSpec, task1_spec
from .submission import default_pack_path, pack_submission
from .task1_submission import create_task1_submission_bundle

PredictionProvider = Callable[[Path, Mapping[str, float], int], np.ndarray]


TASK1_FNO_CHECKPOINTS: dict[str, str] = {
    "nu0.001": "checkpoints/extracted/1D_Burgers_Sols_Nu0.001_FNO.pt",
}

TASK1_UNET_CHECKPOINTS: dict[str, str] = {
    "unet_pf20_nu0.001": "checkpoints/extracted/1D_Burgers_Sols_Nu0.001_Unet-PF-20.pt",
}

TASK1_OFFICIAL_CHECKPOINTS: dict[str, str] = {
    **TASK1_FNO_CHECKPOINTS,
    **TASK1_UNET_CHECKPOINTS,
}

TASK1_OFFICIAL_MODEL_KINDS: dict[str, str] = {
    "nu0.001": "fno",
    "unet_pf20_nu0.001": "unet_pf20",
}


def _read_tensor(path: Path, preferred_key: str = "tensor") -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if preferred_key in h5:
            return h5[preferred_key][:]
        if "prediction" in h5:
            return h5["prediction"][:]
        if len(h5.keys()) == 1:
            return h5[next(iter(h5.keys()))][:]
        raise KeyError(f"{path} must contain {preferred_key!r}, 'prediction', or one dataset")


def _write_prediction(path: Path, prediction: np.ndarray, *, dataset_key: str = "prediction") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with h5py.File(path, "w") as h5:
        h5.create_dataset(dataset_key, data=prediction.astype(np.float32))


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class Task1FNOWorkflow:
    def __init__(
        self,
        *,
        spec: TaskSpec | None = None,
        run_root: str | Path = "runs",
        code_dir: str | Path = "code",
        methodology_path: str | Path = "docs/methodology.pdf",
        prediction_provider: PredictionProvider | None = None,
        project_root: str | Path = ".",
        checkpoint_paths: Mapping[str, str | Path] | None = None,
    ):
        self.project_root = Path(project_root)
        self.spec = spec or task1_spec(self.project_root)
        if self.spec.task_id != "task1":
            raise ValueError("Task1FNOWorkflow requires a task1 TaskSpec")
        self.run_root = Path(run_root)
        self.code_dir = Path(code_dir)
        self.methodology_path = Path(methodology_path)
        self.prediction_provider = prediction_provider
        paths: dict[str, Path] = {key: Path(value) for key, value in TASK1_OFFICIAL_CHECKPOINTS.items()}
        if checkpoint_paths is not None:
            unknown = sorted(set(checkpoint_paths) - set(TASK1_OFFICIAL_CHECKPOINTS))
            if unknown:
                raise ValueError(
                    f"unknown Task 1 official checkpoint key(s) {unknown}; "
                    f"expected one of {sorted(TASK1_OFFICIAL_CHECKPOINTS)}"
                )
            paths.update({key: Path(value) for key, value in checkpoint_paths.items()})
        self.checkpoint_paths = paths

    def run_validation(
        self,
        weights: Mapping[str, float] | None = None,
        *,
        run_name: str | None = None,
    ) -> RunResult:
        if self.spec.validation_target_path is None:
            raise ValueError("Task 1 validation requires validation_target_path")
        weights = dict(weights or DEFAULT_TASK1_FNO_WEIGHTS)
        run_dir = self._prepare_run_dir(run_name or f"task1-fno-val-{_timestamp()}")
        prediction_path = run_dir / "task1_val_pred.hdf5"
        command: list[str] = []
        start = time.perf_counter()
        try:
            prediction, command = self._generate_prediction(
                self.spec.validation_target_path,
                weights,
                prediction_path,
            )
            target = _read_tensor(self.spec.validation_target_path, "tensor")
            prediction = self._normalize_prediction(prediction, target[:, :10, :])
            _write_prediction(prediction_path, prediction)
            metrics = compute_task1_metrics(prediction, target)
            inference_time = time.perf_counter() - start
            (run_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            result = RunResult(
                task_id=self.spec.task_id,
                run_dir=run_dir,
                metrics=metrics,
                prediction_path=prediction_path,
                zip_path=None,
                train_time=0.0,
                inference_time=inference_time,
                success=True,
                error=None,
                weights=weights,
                command=command,
            )
            self._record_memory(
                run_dir,
                stage="validation",
                weights=weights,
                metrics=metrics,
                result=result,
                conclusion=f"validation mse={metrics['mse']:.8g}",
            )
            write_run_result_json(run_dir, result)
            return result
        except Exception as exc:
            result = self._failure_result(run_dir, weights, prediction_path, time.perf_counter() - start, exc, command)
            self._record_memory(run_dir, stage="validation", weights=weights, metrics={}, result=result, conclusion="failed")
            write_run_result_json(run_dir, result)
            return result

    def run_test_submission(
        self,
        weights: Mapping[str, float] | None = None,
        *,
        run_name: str | None = None,
        train_time: float = 0.0,
    ) -> RunResult:
        weights = dict(weights or DEFAULT_TASK1_FNO_WEIGHTS)
        train_time = float(train_time)
        run_dir = self._prepare_run_dir(run_name or f"task1-fno-submission-{_timestamp()}")
        prediction_path = run_dir / self.spec.prediction_name
        log_path = run_dir / "task1_logs.log"
        command: list[str] = []
        start = time.perf_counter()
        try:
            prediction, command = self._generate_prediction(self.spec.test_input_path, weights, prediction_path)
            initial = _read_tensor(self.spec.initial_condition_path, "tensor")
            prediction = self._normalize_prediction(prediction, initial)
            _write_prediction(prediction_path, prediction)
            inference_time = time.perf_counter() - start
            self._write_log(
                log_path,
                elapsed_seconds=inference_time,
                train_time=train_time,
                weights=weights,
                command=command,
            )
            create_task1_submission_bundle(
                prediction_path=prediction_path,
                initial_path=self.spec.initial_condition_path,
                output_dir=run_dir,
                code_dir=self.code_dir,
                log_path=log_path,
                methodology_path=self.methodology_path,
                train_time=train_time,
                inference_time=inference_time,
            )
            zip_path = pack_submission(run_dir, default_pack_path(run_dir))
            metrics: dict[str, float] = {}
            (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            result = RunResult(
                task_id=self.spec.task_id,
                run_dir=run_dir,
                metrics=metrics,
                prediction_path=prediction_path,
                zip_path=zip_path,
                train_time=train_time,
                inference_time=inference_time,
                success=True,
                error=None,
                weights=weights,
                command=command,
            )
            self._record_memory(
                run_dir,
                stage="test_submission",
                weights=weights,
                metrics=metrics,
                result=result,
                conclusion="packed pred.zip",
            )
            write_run_result_json(run_dir, result)
            return result
        except Exception as exc:
            result = self._failure_result(run_dir, weights, prediction_path, time.perf_counter() - start, exc, command, train_time=train_time)
            self._record_memory(run_dir, stage="test_submission", weights=weights, metrics={}, result=result, conclusion="failed")
            write_run_result_json(run_dir, result)
            return result

    def _prepare_run_dir(self, run_name: str) -> Path:
        run_dir = self.run_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _generate_prediction(
        self,
        input_path: Path,
        weights: Mapping[str, float],
        output_path: Path,
    ) -> tuple[np.ndarray, list[str]]:
        if self.prediction_provider is not None:
            return self.prediction_provider(input_path, weights, self.spec.output_steps), ["prediction_provider"]
        command = self._fno_ensemble_command(input_path, weights, output_path)
        subprocess.run(command, cwd=self.project_root, check=True)
        return _read_tensor(output_path, "prediction"), command

    def _fno_ensemble_command(self, input_path: Path, weights: Mapping[str, float], output_path: Path) -> list[str]:
        keys = [key for key in self.checkpoint_paths if key in weights and float(weights[key]) > 0.0]
        if not keys:
            raise ValueError("At least one positive known official Task 1 checkpoint weight is required")
        model_specs = [
            f"{TASK1_OFFICIAL_MODEL_KINDS[key]}={self.project_root / self.checkpoint_paths[key]}"
            for key in keys
        ]
        values = [str(float(weights[key])) for key in keys]
        return [
            sys.executable,
            str(self.project_root / self.code_dir / "official_checkpoint_ensemble.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--models",
            *model_specs,
            "--weights",
            *values,
        ]

    def _normalize_prediction(self, prediction: np.ndarray, initial: np.ndarray) -> np.ndarray:
        prediction = np.asarray(prediction, dtype=np.float32)
        if initial.ndim != 3 or initial.shape[1] < 10:
            raise ValueError(f"initial condition must have at least 10 frames, got {initial.shape}")
        initial = initial[:, :10, :].astype(np.float32)
        if prediction.ndim != 3 or prediction.shape[0] != initial.shape[0] or prediction.shape[2] != self.spec.spatial_size:
            raise ValueError(f"prediction shape {prediction.shape} is incompatible with initial {initial.shape}")
        if prediction.shape[1] == self.spec.output_steps - 10:
            full = np.zeros((prediction.shape[0], self.spec.output_steps, self.spec.spatial_size), dtype=np.float32)
            full[:, :10, :] = initial
            full[:, 10:, :] = prediction
            prediction = full
        if prediction.shape[1:] != (self.spec.output_steps, self.spec.spatial_size):
            raise ValueError(f"prediction shape must be (N, {self.spec.output_steps}, {self.spec.spatial_size}), got {prediction.shape}")
        prediction[:, :10, :] = initial
        return prediction

    def _record_memory(
        self,
        run_dir: Path,
        *,
        stage: str,
        weights: Mapping[str, float],
        metrics: dict[str, float],
        result: RunResult,
        conclusion: str,
    ) -> None:
        ExperimentMemory(run_dir / "experiment_memory.json").append(
            {
                "task_id": self.spec.task_id,
                "stage": stage,
                "model": "official_checkpoint_ensemble",
                "weights": dict(weights),
                "metrics": metrics,
                "prediction_path": str(result.prediction_path) if result.prediction_path else None,
                "zip_path": str(result.zip_path) if result.zip_path else None,
                "command": result.command,
                "success": result.success,
                "error": result.error,
                "conclusion": conclusion,
            }
        )

    def _write_log(
        self,
        path: Path,
        *,
        elapsed_seconds: float,
        train_time: float,
        weights: Mapping[str, float],
        command: list[str],
    ) -> None:
        record = {
            "timestamp": utc_now_iso(),
            "elapsed_seconds": elapsed_seconds,
            "train_time": train_time,
            "provider": "Task1FNOWorkflow",
            "model": "official_checkpoint_ensemble",
            "messages": [{"role": "system", "content": "Generate Task 1 official checkpoint ensemble submission."}],
            "response": {"weights": dict(weights), "command": command},
        }
        path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    def _failure_result(
        self,
        run_dir: Path,
        weights: Mapping[str, float],
        prediction_path: Path,
        inference_time: float,
        exc: Exception,
        command: list[str],
        train_time: float = 0.0,
    ) -> RunResult:
        return RunResult(
            task_id=self.spec.task_id,
            run_dir=run_dir,
            metrics={},
            prediction_path=prediction_path if prediction_path.exists() else None,
            zip_path=None,
            train_time=float(train_time),
            inference_time=inference_time,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            weights=dict(weights),
            command=command,
        )
