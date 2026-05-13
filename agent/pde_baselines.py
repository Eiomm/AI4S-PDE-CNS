from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import h5py
import numpy as np

from .pde_memory import ExperimentMemory
from .pde_metrics import compute_task1_metrics
from .pde_results import RunResult, write_run_result_json
from .pde_tasks import TaskSpec, task1_spec
from .pde_workflow import Task1FNOWorkflow


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    family: str
    trainable: bool
    train_config: dict[str, Any] = field(default_factory=dict)
    predict_config: dict[str, Any] = field(default_factory=dict)
    required_artifacts: tuple[str, ...] = ("metrics.json", "run_result.json", "experiment_memory.json", "baseline_manifest.json")

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_artifacts"] = list(self.required_artifacts)
        return payload


class BaselineWorkflow(Protocol):
    spec: BaselineSpec

    def run_validation(self, config: dict[str, Any] | None = None, *, run_name: str | None = None) -> RunResult:
        ...

    def run_test_submission(self, config: dict[str, Any] | None = None, *, run_name: str | None = None) -> RunResult:
        ...


class BaselineRegistry:
    def __init__(self, specs: list[BaselineSpec] | None = None):
        self._specs: dict[str, BaselineSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: BaselineSpec) -> None:
        if not spec.name:
            raise ValueError("baseline name must not be empty")
        self._specs[spec.name] = spec

    def get(self, name: str) -> BaselineSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"unknown baseline: {name}") from exc

    def names(self) -> list[str]:
        return list(self._specs)

    def specs(self) -> list[BaselineSpec]:
        return [self._specs[name] for name in self.names()]


class FNOBaselineWorkflow:
    def __init__(self, workflow: Task1FNOWorkflow, spec: BaselineSpec | None = None):
        self.workflow = workflow
        self.spec = spec or BaselineSpec(name="fno_ensemble", family="FNO", trainable=False)

    def run_validation(self, config: dict[str, Any] | None = None, *, run_name: str | None = None) -> RunResult:
        config = dict(config or {})
        return self.workflow.run_validation(config.get("weights"), run_name=run_name)

    def run_test_submission(self, config: dict[str, Any] | None = None, *, run_name: str | None = None) -> RunResult:
        config = dict(config or {})
        return self.workflow.run_test_submission(
            config.get("weights"),
            run_name=run_name,
            train_time=float(config.get("train_time", 0.0)),
        )


class FakeTask1BaselineWorkflow:
    def __init__(
        self,
        baseline_spec: BaselineSpec,
        *,
        spec: TaskSpec | None = None,
        spec_task: TaskSpec | None = None,
        spec_override: TaskSpec | None = None,
        task_spec: TaskSpec | None = None,
        run_root: str | Path = "runs",
        fill_value: float = 0.0,
        **kwargs: Any,
    ):
        self.spec = baseline_spec
        self.task_spec = task_spec or spec or spec_override or spec_task or task1_spec()
        self.run_root = Path(run_root)
        self.fill_value = float(fill_value)

    def run_validation(self, config: dict[str, Any] | None = None, *, run_name: str | None = None) -> RunResult:
        if self.task_spec.validation_target_path is None:
            raise ValueError("validation target path is required")
        run_dir = self.run_root / (run_name or f"{self.spec.name}-val")
        target = _read_tensor(self.task_spec.validation_target_path)
        prediction = np.full_like(target[:, : self.task_spec.output_steps, :], self.fill_value, dtype=np.float32)
        prediction[:, :10, :] = target[:, :10, :]
        return _write_baseline_validation_result(
            baseline=self.spec,
            task_spec=self.task_spec,
            run_dir=run_dir,
            prediction=prediction,
            target=target,
            config=dict(config or {}),
            command=["fake_baseline"],
            train_time=0.0,
            inference_time=0.0,
        )

    def run_test_submission(self, config: dict[str, Any] | None = None, *, run_name: str | None = None) -> RunResult:
        run_dir = self.run_root / (run_name or f"{self.spec.name}-submission")
        initial = _read_tensor(self.task_spec.initial_condition_path)
        prediction = np.full((initial.shape[0], self.task_spec.output_steps, self.task_spec.spatial_size), self.fill_value, dtype=np.float32)
        prediction[:, :10, :] = initial[:, :10, :]
        prediction_path = run_dir / self.task_spec.prediction_name
        _write_prediction(prediction_path, prediction)
        result = RunResult(
            task_id=self.task_spec.task_id,
            run_dir=run_dir,
            metrics={},
            prediction_path=prediction_path,
            zip_path=None,
            train_time=0.0,
            inference_time=0.0,
            success=True,
            command=["fake_baseline"],
        )
        write_baseline_artifacts(run_dir, self.spec, dict(config or {}), result)
        return result


def build_default_task1_baseline_registry() -> BaselineRegistry:
    return BaselineRegistry(
        [
            BaselineSpec(name="fno_ensemble", family="FNO", trainable=False),
            BaselineSpec(name="tfno", family="TFNO", trainable=True, train_config={"optional_dependency": "neuralop"}),
            BaselineSpec(name="unet1d", family="U-Net", trainable=True),
            BaselineSpec(name="deeponet_lite", family="DeepONet", trainable=True),
            BaselineSpec(name="pino_fno", family="PINO-FNO", trainable=True),
            BaselineSpec(name="residual_refiner", family="Refiner", trainable=True),
        ]
    )


def _read_tensor(path: str | Path, preferred_key: str = "tensor") -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if preferred_key in h5:
            return h5[preferred_key][:]
        if "prediction" in h5:
            return h5["prediction"][:]
        if len(h5.keys()) == 1:
            return h5[next(iter(h5.keys()))][:]
        raise KeyError(f"{path} must contain {preferred_key!r}, 'prediction', or one dataset")


def _write_prediction(path: str | Path, prediction: np.ndarray, *, dataset_key: str = "prediction") -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset(dataset_key, data=np.asarray(prediction, dtype=np.float32))
    return output_path


def write_baseline_artifacts(
    run_dir: str | Path,
    baseline: BaselineSpec,
    config: dict[str, Any],
    result: RunResult,
    *,
    conclusion: str | None = None,
) -> None:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "baseline": baseline.to_json_dict(),
        "config": config,
        "result": result.to_json_dict(),
    }
    (run_path / "baseline_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ExperimentMemory(run_path / "experiment_memory.json").append(
        {
            "model": baseline.name,
            "family": baseline.family,
            "trainable": baseline.trainable,
            "config": config,
            "metrics": dict(result.metrics),
            "prediction_path": str(result.prediction_path) if result.prediction_path else None,
            "zip_path": str(result.zip_path) if result.zip_path else None,
            "command": result.command,
            "success": result.success,
            "error": result.error,
            "conclusion": conclusion or ("completed" if result.success else "failed"),
        }
    )
    write_run_result_json(run_path, result)


def _write_baseline_validation_result(
    *,
    baseline: BaselineSpec,
    task_spec: TaskSpec,
    run_dir: Path,
    prediction: np.ndarray,
    target: np.ndarray,
    config: dict[str, Any],
    command: list[str],
    train_time: float,
    inference_time: float,
) -> RunResult:
    started = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction = np.asarray(prediction, dtype=np.float32)
    prediction[:, :10, :] = target[:, :10, :]
    prediction_path = _write_prediction(run_dir / "task1_val_pred.hdf5", prediction)
    metrics = compute_task1_metrics(prediction, target)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = RunResult(
        task_id=task_spec.task_id,
        run_dir=run_dir,
        metrics=metrics,
        prediction_path=prediction_path,
        zip_path=None,
        train_time=float(train_time),
        inference_time=float(inference_time) + (time.perf_counter() - started),
        success=True,
        command=command,
    )
    write_baseline_artifacts(run_dir, baseline, config, result, conclusion=f"validation mse={metrics['mse']:.8g}")
    return result
