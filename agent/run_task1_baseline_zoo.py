from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from .logging import utc_now_iso
from .pde_baselines import (
    BaselineSpec,
    FakeTask1BaselineWorkflow,
    FNOBaselineWorkflow,
    build_default_task1_baseline_registry,
    write_baseline_artifacts,
)
from .pde_gating import fit_cluster_em_ensemble, fit_global_convex_ensemble, fit_temporal_tail_blend
from .pde_journal import CandidatePlan, ExperimentJournal
from .pde_registry import export_experiment_records
from .pde_results import RunResult
from .pde_tasks import task1_spec
from .pde_workflow import Task1FNOWorkflow
from .physicsnemo_adapter import physicsnemo_status
from .task1_baseline_train import train_task1_baseline


PHYSICSNEMO_MODELS = {"physicsnemo_fno", "physicsnemo_transolver"}
PROTOTYPE_TRAINABLE_MODELS = {"unet1d", "deeponet_lite", "residual_refiner", "pino_fno", "tfno"} | PHYSICSNEMO_MODELS


def parse_key_value_paths(values: list[str] | None) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"expected KEY=PATH, got {value!r}")
        key, raw_path = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty key in {value!r}")
        output[key] = Path(raw_path)
    return output


def parse_key_value_floats(values: list[str] | None) -> dict[str, float]:
    output: dict[str, float] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"expected KEY=VALUE, got {value!r}")
        key, raw_number = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty key in {value!r}")
        output[key] = float(raw_number)
    return output


def build_task1_fno_workflow(
    *,
    project_root: str | Path,
    study_dir: str | Path,
    checkpoint_overrides: Mapping[str, str | Path] | None = None,
) -> Task1FNOWorkflow:
    return Task1FNOWorkflow(
        project_root=project_root,
        run_root=study_dir,
        checkpoint_paths=checkpoint_overrides,
    )


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _neuralop_available() -> bool:
    return importlib.util.find_spec("neuralop") is not None


def _read_prediction(path: str | Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if "prediction" in h5:
            return h5["prediction"][:]
        if "tensor" in h5:
            return h5["tensor"][:]
        return h5[next(iter(h5.keys()))][:]


def _write_prediction(path: Path, prediction: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with h5py.File(path, "w") as h5:
        h5.create_dataset("prediction", data=np.asarray(prediction, dtype=np.float32))
    return path


def _write_ensemble_result(
    *,
    study_dir: Path,
    name: str,
    prediction: np.ndarray,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    run_dir = study_dir / name
    prediction_path = _write_prediction(run_dir / "task1_val_pred.hdf5", prediction)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = RunResult(
        task_id="task1",
        run_dir=run_dir,
        metrics=dict(metrics),
        prediction_path=prediction_path,
        zip_path=None,
        train_time=0.0,
        inference_time=0.0,
        success=True,
        command=["baseline_ensemble", name],
    )
    write_baseline_artifacts(
        run_dir,
        BaselineSpec(name=name, family="ensemble", trainable=False),
        config,
        result,
        conclusion=f"{name} validation competition_score_proxy={metrics.get('competition_score_proxy')}",
    )
    return {
        "name": name,
        "success": True,
        "metrics": dict(metrics),
        "run_dir": str(run_dir),
        "prediction_path": str(prediction_path),
    }


def run_validation_ensembles(
    *,
    study_dir: str | Path,
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    grid_step: float = 0.05,
) -> list[dict[str, Any]]:
    if len(predictions) < 2:
        return []
    study_path = Path(study_dir)
    outputs: list[dict[str, Any]] = []
    global_result = fit_global_convex_ensemble(predictions, target, grid_step=grid_step)
    outputs.append(
        _write_ensemble_result(
            study_dir=study_path,
            name="global_ensemble",
            prediction=global_result.prediction,
            metrics=global_result.metrics,
            config={"kind": "global_convex", "weights": global_result.weights, "grid_step": grid_step},
        )
    )
    if target.shape[0] >= 2:
        cluster_result = fit_cluster_em_ensemble(
            target[:, :10, :],
            predictions,
            target,
            n_clusters=min(3, target.shape[0]),
        )
        outputs.append(
            _write_ensemble_result(
                study_dir=study_path,
                name="cluster_em_ensemble",
                prediction=cluster_result.prediction,
                metrics=cluster_result.metrics,
                config={
                    "kind": "feature_cluster_em",
                    "cluster_weights": cluster_result.cluster_weights,
                    "n_clusters": int(len(cluster_result.cluster_weights)),
                },
            )
        )
    if "fno_ensemble" in predictions:
        for tail_name in sorted(name for name in predictions if name != "fno_ensemble"):
            temporal_result = fit_temporal_tail_blend(
                predictions,
                target,
                base_name="fno_ensemble",
                tail_name=tail_name,
            )
            outputs.append(
                _write_ensemble_result(
                    study_dir=study_path,
                    name=f"temporal_tail_blend_{tail_name}",
                    prediction=temporal_result.prediction,
                    metrics=temporal_result.metrics,
                    config=temporal_result.config,
                )
            )
    return outputs


def _append_result(
    journal: ExperimentJournal,
    *,
    model: str,
    action_type: str,
    success: bool,
    metrics: dict[str, float] | None = None,
    artifacts: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    node_artifacts = dict(artifacts or {})
    node_artifacts.setdefault(
        "candidate_results",
        [
            {
                "name": model,
                "metrics": dict(metrics or {}),
                "success": success,
                "run_dir": node_artifacts.get("run_dir"),
                "prediction_path": node_artifacts.get("prediction_path"),
                "zip_path": node_artifacts.get("zip_path"),
                "weights": node_artifacts.get("weights", {}),
                "error": error,
            }
        ],
    )
    node = journal.append_plan(
        CandidatePlan(
            intent=f"run {model}",
            hypothesis=f"Evaluate Baseline Zoo model {model} under the current Task 1 validation gate.",
            action_type=action_type,
            params={"model": model},
            expected_effect="Collect comparable validation metrics and artifacts.",
            risk="Prototype may be skipped when dependencies are unavailable.",
        )
    )
    journal.mark_running(node.id)
    journal.update_result(
        node.id,
        success=success,
        metrics=metrics or {},
        artifacts=node_artifacts,
        error=error,
    )


def _skip_model(
    journal: ExperimentJournal,
    *,
    model: str,
    reason: str,
) -> dict[str, Any]:
    _append_result(
        journal,
        model=model,
        action_type="baseline_train",
        success=False,
        artifacts={"model": model, "skipped": True, "reason": reason},
        error=reason,
    )
    return {"model": model, "success": False, "skipped": True, "error": reason}


def run_task1_baseline_zoo(
    *,
    project_root: str | Path = ".",
    study_name: str = "task1-zoo-prototype",
    models: list[str] | None = None,
    max_samples: int = 1024,
    steps: int = 200,
    batch_size: int = 4,
    lr: float = 1.0e-3,
    hidden: int = 64,
    device: str | None = None,
    loss_start_step: int = 10,
    loss_end_step: int | None = None,
    checkpoint_overrides: Mapping[str, str | Path] | None = None,
    fno_weights: Mapping[str, float] | None = None,
    base_train_hdf5: list[str | Path] | None = None,
    base_validation_prediction_path: str | Path | None = None,
    initial_loss_weight: float = 0.05,
    spectral_loss_weight: float = 0.0,
    spectral_high_weight: float = 2.0,
    physics_loss_weight: float = 0.0,
    physics_nu: float = 0.001,
    physics_dt: float = 0.05,
    physics_dx: float = 1.0 / 256.0,
) -> Path:
    root = Path(project_root)
    runs_root = root / "runs"
    study_dir = runs_root / study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    journal = ExperimentJournal(study_dir / "journal.json")
    registry = build_default_task1_baseline_registry()
    task_spec = task1_spec(root)
    selected = models or ["fno_ensemble", "unet1d", "deeponet_lite", "residual_refiner"]
    results: list[dict[str, Any]] = []
    validation_predictions: dict[str, np.ndarray] = {}

    for model in selected:
        if model == "fake":
            workflow = FakeTask1BaselineWorkflow(
                BaselineSpec(name="fake", family="test", trainable=False),
                spec=task_spec,
                run_root=study_dir,
                fill_value=1.0,
            )
            result = workflow.run_validation({"max_samples": max_samples, "steps": steps}, run_name="fake")
            _append_result(
                journal,
                model="fake",
                action_type="baseline_validate",
                success=result.success,
                metrics=dict(result.metrics),
                artifacts={
                    "model": "fake",
                    "run_dir": str(result.run_dir),
                    "prediction_path": str(result.prediction_path) if result.prediction_path else None,
                    "zip_path": str(result.zip_path) if result.zip_path else None,
                },
                error=result.error,
            )
            results.append({"model": "fake", "success": result.success, "metrics": result.metrics, "run_dir": str(result.run_dir)})
            if result.success and result.prediction_path:
                validation_predictions["fake"] = _read_prediction(result.prediction_path)
            continue

        if model not in registry.names():
            results.append(_skip_model(journal, model=model, reason=f"unknown baseline model {model!r}"))
            continue

        if model == "fno_ensemble":
            workflow = FNOBaselineWorkflow(
                build_task1_fno_workflow(
                    project_root=root,
                    study_dir=study_dir,
                    checkpoint_overrides=checkpoint_overrides,
                )
            )
            result = workflow.run_validation({"weights": dict(fno_weights or {})} if fno_weights else {}, run_name="fno_ensemble")
            _append_result(
                journal,
                model=model,
                action_type="baseline_validate",
                success=result.success,
                metrics=dict(result.metrics),
                artifacts={
                    "model": model,
                    "run_dir": str(result.run_dir),
                    "prediction_path": str(result.prediction_path) if result.prediction_path else None,
                    "zip_path": str(result.zip_path) if result.zip_path else None,
                },
                error=result.error,
            )
            results.append({"model": model, "success": result.success, "metrics": result.metrics, "run_dir": str(result.run_dir)})
            if result.success and result.prediction_path:
                validation_predictions[model] = _read_prediction(result.prediction_path)
            continue

        if model == "tfno" and not _neuralop_available():
            results.append(_skip_model(journal, model=model, reason="neuralop is not installed"))
            continue
        if model in PHYSICSNEMO_MODELS:
            status = physicsnemo_status()
            if not status.usable:
                results.append(
                    _skip_model(
                        journal,
                        model=model,
                        reason=f"{status.reason} Recommendation: {status.recommendation}",
                    )
                )
                continue
            results.append(
                _skip_model(
                    journal,
                    model=model,
                    reason=(
                        "PhysicsNeMo is available, but this repository currently exposes it as a gated research candidate only; "
                        "implement a Task 1 runner before enabling training."
                    ),
                )
            )
            continue
        if model in PROTOTYPE_TRAINABLE_MODELS and not _torch_available():
            results.append(_skip_model(journal, model=model, reason="torch is not installed in the active Python environment"))
            continue

        try:
            result = train_task1_baseline(
                model_name=model,
                run_dir=study_dir / model,
                project_root=root,
                max_samples=max_samples,
                steps=steps,
                batch_size=batch_size,
                lr=lr,
                hidden=hidden,
                device=device,
                loss_start_step=loss_start_step,
                loss_end_step=loss_end_step,
                base_train_hdf5=base_train_hdf5,
                base_validation_prediction_path=base_validation_prediction_path,
                initial_loss_weight=initial_loss_weight,
                spectral_loss_weight=spectral_loss_weight,
                spectral_high_weight=spectral_high_weight,
                physics_loss_weight=physics_loss_weight,
                physics_nu=physics_nu,
                physics_dt=physics_dt,
                physics_dx=physics_dx,
            )
            _append_result(
                journal,
                model=model,
                action_type="baseline_train",
                success=result.success,
                metrics=dict(result.metrics),
                artifacts={
                    "model": model,
                    "run_dir": str(result.run_dir),
                    "prediction_path": str(result.prediction_path) if result.prediction_path else None,
                    "zip_path": str(result.zip_path) if result.zip_path else None,
                    "command": result.command,
                },
                error=result.error,
            )
            results.append({"model": model, "success": result.success, "metrics": result.metrics, "run_dir": str(result.run_dir)})
            if result.success and result.prediction_path:
                validation_predictions[model] = _read_prediction(result.prediction_path)
        except Exception as exc:
            _append_result(
                journal,
                model=model,
                action_type="baseline_train",
                success=False,
                artifacts={"model": model},
                error=f"{type(exc).__name__}: {exc}",
            )
            results.append({"model": model, "success": False, "error": f"{type(exc).__name__}: {exc}"})

    ensemble_results: list[dict[str, Any]] = []
    if len(validation_predictions) >= 2 and task_spec.validation_target_path is not None:
        target = _read_prediction(task_spec.validation_target_path)
        ensemble_results = run_validation_ensembles(
            study_dir=study_dir,
            target=target,
            predictions=validation_predictions,
        )
        for item in ensemble_results:
            _append_result(
                journal,
                model=str(item["name"]),
                action_type="baseline_ensemble",
                success=bool(item["success"]),
                metrics=dict(item["metrics"]),
                artifacts={
                    "model": item["name"],
                    "run_dir": item["run_dir"],
                    "prediction_path": item["prediction_path"],
                },
            )
        results.extend({"model": item["name"], **item} for item in ensemble_results)

    export_experiment_records(
        journal,
        study_dir=study_dir,
        study_name=study_name,
        runs_root=runs_root,
        metric="competition_score_proxy",
        maximize=True,
    )
    summary = {
        "version": 1,
        "updated_at": utc_now_iso(),
        "study_name": study_name,
        "max_samples": max_samples,
        "steps": steps,
        "train_config": {
            "max_samples": max_samples,
            "steps": steps,
            "batch_size": batch_size,
            "lr": lr,
            "hidden": hidden,
            "device": device,
            "loss_start_step": loss_start_step,
            "loss_end_step": loss_end_step,
            "base_train_hdf5": [str(path) for path in (base_train_hdf5 or [])],
            "base_validation_prediction_path": str(base_validation_prediction_path) if base_validation_prediction_path else None,
            "initial_loss_weight": initial_loss_weight,
            "spectral_loss_weight": spectral_loss_weight,
            "spectral_high_weight": spectral_high_weight,
            "physics_loss_weight": physics_loss_weight,
            "physics_nu": physics_nu,
            "physics_dt": physics_dt,
            "physics_dx": physics_dx,
        },
        "checkpoint_overrides": {key: str(value) for key, value in (checkpoint_overrides or {}).items()},
        "fno_weights": dict(fno_weights or {}),
        "results": results,
        "ensemble_results": ensemble_results,
    }
    summary_path = study_dir / "baseline_zoo_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 1 Baseline Zoo prototype experiments.")
    parser.add_argument("--study-name", default="task1-zoo-prototype")
    parser.add_argument("--models", default="fno_ensemble,unet1d,deeponet_lite,residual_refiner")
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--loss-start-step", type=int, default=10)
    parser.add_argument("--loss-end-step", type=int, default=None)
    parser.add_argument("--checkpoint-override", action="append", default=None, help="Override official Task 1 checkpoint path, e.g. nu0.001=runs/finetune/best.pt")
    parser.add_argument("--fno-weight", action="append", default=None, help="Override official checkpoint ensemble weight, e.g. unet_pf20_nu0.001=0.25")
    parser.add_argument("--base-train-hdf5", action="append", default=None, help="Base prediction HDF5 aligned with each train HDF5, used by residual_refiner.")
    parser.add_argument("--base-validation-prediction-path", default=None, help="Base validation prediction HDF5 used by residual_refiner.")
    parser.add_argument("--initial-loss-weight", type=float, default=0.05)
    parser.add_argument("--spectral-loss-weight", type=float, default=0.0)
    parser.add_argument("--spectral-high-weight", type=float, default=2.0)
    parser.add_argument("--physics-loss-weight", type=float, default=0.0)
    parser.add_argument("--physics-nu", type=float, default=0.001)
    parser.add_argument("--physics-dt", type=float, default=0.05)
    parser.add_argument("--physics-dx", type=float, default=1.0 / 256.0)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    path = run_task1_baseline_zoo(
        project_root=args.project_root,
        study_name=args.study_name,
        models=models,
        max_samples=args.max_samples,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        device=args.device,
        loss_start_step=args.loss_start_step,
        loss_end_step=args.loss_end_step,
        checkpoint_overrides=parse_key_value_paths(args.checkpoint_override),
        fno_weights=parse_key_value_floats(args.fno_weight),
        base_train_hdf5=[Path(path) for path in (args.base_train_hdf5 or [])],
        base_validation_prediction_path=Path(args.base_validation_prediction_path) if args.base_validation_prediction_path else None,
        initial_loss_weight=args.initial_loss_weight,
        spectral_loss_weight=args.spectral_loss_weight,
        spectral_high_weight=args.spectral_high_weight,
        physics_loss_weight=args.physics_loss_weight,
        physics_nu=args.physics_nu,
        physics_dt=args.physics_dt,
        physics_dx=args.physics_dx,
    )
    print(path)


if __name__ == "__main__":
    main()
