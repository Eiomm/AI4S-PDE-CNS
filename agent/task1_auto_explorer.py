from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from .logging import utc_now_iso
from .pde_baselines import BaselineSpec, write_baseline_artifacts
from .pde_journal import CandidatePlan, ExperimentJournal
from .pde_results import RunResult
from .run_task1_combo_search import collect_study_predictions
from .task1_combo_space import ComboCandidate, ComboSearchConfig, search_task1_combinations


def run_task1_auto_explorer(
    *,
    study_dir: str | Path,
    output_dir: str | Path,
    target_path: str | Path = "data/Task1/task1_val.hdf5",
    config: ComboSearchConfig | None = None,
    base_train_hdf5: list[str | Path] | None = None,
    execute_ready: bool = False,
    project_root: str | Path = ".",
    command_runner: Callable[..., dict[str, Any]] | None = None,
) -> Path:
    study_path = Path(study_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target = _read_prediction(target_path)
    predictions = collect_study_predictions(study_path, include_ensembles=False)
    if len(predictions) < 2:
        raise ValueError(f"need at least two base validation predictions in {study_path}")

    combo_config = config or ComboSearchConfig()
    ranked = search_task1_combinations(predictions=predictions, target=target, config=combo_config)
    journal = ExperimentJournal(output_path / "journal.json")
    result_summaries = []
    for candidate in ranked:
        result_summaries.append(_record_candidate(output_path, journal, candidate))

    best = result_summaries[0]
    summary = {
        "version": 1,
        "updated_at": utc_now_iso(),
        "study_dir": str(study_path),
        "output_dir": str(output_path),
        "target_path": str(target_path),
        "input_models": sorted(predictions),
        "combo_config": _combo_config_summary(combo_config),
        "best": best,
        "ranked_results": result_summaries,
        "training_queue": propose_training_queue(
            best=best,
            input_models=sorted(predictions),
            base_validation_prediction_path=study_path / "fno_ensemble" / "task1_val_pred.hdf5",
            base_train_hdf5=base_train_hdf5,
        ),
    }
    executed_training: list[dict[str, Any]] = []
    if execute_ready:
        executed_training = _execute_ready_training(
            summary["training_queue"],
            predictions=predictions,
            target=target,
            combo_config=combo_config,
            project_root=Path(project_root),
            command_runner=command_runner,
        )
        summary["executed_training"] = executed_training
        successful = {
            item["model"]: _read_prediction(item["prediction_path"])
            for item in executed_training
            if item.get("success") and item.get("prediction_path")
        }
        if successful:
            expanded_predictions = {**predictions, **successful}
            post_ranked = search_task1_combinations(predictions=expanded_predictions, target=target, config=combo_config)
            post_summaries = [_record_candidate(output_path / "post_training", journal, candidate) for candidate in post_ranked]
            summary["post_training_best"] = post_summaries[0]
            summary["post_training_ranked_results"] = post_summaries
    else:
        summary["executed_training"] = []
    summary_path = output_path / "auto_explorer_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_summary_markdown(output_path / "auto_explorer_summary.md", summary)
    return summary_path


def propose_training_queue(
    *,
    best: dict[str, Any],
    input_models: list[str],
    base_validation_prediction_path: str | Path | None = None,
    base_train_hdf5: list[str | Path] | None = None,
) -> list[dict[str, Any]]:
    best_kind = str(best.get("kind", ""))
    queue = [
        _training_plan(
            name="tail-deeponet-lite-seed-refresh",
            model="deeponet_lite",
            hypothesis="Train another DeepONetLite long-horizon expert so the combination search has a non-identical tail candidate.",
            params={
                "max_samples": 12000,
                "steps": 8000,
                "batch_size": 4,
                "lr": 3.0e-4,
                "hidden": 96,
                "device": "cuda",
                "loss_start_step": 120,
                "loss_end_step": 200,
                "seed": 43,
            },
            policy={
                "priority": 0.9,
                "cost": "medium",
                "status": "requires_trainer_knob",
                "tags": ["stronger-backbone", "long-horizon", "tail-window", "multi-seed"],
                "missing_knobs": ["seed"],
            },
        ),
        _training_plan(
            name="residual-refiner-long-horizon",
            model="residual_refiner",
            hypothesis="Train a residual correction model over the current FNO trajectory with long-horizon loss emphasis.",
            params={
                "max_samples": 12000,
                "steps": 6000,
                "batch_size": 4,
                "lr": 2.0e-4,
                "hidden": 128,
                "device": "cuda",
                "loss_start_step": 105,
                "loss_end_step": 200,
                "base_train_hdf5": [str(path) for path in (base_train_hdf5 or [])],
                "base_validation_prediction_path": str(base_validation_prediction_path) if base_validation_prediction_path else None,
                "seed": 44,
            },
            policy=_residual_refiner_policy(base_train_hdf5),
        ),
        _training_plan(
            name="pino-fno-spectral-physics",
            model="pino_fno",
            hypothesis="Evaluate a PINO-style branch with spectral and initial-consistency losses before adding heavier PhysicsNeMo backbones.",
            params={
                "max_samples": 16000,
                "steps": 6000,
                "batch_size": 4,
                "lr": 2.0e-4,
                "hidden": 128,
                "device": "cuda",
                "loss_start_step": 10,
                "loss_end_step": 200,
                "seed": 45,
            },
            policy={
                "priority": 0.75,
                "cost": "medium",
                "status": "requires_trainer_knob",
                "tags": ["stronger-backbone", "physics-loss", "spectral-loss", "long-horizon"],
                "desired_trainer_params": {"spectral_loss_weight": 0.02, "physics_residual_weight": 0.01},
            },
        ),
    ]
    if "physicsnemo_fno" not in input_models:
        queue.append(
            _training_plan(
                name="physicsnemo-fno-gated-probe",
                model="physicsnemo_fno",
                hypothesis="Probe the optional PhysicsNeMo FNO branch only when the environment gate reports it is usable.",
                params={
                    "max_samples": 4096,
                    "steps": 1000,
                    "batch_size": 2,
                    "lr": 1.0e-4,
                    "hidden": 64,
                    "device": "cuda",
                    "optional_dependency": "physicsnemo",
                },
                policy={
                    "priority": 0.55,
                    "cost": "heavy",
                    "status": "requires_dependency",
                    "tags": ["stronger-backbone", "physicsnemo", "optional-dependency"],
                    "dependency": "physicsnemo",
                },
            )
        )
    if best_kind == "single_model":
        queue.insert(
            0,
            _training_plan(
                name="add-tail-expert-after-single-best",
                model="deeponet_lite",
                hypothesis="The current pool is single-model dominated; add a targeted tail expert to create useful ensemble diversity.",
                params={
                    "max_samples": 12000,
                    "steps": 8000,
                    "batch_size": 4,
                    "lr": 3.0e-4,
                    "hidden": 96,
                    "device": "cuda",
                    "loss_start_step": 120,
                    "loss_end_step": 200,
                    "seed": 46,
                },
                policy={
                    "priority": 0.88,
                    "cost": "medium",
                    "status": "requires_trainer_knob",
                    "tags": ["stronger-backbone", "long-horizon", "tail-window", "multi-seed"],
                    "missing_knobs": ["seed"],
                },
            ),
        )
    return queue


def _residual_refiner_policy(base_train_hdf5: list[str | Path] | None) -> dict[str, Any]:
    if base_train_hdf5:
        return {
            "priority": 0.85,
            "cost": "medium",
            "status": "ready",
            "tags": ["refiner", "long-horizon", "multi-step-rollout"],
        }
    return {
        "priority": 0.85,
        "cost": "medium",
        "status": "requires_base_train_predictions",
        "tags": ["refiner", "long-horizon", "multi-step-rollout"],
        "missing_artifacts": ["base_train_hdf5"],
    }


def _record_candidate(output_dir: Path, journal: ExperimentJournal, candidate: ComboCandidate) -> dict[str, Any]:
    run_dir = output_dir / _safe_name(candidate.name)
    prediction_path = _write_prediction(run_dir / "task1_val_pred.hdf5", candidate.prediction)
    (run_dir / "metrics.json").write_text(json.dumps(candidate.metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = RunResult(
        task_id="task1",
        run_dir=run_dir,
        metrics=dict(candidate.metrics),
        prediction_path=prediction_path,
        zip_path=None,
        train_time=0.0,
        inference_time=0.0,
        success=True,
        command=["auto_explorer", candidate.kind],
    )
    write_baseline_artifacts(
        run_dir,
        BaselineSpec(name=candidate.name, family="auto_combo", trainable=False),
        candidate.config,
        result,
        conclusion=f"auto combo competition_score_proxy={candidate.metrics.get('competition_score_proxy')}",
    )
    node = journal.append_plan(
        CandidatePlan(
            intent="combine",
            hypothesis=f"Evaluate automatically generated combination {candidate.name}.",
            action_type="baseline_ensemble",
            params={"name": candidate.name, **candidate.config},
            expected_effect="Score a generated validation combination and add it to the ranked pool.",
            risk="Validation-only combination can overfit if promoted without hidden-score confirmation.",
        )
    )
    journal.mark_running(node.id)
    updated = journal.update_result(
        node.id,
        success=True,
        metrics=dict(candidate.metrics),
        artifacts={
            "run_dir": str(run_dir),
            "prediction_path": str(prediction_path),
            "candidate_config": dict(candidate.config),
        },
    )
    summary = candidate.to_summary()
    summary.update(
        {
            "node_id": updated.id,
            "run_dir": str(run_dir),
            "prediction_path": str(prediction_path),
        }
    )
    return summary


def _training_plan(
    *,
    name: str,
    model: str,
    hypothesis: str,
    params: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(params)
    payload["name"] = name
    payload["model"] = model
    payload["policy"] = dict(policy)
    payload["command"] = _baseline_zoo_command(name=name, model=model, params=payload)
    payload["metrics_path"] = str(Path("runs") / name / model / "metrics.json")
    payload["timeout_seconds"] = _timeout_for_policy(policy)
    return {
        "name": name,
        "intent": "improve",
        "hypothesis": hypothesis,
        "action_type": "baseline_train",
        "params": payload,
        "expected_effect": "Generate a new validation prediction for the automatic combination pool.",
        "risk": "GPU training can be slow or underperform; keep current FNO fallback active.",
    }


def _baseline_zoo_command(*, name: str, model: str, params: dict[str, Any]) -> list[str]:
    selected_models = "fno_ensemble" if model == "fno_ensemble" else f"fno_ensemble,{model}"
    command = [
        "python",
        "-m",
        "agent.run_task1_baseline_zoo",
        "--study-name",
        name,
        "--models",
        selected_models,
        "--max-samples",
        str(int(params.get("max_samples", 1024))),
        "--steps",
        str(int(params.get("steps", 200))),
        "--batch-size",
        str(int(params.get("batch_size", 4))),
        "--lr",
        str(float(params.get("lr", 1.0e-3))),
        "--hidden",
        str(int(params.get("hidden", 64))),
        "--loss-start-step",
        str(int(params.get("loss_start_step", 10))),
        "--loss-end-step",
        str(int(params.get("loss_end_step", 200))),
    ]
    if params.get("device"):
        command.extend(["--device", str(params["device"])])
    if params.get("base_validation_prediction_path"):
        command.extend(["--base-validation-prediction-path", str(params["base_validation_prediction_path"])])
    for base_train_path in params.get("base_train_hdf5", []) or []:
        command.extend(["--base-train-hdf5", str(base_train_path)])
    return command


def _timeout_for_policy(policy: dict[str, Any]) -> int:
    cost = str(policy.get("cost", "medium"))
    if cost == "light":
        return 1800
    if cost == "heavy":
        return 14400
    return 7200


def _execute_ready_training(
    queue: list[dict[str, Any]],
    *,
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    combo_config: ComboSearchConfig,
    project_root: Path,
    command_runner: Callable[..., dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    del predictions, combo_config
    runner = command_runner or _default_command_runner
    executed: list[dict[str, Any]] = []
    for item in queue:
        params = dict(item.get("params") or {})
        policy = dict(params.get("policy") or {})
        if policy.get("status") != "ready":
            continue
        command = [str(part) for part in (params.get("command") or [])]
        timeout = int(params.get("timeout_seconds") or _timeout_for_policy(policy))
        model = str(params.get("model") or item.get("name"))
        metrics_path = project_root / Path(str(params.get("metrics_path", "")))
        prediction_path = metrics_path.parent / "task1_val_pred.hdf5"
        result: dict[str, Any] = {
            "name": item.get("name"),
            "model": model,
            "command": command,
            "timeout_seconds": timeout,
            "metrics_path": str(metrics_path),
            "prediction_path": str(prediction_path),
            "success": False,
        }
        try:
            command_result = runner(command, cwd=project_root, timeout=timeout)
        except Exception as exc:
            result["command_result"] = {"returncode": 1, "error": f"{type(exc).__name__}: {exc}"}
            result["error"] = result["command_result"]["error"]
            executed.append(result)
            continue

        result["command_result"] = command_result
        if int(command_result.get("returncode", 1)) != 0:
            result["error"] = command_result.get("stderr_tail") or command_result.get("error") or "training command failed"
            executed.append(result)
            continue
        if not prediction_path.exists():
            result["error"] = f"missing prediction artifact: {prediction_path}"
            executed.append(result)
            continue
        try:
            prediction = _read_prediction(prediction_path)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: failed to read prediction artifact: {exc}"
            executed.append(result)
            continue
        if prediction.shape != target.shape:
            result["error"] = f"prediction shape {prediction.shape} does not match target shape {target.shape}"
            executed.append(result)
            continue
        result["success"] = True
        executed.append(result)
    return executed


def _default_command_runner(command: list[str], *, cwd: str | Path, timeout: int) -> dict[str, Any]:
    executable_command = list(command)
    if executable_command and executable_command[0].lower() == "python":
        executable_command[0] = sys.executable
    try:
        completed = subprocess.run(
            executable_command,
            cwd=str(cwd),
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "command": executable_command,
            "timeout": timeout,
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "error": f"timed out after {timeout} seconds",
        }
    return {
        "returncode": completed.returncode,
        "command": executable_command,
        "timeout": timeout,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


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


def _combo_config_summary(config: ComboSearchConfig) -> dict[str, Any]:
    return {
        "base_name": config.base_name,
        "include_single_models": config.include_single_models,
        "include_global": config.include_global,
        "include_cluster": config.include_cluster,
        "include_temporal": config.include_temporal,
        "include_piecewise": config.include_piecewise,
        "include_cross_piecewise": config.include_cross_piecewise,
        "grid_step": config.grid_step,
        "temporal_cut_min": config.temporal_cut_min,
        "temporal_cut_max": config.temporal_cut_max,
        "temporal_cut_stride": config.temporal_cut_stride,
        "temporal_weight_step": config.temporal_weight_step,
        "piecewise_split_candidates": list(config.piecewise_split_candidates),
        "top_k": config.top_k,
    }


def _write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Task 1 Auto Explorer",
        "",
        f"- Updated: {summary['updated_at']}",
        f"- Study: {summary['study_dir']}",
        f"- Output: {summary['output_dir']}",
        f"- Target: {summary['target_path']}",
        "",
        "| Rank | Name | Kind | Proxy | MSE | Forecast MSE | Long MSE | Segment3 RMSE |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, item in enumerate(summary["ranked_results"], start=1):
        metrics = item.get("metrics", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(item["name"]),
                    str(item["kind"]),
                    _format_float(metrics.get("competition_score_proxy")),
                    _format_float(metrics.get("mse")),
                    _format_float(metrics.get("forecast_mse")),
                    _format_float(metrics.get("long_horizon_mse")),
                    _format_float(metrics.get("segment3_rmse")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Training Queue", ""])
    for item in summary["training_queue"]:
        lines.append(f"- `{item['name']}`: {item['hypothesis']}")
    if summary.get("executed_training"):
        lines.extend(["", "## Executed Training", ""])
        for item in summary["executed_training"]:
            status = "success" if item.get("success") else "failed"
            lines.append(f"- `{item['name']}`: {status}")
    if summary.get("post_training_best"):
        best = summary["post_training_best"]
        metrics = best.get("metrics", {})
        lines.extend(
            [
                "",
                "## Post Training Best",
                "",
                f"- `{best['name']}` ({best['kind']}): proxy={_format_float(metrics.get('competition_score_proxy'))}, mse={_format_float(metrics.get('mse'))}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.10g}"


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automatic Task 1 validation combination exploration.")
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="data/Task1/task1_val.hdf5")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--temporal-cut-min", type=int, default=105)
    parser.add_argument("--temporal-cut-max", type=int, default=199)
    parser.add_argument("--temporal-cut-stride", type=int, default=1)
    parser.add_argument("--temporal-weight-step", type=float, default=0.01)
    parser.add_argument("--no-global", action="store_true")
    parser.add_argument("--no-cluster", action="store_true")
    parser.add_argument("--no-piecewise", action="store_true")
    parser.add_argument("--base-train-hdf5", action="append", default=None)
    parser.add_argument("--execute-ready", action="store_true")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    config = ComboSearchConfig(
        top_k=args.top_k,
        temporal_cut_min=args.temporal_cut_min,
        temporal_cut_max=args.temporal_cut_max,
        temporal_cut_stride=args.temporal_cut_stride,
        temporal_weight_step=args.temporal_weight_step,
        include_global=not args.no_global,
        include_cluster=not args.no_cluster,
        include_piecewise=not args.no_piecewise,
    )
    path = run_task1_auto_explorer(
        study_dir=args.study_dir,
        output_dir=args.output_dir,
        target_path=args.target,
        config=config,
        base_train_hdf5=[Path(path) for path in (args.base_train_hdf5 or [])],
        execute_ready=args.execute_ready,
        project_root=args.project_root,
    )
    print(path)


if __name__ == "__main__":
    main()
