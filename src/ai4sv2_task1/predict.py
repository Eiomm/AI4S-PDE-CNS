from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import default_input_for_split, default_run_label
from .hdf5_io import read_task1_input, write_prediction
from .logging import JsonlLogger
from .metrics import compute_task1_metrics
from .models.fno import load_fno_checkpoint, rollout_fno
from .models.unet_pf import load_unet_pf_checkpoint, rollout_unet_pf
from .paths import checkpoint_path, resolve_path, runs_root, task_root
from .validate import validate_task1_prediction

TASK1_SEGMENTS: tuple[tuple[int, int], ...] = ((10, 57), (57, 105), (105, 200))
SAFE_RUN_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def timestamp_prefix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(name: str | None) -> torch.device:
    if not name or name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def combine_predictions(predictions: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    if not predictions:
        raise ValueError("at least one prediction is required")
    if weights is None:
        return np.mean(predictions, axis=0).astype(np.float32)
    if len(weights) != len(predictions):
        raise ValueError("weights length must match predictions length")
    weights_array = np.asarray(weights, dtype=np.float64)
    total = float(weights_array.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    weights_array = weights_array / total
    combined = np.zeros_like(predictions[0], dtype=np.float64)
    for weight, prediction in zip(weights_array, predictions):
        combined += float(weight) * prediction
    return combined.astype(np.float32)


def apply_segment_postprocess(
    combined: np.ndarray,
    *,
    initial: np.ndarray,
    predictions_by_kind: dict[str, np.ndarray],
    segment_fno_weights: list[float] | None,
    persistence_segment_alpha: list[float] | None,
) -> np.ndarray:
    result = combined.astype(np.float32, copy=True)
    if segment_fno_weights is not None:
        if len(segment_fno_weights) != len(TASK1_SEGMENTS):
            raise ValueError("segment_fno_weights must have exactly 3 values")
        if "fno" not in predictions_by_kind or "unet_pf20" not in predictions_by_kind:
            raise ValueError("segment_fno_weights requires both fno and unet_pf20 predictions")
        result = predictions_by_kind["unet_pf20"].copy()
        for (start, end), fno_weight in zip(TASK1_SEGMENTS, segment_fno_weights):
            fno_weight = float(fno_weight)
            result[:, start:end, :] = (
                fno_weight * predictions_by_kind["fno"][:, start:end, :]
                + (1.0 - fno_weight) * predictions_by_kind["unet_pf20"][:, start:end, :]
            )
    if persistence_segment_alpha is not None:
        if len(persistence_segment_alpha) != len(TASK1_SEGMENTS):
            raise ValueError("persistence_segment_alpha must have exactly 3 values")
        persistence = np.zeros_like(result, dtype=np.float32)
        persistence[:, :10, :] = initial
        persistence[:, 10:, :] = initial[:, -1:, :]
        for (start, end), alpha in zip(TASK1_SEGMENTS, persistence_segment_alpha):
            alpha = float(alpha)
            result[:, start:end, :] = alpha * result[:, start:end, :] + (1.0 - alpha) * persistence[:, start:end, :]
    result[:, :10, :] = initial
    return result.astype(np.float32)


def _model_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    specs = config.get("models")
    if specs is None:
        route = str(config.get("route", "official_fno"))
        if route == "official_fno":
            specs = [{"kind": "fno", "checkpoint": "checkpoints/official/nu0.001_fno.pt", "weight": 1.0}]
        elif route == "official_unet_pf20":
            specs = [{"kind": "unet_pf20", "checkpoint": "checkpoints/official/nu0.001_unet_pf20.pt", "weight": 1.0}]
        elif route == "official_ensemble":
            specs = [
                {"kind": "fno", "checkpoint": "checkpoints/official/nu0.001_fno.pt", "weight": 0.12},
                {"kind": "unet_pf20", "checkpoint": "checkpoints/official/nu0.001_unet_pf20.pt", "weight": 0.88},
            ]
        else:
            raise ValueError(f"route {route!r} requires explicit models")
    if not isinstance(specs, list) or not specs:
        raise ValueError("models must be a non-empty list")
    return specs


def _resolve_checkpoint(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if len(path.parts) == 1:
        return checkpoint_path(path.as_posix())
    return resolve_path(path)


def run_prediction(config: dict[str, Any], *, split: str = "test", run_name: str | None = None, limit: int | None = None) -> dict[str, Any]:
    requested_run_name = run_name
    label = run_name if run_name and SAFE_RUN_LABEL.fullmatch(run_name) else timestamp_prefix()
    run_dir = runs_root() / label
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "task1_logs.log"
    logger = JsonlLogger(log_path)
    prediction_path = run_dir / "task1_pred.hdf5"
    time_path = run_dir / "task1_time.csv"
    metadata_path = run_dir / "metadata.json"
    metrics_path = run_dir / "metrics.json"

    if split == "val":
        input_path = resolve_path(config.get("val_input_path") or default_input_for_split(split))
    else:
        input_path = resolve_path(config.get("input_path") or default_input_for_split(split))
    full_t_path = resolve_path(config.get("full_t_path", "data/task1_val.hdf5"))
    batch_size = int(config.get("batch_size", 50))
    device = resolve_device(config.get("device", "auto"))
    started = time.perf_counter()
    logger.write(
        {
            "action": "task1_prediction_start",
            "split": split,
            "run_dir": str(run_dir),
            "input_path": str(input_path),
            "device": str(device),
            "batch_size": batch_size,
        }
    )

    initial, x_coords, t_coords, target = read_task1_input(input_path, full_t_path=full_t_path)
    if limit is not None:
        initial = initial[:limit]
        target = target[:limit] if target is not None else None
    predictions: list[np.ndarray] = []
    predictions_by_kind: dict[str, np.ndarray] = {}
    checkpoint_records: list[dict[str, Any]] = []

    for spec in _model_specs(config):
        kind = str(spec["kind"])
        checkpoint = _resolve_checkpoint(str(spec["checkpoint"]))
        model_started = time.perf_counter()
        logger.write({"action": "load_model", "kind": kind, "checkpoint": str(checkpoint)})
        if kind == "fno":
            model = load_fno_checkpoint(checkpoint, device)
            prediction = rollout_fno(model, initial, x_coords, t_coords, device, batch_size)
        elif kind == "unet_pf20":
            model = load_unet_pf_checkpoint(checkpoint, device)
            prediction = rollout_unet_pf(model, initial, t_coords, device, batch_size)
        else:
            raise ValueError(f"unsupported model kind: {kind}")
        predictions.append(prediction.astype(np.float32))
        predictions_by_kind[kind] = prediction.astype(np.float32)
        checkpoint_records.append(
            {
                "kind": kind,
                "path": str(checkpoint),
                "sha256": file_sha256(checkpoint),
                "weight": float(spec.get("weight", 1.0)),
                "seconds": float(time.perf_counter() - model_started),
            }
        )

    weights = [float(spec.get("weight", 1.0)) for spec in _model_specs(config)]
    combined = combine_predictions(predictions, weights)
    combined = apply_segment_postprocess(
        combined,
        initial=initial,
        predictions_by_kind=predictions_by_kind,
        segment_fno_weights=config.get("segment_fno_weights"),
        persistence_segment_alpha=config.get("persistence_segment_alpha"),
    )
    write_prediction(prediction_path, combined, dataset_key="tensor")
    inference_time = time.perf_counter() - started
    with time_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["train_time", "inference_time"])
        writer.writeheader()
        writer.writerow({"train_time": f"{float(config.get('train_time', 0.0)):.6f}", "inference_time": f"{inference_time:.6f}"})

    validation = validate_task1_prediction(prediction_path, input_path, allow_prefix_samples=limit is not None)
    metrics = compute_task1_metrics(combined, target) if target is not None else None
    if metrics is not None:
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "task": "task1",
        "route": config.get("route"),
        "split": split,
        "run_name": label,
        "requested_run_name": requested_run_name,
        "default_run_label": default_run_label(config, split),
        "run_dir": str(run_dir),
        "config_path": config.get("_config_path"),
        "input_path": str(input_path),
        "prediction_path": str(prediction_path),
        "time_path": str(time_path),
        "log_path": str(log_path),
        "metrics_path": str(metrics_path) if metrics is not None else None,
        "checkpoints": checkpoint_records,
        "batch_size": batch_size,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "inference_time": inference_time,
        "validation": validation,
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.write(
        {
            "action": "task1_prediction_done",
            "prediction_path": str(prediction_path),
            "time_path": str(time_path),
            "metadata_path": str(metadata_path),
            "metrics_path": str(metrics_path) if metrics is not None else None,
            "validation": validation,
            "metrics": metrics,
        }
    )
    return metadata
