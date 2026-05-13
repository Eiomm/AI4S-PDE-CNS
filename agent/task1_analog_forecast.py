from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .logging import utc_now_iso
from .pde_baselines import BaselineSpec, write_baseline_artifacts
from .pde_finetune_data import spatial_indices
from .pde_metrics import compute_task1_metrics
from .pde_results import RunResult
from .task1_trajectory_data import parse_nu_from_path


@dataclass(frozen=True)
class AnalogSearchConfig:
    top_k: int = 3
    max_candidates_per_file: int = 4000
    spatial_size: int = 256
    initial_step: int = 10
    output_steps: int = 200
    chunk_size: int = 128
    feature_mode: str = "initial_gradient"
    weight_temperature: float = 1.0e-6
    normalize_per_sample: bool = False


def _default_raw_paths(project_root: Path) -> list[Path]:
    raw = project_root / "data" / "pdebench_burgers" / "raw"
    return sorted(raw.glob("1D_Burgers_Sols_Nu*.hdf5"))


def _read_tensor(path: str | Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if "tensor" in h5:
            return h5["tensor"][:]
        if "prediction" in h5:
            return h5["prediction"][:]
        return h5[next(iter(h5.keys()))][:]


def _read_coords(path: str | Path, spatial_size: int, output_steps: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        if "tensor" in h5:
            source_size = h5["tensor"].shape[2]
        else:
            source_size = spatial_size
        indices = spatial_indices(source_size=source_size, target_size=spatial_size)
        if "x-coordinate" in h5:
            x = h5["x-coordinate"][indices]
        else:
            x = np.linspace(0.0, 1.0, spatial_size, endpoint=False, dtype=np.float32)
        if "t-coordinate" in h5:
            t = h5["t-coordinate"][:output_steps]
        else:
            t = np.linspace(0.0, 1.0, output_steps, dtype=np.float32)
    return np.asarray(x, dtype=np.float32), np.asarray(t, dtype=np.float32)


def _write_prediction(path: Path, prediction: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with h5py.File(path, "w") as h5:
        h5.create_dataset("prediction", data=np.asarray(prediction, dtype=np.float32))
    return path


def extract_analog_features(initial: np.ndarray, mode: str, *, normalize_per_sample: bool = False) -> np.ndarray:
    values = np.asarray(initial, dtype=np.float32)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3:
        raise ValueError(f"initial must have shape (N,T,X) or (T,X), got {values.shape}")
    if mode == "initial":
        features = values.reshape(values.shape[0], -1)
    elif mode == "initial_gradient":
        gradients = np.diff(values, axis=2, append=values[:, :, :1])
        features = np.concatenate([values.reshape(values.shape[0], -1), gradients.reshape(values.shape[0], -1)], axis=1)
    else:
        raise ValueError(f"unknown feature mode: {mode}")
    if normalize_per_sample:
        mean = features.mean(axis=1, keepdims=True)
        std = features.std(axis=1, keepdims=True) + 1.0e-6
        features = (features - mean) / std
    return features.astype(np.float32)


def _update_topk(
    *,
    query_features: np.ndarray,
    candidate_features: np.ndarray,
    records: list[dict[str, Any]],
    best_distances: np.ndarray,
    best_records: list[list[dict[str, Any]]],
    top_k: int,
) -> None:
    distances = np.mean((query_features[:, None, :] - candidate_features[None, :, :]) ** 2, axis=2)
    for query_idx in range(query_features.shape[0]):
        merged: list[tuple[float, dict[str, Any]]] = [
            (float(best_distances[query_idx, slot]), best_records[query_idx][slot])
            for slot in range(top_k)
            if math.isfinite(float(best_distances[query_idx, slot]))
        ]
        merged.extend((float(distances[query_idx, idx]), records[idx]) for idx in range(len(records)))
        merged.sort(key=lambda item: item[0])
        for slot, (distance, record) in enumerate(merged[:top_k]):
            best_distances[query_idx, slot] = distance
            best_records[query_idx][slot] = record


def _read_candidate_record(record: dict[str, Any], config: AnalogSearchConfig) -> np.ndarray:
    path = Path(record["source_path"])
    with h5py.File(path, "r") as h5:
        tensor = h5["tensor"]
        indices = spatial_indices(source_size=tensor.shape[2], target_size=config.spatial_size)
        data = tensor[int(record["sample_index"]), : config.output_steps, indices]
    return np.asarray(data, dtype=np.float32)


def _blend_neighbors(best_distances: np.ndarray, best_records: list[list[dict[str, Any]]], target: np.ndarray, config: AnalogSearchConfig) -> np.ndarray:
    output = np.zeros((target.shape[0], config.output_steps, config.spatial_size), dtype=np.float32)
    for query_idx, records in enumerate(best_records):
        distances = np.asarray(best_distances[query_idx, : len(records)], dtype=np.float64)
        if len(records) == 1 or np.min(distances) <= 1.0e-12:
            weights = np.zeros(len(records), dtype=np.float64)
            weights[int(np.argmin(distances))] = 1.0
        else:
            scale = max(float(np.median(distances)), config.weight_temperature)
            logits = -distances / scale
            logits -= np.max(logits)
            weights = np.exp(logits)
            weights /= np.sum(weights)
        pred = np.zeros((config.output_steps, config.spatial_size), dtype=np.float32)
        for weight, record in zip(weights, records):
            pred += float(weight) * _read_candidate_record(record, config)
        pred[: config.initial_step] = target[query_idx, : config.initial_step]
        output[query_idx] = pred
    return output


def estimate_burgers_nu(initial: np.ndarray, x_coords: np.ndarray, t_coords: np.ndarray) -> float:
    values = np.asarray(initial, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("initial must have shape (T, X)")
    x = np.asarray(x_coords, dtype=np.float64)
    t = np.asarray(t_coords, dtype=np.float64)
    dx = float(np.mean(np.diff(x))) if len(x) > 1 else 1.0
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0
    u = values[1:-1]
    u_t = (values[2:] - values[:-2]) / (2.0 * dt)
    u_x = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dx)
    u_xx = (np.roll(u, -1, axis=1) - 2.0 * u + np.roll(u, 1, axis=1)) / (dx * dx)
    numerator = np.sum((u_t + u * u_x) * u_xx)
    denominator = np.sum(u_xx * u_xx) + 1.0e-12
    return float(numerator / denominator)


def _estimate_query_nus(target: np.ndarray, x_coords: np.ndarray, t_coords: np.ndarray, config: AnalogSearchConfig) -> list[float]:
    t = t_coords[: config.initial_step]
    return [estimate_burgers_nu(target[idx, : config.initial_step], x_coords, t) for idx in range(target.shape[0])]


def run_task1_analog_validation(
    *,
    run_dir: str | Path,
    target_hdf5: str | Path = "data/Task1/task1_val.hdf5",
    raw_hdf5: list[str | Path] | None = None,
    project_root: str | Path = ".",
    config: AnalogSearchConfig | None = None,
) -> Path:
    cfg = config or AnalogSearchConfig()
    root = Path(project_root)
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    raw_paths = [Path(path) for path in (raw_hdf5 or _default_raw_paths(root))]
    if not raw_paths:
        raise FileNotFoundError("no raw PDEBench HDF5 files found")
    target = _read_tensor(target_hdf5).astype(np.float32)
    if target.shape[1] < cfg.output_steps or target.shape[2] != cfg.spatial_size:
        raise ValueError(f"target must have shape (N, >= {cfg.output_steps}, {cfg.spatial_size}), got {target.shape}")
    target = target[:, : cfg.output_steps, :]
    query_features = extract_analog_features(
        target[:, : cfg.initial_step, :],
        cfg.feature_mode,
        normalize_per_sample=cfg.normalize_per_sample,
    )
    best_distances = np.full((target.shape[0], cfg.top_k), np.inf, dtype=np.float64)
    best_records: list[list[dict[str, Any]]] = [[{} for _ in range(cfg.top_k)] for _ in range(target.shape[0])]

    for raw_path in raw_paths:
        with h5py.File(raw_path, "r") as h5:
            tensor = h5["tensor"]
            sample_count = min(int(tensor.shape[0]), int(cfg.max_candidates_per_file))
            indices = spatial_indices(source_size=tensor.shape[2], target_size=cfg.spatial_size)
            for start in range(0, sample_count, cfg.chunk_size):
                end = min(sample_count, start + cfg.chunk_size)
                initial = np.asarray(tensor[start:end, : cfg.initial_step, indices], dtype=np.float32)
                candidate_features = extract_analog_features(
                    initial,
                    cfg.feature_mode,
                    normalize_per_sample=cfg.normalize_per_sample,
                )
                records = [
                    {
                        "source_path": str(raw_path),
                        "sample_index": int(idx),
                        "nu": parse_nu_from_path(raw_path),
                    }
                    for idx in range(start, end)
                ]
                _update_topk(
                    query_features=query_features,
                    candidate_features=candidate_features,
                    records=records,
                    best_distances=best_distances,
                    best_records=best_records,
                    top_k=cfg.top_k,
                )
    prediction = _blend_neighbors(best_distances, best_records, target, cfg)
    prediction_path = _write_prediction(run_path / "task1_val_pred.hdf5", prediction)
    metrics = compute_task1_metrics(prediction, target)
    metrics_path = run_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    x_coords, t_coords = _read_coords(target_hdf5, cfg.spatial_size, cfg.output_steps)
    result = RunResult(
        task_id="task1",
        run_dir=run_path,
        metrics=metrics,
        prediction_path=prediction_path,
        zip_path=None,
        train_time=0.0,
        inference_time=0.0,
        success=True,
        command=["analog_forecast"],
    )
    write_baseline_artifacts(
        run_path,
        BaselineSpec(name="analog_knn", family="retrieval", trainable=False),
        {
            **asdict(cfg),
            "target_hdf5": str(target_hdf5),
            "raw_hdf5": [str(path) for path in raw_paths],
        },
        result,
        conclusion=f"analog validation competition_score_proxy={metrics['competition_score_proxy']:.6g}",
    )
    payload = {
        "version": 1,
        "updated_at": utc_now_iso(),
        "config": asdict(cfg),
        "target_hdf5": str(target_hdf5),
        "raw_hdf5": [str(path) for path in raw_paths],
        "prediction_path": str(prediction_path),
        "metrics": metrics,
        "neighbor_records": [
            [
                {
                    **record,
                    "distance": float(distance),
                }
                for record, distance in zip(records, best_distances[idx])
            ]
            for idx, records in enumerate(best_records)
        ],
        "estimated_nu": _estimate_query_nus(target, x_coords, t_coords, cfg),
    }
    summary_path = run_path / "analog_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_summary_markdown(run_path / "analog_summary.md", payload)
    return summary_path


def _write_summary_markdown(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    lines = [
        "# Task 1 Analog Forecast",
        "",
        f"- Updated: {payload['updated_at']}",
        f"- Prediction: `{payload['prediction_path']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| competition_score_proxy | {metrics.get('competition_score_proxy', '')} |",
        f"| mse | {metrics.get('mse', '')} |",
        f"| forecast_mse | {metrics.get('forecast_mse', '')} |",
        f"| long_horizon_mse | {metrics.get('long_horizon_mse', '')} |",
        f"| segment3_rmse | {metrics.get('segment3_rmse', '')} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 1 analog/kNN validation forecast.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--target", default="data/Task1/task1_val.hdf5")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--raw-hdf5", action="append", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-candidates-per-file", type=int, default=4000)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--feature-mode", default="initial_gradient", choices=["initial", "initial_gradient"])
    parser.add_argument("--normalize-per-sample", action="store_true")
    args = parser.parse_args()
    path = run_task1_analog_validation(
        run_dir=args.run_dir,
        target_hdf5=args.target,
        raw_hdf5=args.raw_hdf5,
        project_root=args.project_root,
        config=AnalogSearchConfig(
            top_k=args.top_k,
            max_candidates_per_file=args.max_candidates_per_file,
            chunk_size=args.chunk_size,
            feature_mode=args.feature_mode,
            normalize_per_sample=args.normalize_per_sample,
        ),
    )
    print(path)


if __name__ == "__main__":
    main()
