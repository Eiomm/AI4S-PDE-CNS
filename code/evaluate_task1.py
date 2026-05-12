from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _read_dataset(path: Path, preferred_key: str) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if preferred_key in h5:
            return h5[preferred_key][:]
        if len(h5.keys()) == 1:
            return h5[next(iter(h5.keys()))][:]
        raise KeyError(f"{path} must contain {preferred_key!r} or a single dataset")


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))


def compute_task1_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {prediction.shape} does not match target shape {target.shape}")
    if prediction.ndim != 3 or prediction.shape[1:] != (200, 256):
        raise ValueError(f"expected shape (N, 200, 256), got {prediction.shape}")
    metrics = {
        "num_samples": int(prediction.shape[0]),
        "mse": _mse(prediction, target),
        "initial_mse": _mse(prediction[:, :10, :], target[:, :10, :]),
        "forecast_mse": _mse(prediction[:, 10:, :], target[:, 10:, :]),
        "long_horizon_mse": _mse(prediction[:, 105:, :], target[:, 105:, :]),
    }
    return metrics


def evaluate_prediction_file(
    prediction_path: str | Path,
    target_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    prediction = _read_dataset(Path(prediction_path), "prediction")
    target = _read_dataset(Path(target_path), "tensor")
    metrics = compute_task1_metrics(prediction, target)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task 1 predictions against validation targets.")
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    metrics = evaluate_prediction_file(args.prediction, args.target, args.output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
