"""Check prediction shape and first-10-frame consistency."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


DEFAULT_INPUT = (
    "data/data_and_sample_submission/data_and_sample_submission/"
    "train_val_test_init/task1_test.hdf5"
)


def _read_single_or_named(path: Path, preferred_key: str) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if preferred_key in h5:
            return h5[preferred_key][:]
        keys = list(h5.keys())
        if len(keys) == 1:
            return h5[keys[0]][:]
        raise KeyError(f"{path} must contain {preferred_key!r} or one dataset, got {keys}")


def check_prediction_file(
    prediction_path: str | Path,
    input_path: str | Path = DEFAULT_INPUT,
    *,
    atol: float = 1e-6,
) -> dict[str, object]:
    prediction = _read_single_or_named(Path(prediction_path), "prediction")
    initial = _read_single_or_named(Path(input_path), "tensor")
    if prediction.ndim != 3:
        raise ValueError(f"prediction must be 3D, got {prediction.shape}")
    if initial.ndim != 3 or initial.shape[1:] != (10, 256):
        raise ValueError(f"initial tensor must have shape (N, 10, 256), got {initial.shape}")
    if prediction.shape[0] != initial.shape[0]:
        raise ValueError(f"sample count mismatch: prediction {prediction.shape[0]} vs initial {initial.shape[0]}")
    initial_error = np.abs(prediction[:, :10, :] - initial)
    max_initial_error = float(initial_error.max())
    return {
        "shape": tuple(prediction.shape),
        "dtype": str(prediction.dtype),
        "first_ten_match": bool(np.allclose(prediction[:, :10, :], initial, atol=atol, rtol=0.0)),
        "max_initial_error": max_initial_error,
        "finite": bool(np.isfinite(prediction).all()),
        "min": float(prediction.min()),
        "max": float(prediction.max()),
        "mean": float(prediction.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Task 1 prediction file shape and initial frames.")
    parser.add_argument("prediction", nargs="?", default="runs/task1-fno-ensemble-test/task1_pred.hdf5")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    report = check_prediction_file(args.prediction, args.input, atol=args.atol)
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
