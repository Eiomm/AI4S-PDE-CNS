from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .hdf5_io import read_named_or_single


def validate_task1_prediction(
    prediction_path: str | Path,
    initial_path: str | Path,
    *,
    atol: float = 1.0e-6,
    allow_prefix_samples: bool = False,
) -> dict[str, Any]:
    prediction = read_named_or_single(prediction_path, "tensor")
    initial = read_named_or_single(initial_path, "tensor")
    expected_samples = prediction.shape[0] if allow_prefix_samples else initial.shape[0]
    if allow_prefix_samples and prediction.shape[0] > initial.shape[0]:
        raise ValueError(f"prefix prediction has too many samples: {prediction.shape[0]} > {initial.shape[0]}")
    if prediction.shape != (expected_samples, 200, 256):
        raise ValueError(f"prediction shape must be ({initial.shape[0]}, 200, 256), got {prediction.shape}")
    if initial.ndim != 3 or initial.shape[1] < 10 or initial.shape[2] != 256:
        raise ValueError(f"initial tensor must have shape (N, >=10, 256), got {initial.shape}")
    initial10 = initial[: prediction.shape[0], :10, :].astype(np.float32)
    pred10 = prediction[:, :10, :].astype(np.float32)
    max_initial_error = float(np.max(np.abs(pred10 - initial10)))
    finite = bool(np.isfinite(prediction).all())
    report = {
        "shape": list(prediction.shape),
        "dtype": str(prediction.dtype),
        "first_ten_match": bool(np.allclose(pred10, initial10, atol=atol, rtol=0.0)),
        "max_initial_error": max_initial_error,
        "finite": finite,
        "min": float(np.min(prediction)),
        "max": float(np.max(prediction)),
        "mean": float(np.mean(prediction)),
    }
    if not report["first_ten_match"]:
        raise ValueError(f"first 10 frames do not match initial condition; max error {max_initial_error}")
    if not finite:
        raise ValueError("prediction contains non-finite values")
    return report
