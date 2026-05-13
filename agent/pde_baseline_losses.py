from __future__ import annotations

import numpy as np

from .pde_physics import burgers_residual_mse as _burgers_residual_mse


def initial_consistency_mse(prediction: np.ndarray, initial: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=np.float32)
    init = np.asarray(initial, dtype=np.float32)
    if pred.ndim != 3 or init.ndim != 3:
        raise ValueError("prediction and initial must have shape (N, T, X)")
    if pred.shape[0] != init.shape[0] or pred.shape[2] != init.shape[2] or pred.shape[1] < init.shape[1]:
        raise ValueError(f"prediction shape {pred.shape} is incompatible with initial {init.shape}")
    return float(np.mean((pred[:, : init.shape[1], :] - init) ** 2))


def spectral_mse(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    low_weight: float = 1.0,
    mid_weight: float = 1.0,
    high_weight: float = 1.0,
) -> float:
    pred = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    if pred.shape != truth.shape:
        raise ValueError(f"prediction shape {pred.shape} does not match target shape {truth.shape}")
    pred_fft = np.fft.rfft(pred, axis=-1)
    truth_fft = np.fft.rfft(truth, axis=-1)
    error = np.abs(pred_fft - truth_fft) ** 2
    bands = np.array_split(np.arange(error.shape[-1]), 3)
    weights = np.ones(error.shape[-1], dtype=np.float32)
    for band, weight in zip(bands, (low_weight, mid_weight, high_weight)):
        weights[band] = float(weight)
    return float(np.mean(error * weights.reshape((1,) * (error.ndim - 1) + (-1,))))


def burgers_residual_mse(
    trajectory: np.ndarray,
    x_coords: np.ndarray,
    t_coords: np.ndarray,
    *,
    nu: float | np.ndarray,
) -> float:
    return _burgers_residual_mse(trajectory, x_coords, t_coords, nu=nu)
