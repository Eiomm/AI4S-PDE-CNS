from __future__ import annotations

import math
from typing import Any

import numpy as np


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2))


def relative_mse(prediction: np.ndarray, target: np.ndarray, *, sample_cap: float = 5.0, eps: float = 1.0e-12) -> float:
    diff = prediction.astype(np.float64) - target.astype(np.float64)
    target64 = target.astype(np.float64)
    numerator = np.sum(diff * diff, axis=-1)
    denominator = np.sum(target64 * target64, axis=-1)
    rel_by_time = numerator / np.maximum(denominator, eps)
    rel_by_sample = np.minimum(np.mean(rel_by_time, axis=1), float(sample_cap))
    return float(np.mean(rel_by_sample))


def _trajectory_features(data: np.ndarray, *, spectral_modes: int = 8) -> np.ndarray:
    """Extract compact long-horizon distribution features for Frechet scoring.

    The official evaluator does not publish its exact FD feature map. This local
    proxy keeps the feature dimension small and focuses on statistics relevant
    to Burgers rollouts: amplitude, spatial gradients, temporal increments, and
    low-frequency spectral energy.
    """

    arr = data.astype(np.float64, copy=False)
    dx = np.diff(arr, axis=2, append=arr[:, :, :1])
    dt = np.diff(arr, axis=1)
    spectrum = np.abs(np.fft.rfft(arr, axis=2)) ** 2
    modes = spectrum[:, :, 1 : spectral_modes + 1]
    if modes.shape[2] < spectral_modes:
        pad = np.zeros((modes.shape[0], modes.shape[1], spectral_modes - modes.shape[2]), dtype=modes.dtype)
        modes = np.concatenate([modes, pad], axis=2)

    parts = [
        np.mean(arr, axis=(1, 2)),
        np.std(arr, axis=(1, 2)),
        np.min(arr, axis=(1, 2)),
        np.max(arr, axis=(1, 2)),
        np.mean(arr * arr, axis=(1, 2)),
        np.std(np.mean(arr * arr, axis=2), axis=1),
        np.mean(dx * dx, axis=(1, 2)),
        np.std(np.mean(dx * dx, axis=2), axis=1),
        np.mean(dt * dt, axis=(1, 2)),
        np.std(np.mean(dt * dt, axis=2), axis=1),
        np.mean(modes, axis=1),
        np.std(modes, axis=1),
    ]
    return np.column_stack(parts)


def _symmetric_matrix_sqrt(matrix: np.ndarray, *, eps: float = 1.0e-9) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(sym)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values + eps)) @ vectors.T


def frechet_distance(features_a: np.ndarray, features_b: np.ndarray, *, eps: float = 1.0e-6) -> float:
    """Frechet distance between two empirical Gaussian feature distributions."""

    if features_a.ndim != 2 or features_b.ndim != 2:
        raise ValueError("Frechet features must be rank-2 arrays")
    if features_a.shape[1] != features_b.shape[1]:
        raise ValueError(f"feature dimensions differ: {features_a.shape[1]} vs {features_b.shape[1]}")
    combined = np.vstack([features_a, features_b])
    scale = np.std(combined, axis=0)
    scale = np.where(scale > eps, scale, 1.0)
    features_a = features_a / scale
    features_b = features_b / scale
    if features_a.shape[0] < 2 or features_b.shape[0] < 2:
        return float(np.sum((np.mean(features_a, axis=0) - np.mean(features_b, axis=0)) ** 2))

    mu_a = np.mean(features_a, axis=0)
    mu_b = np.mean(features_b, axis=0)
    cov_a = np.cov(features_a, rowvar=False) + eps * np.eye(features_a.shape[1])
    cov_b = np.cov(features_b, rowvar=False) + eps * np.eye(features_b.shape[1])
    sqrt_a = _symmetric_matrix_sqrt(cov_a, eps=0.0)
    middle = sqrt_a @ cov_b @ sqrt_a
    covmean = _symmetric_matrix_sqrt(middle, eps=0.0)
    distance = np.sum((mu_a - mu_b) ** 2) + np.trace(cov_a + cov_b - 2.0 * covmean)
    return float(max(distance, 0.0))


def compute_task1_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {prediction.shape} does not match target shape {target.shape}")
    if prediction.ndim != 3 or prediction.shape[1:] != (200, 256):
        raise ValueError(f"expected shape (N, 200, 256), got {prediction.shape}")

    future_prediction = prediction[:, 10:, :]
    future_target = target[:, 10:, :]
    segment1_pred, segment1_target = future_prediction[:, :47, :], future_target[:, :47, :]
    segment2_pred, segment2_target = future_prediction[:, 47:95, :], future_target[:, 47:95, :]
    segment3_pred, segment3_target = future_prediction[:, 95:, :], future_target[:, 95:, :]

    segment1_rel_mse = relative_mse(segment1_pred, segment1_target)
    segment2_rel_mse = relative_mse(segment2_pred, segment2_target)
    segment3_rmse = math.sqrt(mse(segment3_pred, segment3_target))
    segment1_score = 100.0 * math.exp(-20.0 * segment1_rel_mse)
    segment2_score = 100.0 * math.exp(-10.0 * segment2_rel_mse)
    segment3_lorentzian_score = 100.0 / (1.0 + 10.0 * segment3_rmse)
    segment3_fd = frechet_distance(_trajectory_features(segment3_pred), _trajectory_features(segment3_target))
    segment3_frechet_score = 50.0 * math.exp(-segment3_fd * segment3_fd)
    segment3_score = max(segment3_lorentzian_score, segment3_frechet_score)
    competition_score_proxy = 0.25 * segment1_score + 0.25 * segment2_score + 0.5 * segment3_score

    return {
        "num_samples": int(prediction.shape[0]),
        "mse": mse(prediction, target),
        "initial_mse": mse(prediction[:, :10, :], target[:, :10, :]),
        "forecast_mse": mse(prediction[:, 10:, :], target[:, 10:, :]),
        "long_horizon_mse": mse(prediction[:, 105:, :], target[:, 105:, :]),
        "segment1_rel_mse": segment1_rel_mse,
        "segment2_rel_mse": segment2_rel_mse,
        "segment3_rmse": segment3_rmse,
        "segment3_frechet_distance_proxy": segment3_fd,
        "segment1_score": segment1_score,
        "segment2_score": segment2_score,
        "segment3_lorentzian_score": segment3_lorentzian_score,
        "segment3_frechet_score_proxy": segment3_frechet_score,
        "segment3_score_proxy": segment3_score,
        "competition_score_proxy": competition_score_proxy,
    }
