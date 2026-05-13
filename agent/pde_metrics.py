from __future__ import annotations

import math
from typing import Any

import numpy as np


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2))


def relative_mse(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    sample_cap: float = 5.0,
    eps: float = 1.0e-12,
) -> float:
    diff = prediction.astype(np.float64) - target.astype(np.float64)
    target64 = target.astype(np.float64)
    numerator = np.sum(diff * diff, axis=-1)
    denominator = np.sum(target64 * target64, axis=-1)
    rel_by_time = numerator / np.maximum(denominator, eps)
    rel_by_sample = np.mean(rel_by_time, axis=1)
    rel_by_sample = np.minimum(rel_by_sample, float(sample_cap))
    return float(np.mean(rel_by_sample))


def task1_competition_proxy_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
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
    competition_score_proxy = 0.25 * segment1_score + 0.25 * segment2_score + 0.5 * segment3_lorentzian_score
    return {
        "segment1_rel_mse": segment1_rel_mse,
        "segment2_rel_mse": segment2_rel_mse,
        "segment3_rmse": segment3_rmse,
        "segment1_score": segment1_score,
        "segment2_score": segment2_score,
        "segment3_lorentzian_score": segment3_lorentzian_score,
        "competition_score_proxy": competition_score_proxy,
    }


def compute_task1_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {prediction.shape} does not match target shape {target.shape}")
    if prediction.ndim != 3 or prediction.shape[1:] != (200, 256):
        raise ValueError(f"expected shape (N, 200, 256), got {prediction.shape}")
    metrics: dict[str, Any] = {
        "num_samples": int(prediction.shape[0]),
        "mse": mse(prediction, target),
        "initial_mse": mse(prediction[:, :10, :], target[:, :10, :]),
        "forecast_mse": mse(prediction[:, 10:, :], target[:, 10:, :]),
        "long_horizon_mse": mse(prediction[:, 105:, :], target[:, 105:, :]),
    }
    metrics.update(task1_competition_proxy_metrics(prediction, target))
    return metrics
