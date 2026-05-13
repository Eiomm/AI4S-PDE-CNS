from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pde_metrics import compute_task1_metrics


@dataclass
class EnsembleResult:
    weights: dict[str, float]
    prediction: np.ndarray
    metrics: dict[str, float]


@dataclass
class ClusterEnsembleResult:
    cluster_weights: dict[int, dict[str, float]]
    assignments: np.ndarray
    prediction: np.ndarray
    metrics: dict[str, float]


@dataclass
class TemporalBlendResult:
    prediction: np.ndarray
    metrics: dict[str, float]
    config: dict[str, float | int | str]


def softmax_rows(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32)


def extract_task1_initial_features(initial: np.ndarray) -> np.ndarray:
    values = np.asarray(initial, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"initial must have shape (N, T, X), got {values.shape}")
    gradients = np.diff(values, axis=2)
    spectrum = np.fft.rfft(values[:, -1, :], axis=1)
    power = np.abs(spectrum) ** 2
    cutoff = max(1, power.shape[1] // 4)
    high = np.sum(power[:, cutoff:], axis=1)
    total = np.sum(power, axis=1) + 1.0e-12
    temporal = values[:, -1, :] - values[:, 0, :]
    features = np.stack(
        [
            np.mean(values, axis=(1, 2)),
            np.std(values, axis=(1, 2)),
            np.mean(values**2, axis=(1, 2)),
            np.max(np.abs(gradients), axis=(1, 2)),
            high / total,
            np.mean(np.abs(temporal), axis=1),
        ],
        axis=1,
    )
    return np.nan_to_num(features.astype(np.float32))


def _simplex_grid(size: int, step: float) -> list[np.ndarray]:
    if size <= 0:
        raise ValueError("size must be positive")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("grid_step must divide 1.0")
    out: list[np.ndarray] = []

    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            out.append(np.asarray(prefix + [remaining], dtype=np.float64) / units)
            return
        for value in range(remaining + 1):
            rec(prefix + [value], remaining - value, slots - 1)

    rec([], units, size)
    return out


def _combine(names: list[str], predictions: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    output = np.zeros_like(predictions[names[0]], dtype=np.float32)
    for name, weight in zip(names, weights):
        output += float(weight) * np.asarray(predictions[name], dtype=np.float32)
    return output


def _metrics_safe(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((np.asarray(prediction, dtype=np.float32) - np.asarray(target, dtype=np.float32)) ** 2))
    try:
        return compute_task1_metrics(prediction, target)
    except ValueError:
        return {
            "mse": mse,
            "forecast_mse": mse,
            "competition_score_proxy": 100.0 / (1.0 + mse),
        }


def fit_global_convex_ensemble(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    grid_step: float = 0.05,
    metric: str = "competition_score_proxy",
    maximize: bool = True,
) -> EnsembleResult:
    if not predictions:
        raise ValueError("predictions must not be empty")
    names = list(predictions)
    target = np.asarray(target, dtype=np.float32)
    best_weights: np.ndarray | None = None
    best_prediction: np.ndarray | None = None
    best_value: float | None = None
    for weights in _simplex_grid(len(names), grid_step):
        prediction = _combine(names, predictions, weights)
        prediction[:, :10, :] = target[:, :10, :]
        # The full Task 1 proxy is useful for final reporting but expensive
        # inside a dense simplex grid. Screen candidates with MSE, then compute
        # full metrics once for the selected prediction.
        mse = float(np.mean((prediction - target) ** 2))
        value = -mse if maximize else mse
        if best_value is None:
            better = True
        elif maximize:
            better = value > best_value
        else:
            better = value < best_value
        if better:
            best_weights = weights
            best_prediction = prediction
            best_value = value
    assert best_weights is not None and best_prediction is not None
    best_metrics = _metrics_safe(best_prediction, target)
    return EnsembleResult(
        weights={name: float(weight) for name, weight in zip(names, best_weights)},
        prediction=best_prediction,
        metrics=best_metrics,
    )


def _standardize(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    return (values - values.mean(axis=0, keepdims=True)) / (values.std(axis=0, keepdims=True) + 1.0e-8)


def _kmeans(features: np.ndarray, n_clusters: int, *, iterations: int = 20) -> np.ndarray:
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if features.shape[0] < n_clusters:
        raise ValueError("n_clusters must be <= number of samples")
    values = _standardize(features)
    order = np.argsort(values[:, 0])
    centers = values[order[np.linspace(0, len(order) - 1, n_clusters, dtype=int)]].copy()
    assignments = np.zeros(values.shape[0], dtype=np.int64)
    for _ in range(iterations):
        distances = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        next_assignments = np.argmin(distances, axis=1)
        if np.array_equal(assignments, next_assignments):
            break
        assignments = next_assignments
        for idx in range(n_clusters):
            mask = assignments == idx
            if np.any(mask):
                centers[idx] = values[mask].mean(axis=0)
    return assignments


def fit_cluster_em_ensemble(
    initial: np.ndarray,
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    n_clusters: int = 3,
) -> ClusterEnsembleResult:
    if not predictions:
        raise ValueError("predictions must not be empty")
    features = extract_task1_initial_features(initial)
    assignments = _kmeans(features, n_clusters=n_clusters)
    names = list(predictions)
    target = np.asarray(target, dtype=np.float32)
    output = np.zeros_like(target, dtype=np.float32)
    cluster_weights: dict[int, dict[str, float]] = {}
    for cluster_id in sorted(set(int(value) for value in assignments)):
        mask = assignments == cluster_id
        errors = {
            name: float(np.mean((np.asarray(predictions[name], dtype=np.float32)[mask] - target[mask]) ** 2))
            for name in names
        }
        best_name = min(errors, key=errors.get)
        weights = {name: (1.0 if name == best_name else 0.0) for name in names}
        cluster_weights[cluster_id] = weights
        output[mask] = np.asarray(predictions[best_name], dtype=np.float32)[mask]
    output[:, :10, :] = target[:, :10, :]
    return ClusterEnsembleResult(
        cluster_weights=cluster_weights,
        assignments=assignments,
        prediction=output,
        metrics=_metrics_safe(output, target),
    )


def fit_temporal_tail_blend(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    base_name: str,
    tail_name: str,
    cut_candidates: list[int] | tuple[int, ...] = (105, 120, 140, 160),
    tail_weights: list[float] | tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 21)),
) -> TemporalBlendResult:
    if base_name not in predictions:
        raise KeyError(f"missing base prediction {base_name!r}")
    if tail_name not in predictions:
        raise KeyError(f"missing tail prediction {tail_name!r}")
    base = np.asarray(predictions[base_name], dtype=np.float32)
    tail = np.asarray(predictions[tail_name], dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    if base.shape != truth.shape or tail.shape != truth.shape:
        raise ValueError("base, tail, and target predictions must have the same shape")
    best_prediction: np.ndarray | None = None
    best_metrics: dict[str, float] | None = None
    best_config: dict[str, float | int | str] | None = None
    for cut in cut_candidates:
        if cut < 10 or cut > truth.shape[1]:
            raise ValueError(f"cut must be inside [10, {truth.shape[1]}], got {cut}")
        for tail_weight in tail_weights:
            weight = float(tail_weight)
            if weight < 0.0 or weight > 1.0:
                raise ValueError("tail weights must be between 0 and 1")
            prediction = base.copy()
            prediction[:, cut:, :] = (1.0 - weight) * base[:, cut:, :] + weight * tail[:, cut:, :]
            prediction[:, :10, :] = truth[:, :10, :]
            metrics = _metrics_safe(prediction, truth)
            if best_metrics is None or float(metrics["competition_score_proxy"]) > float(best_metrics["competition_score_proxy"]):
                best_prediction = prediction
                best_metrics = metrics
                best_config = {
                    "kind": "temporal_tail_blend",
                    "base_name": base_name,
                    "tail_name": tail_name,
                    "cut": int(cut),
                    "tail_weight": weight,
                }
    assert best_prediction is not None and best_metrics is not None and best_config is not None
    return TemporalBlendResult(prediction=best_prediction, metrics=best_metrics, config=best_config)
