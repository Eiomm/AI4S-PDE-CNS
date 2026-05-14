from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .pde_gating import fit_cluster_em_ensemble, fit_global_convex_ensemble
from .pde_metrics import compute_task1_metrics


@dataclass(frozen=True)
class ComboSearchConfig:
    base_name: str = "fno_ensemble"
    include_single_models: bool = True
    include_global: bool = True
    include_cluster: bool = True
    include_temporal: bool = True
    include_piecewise: bool = True
    include_cross_piecewise: bool = True
    grid_step: float = 0.05
    temporal_cut_min: int = 105
    temporal_cut_max: int = 199
    temporal_cut_stride: int = 1
    temporal_weight_step: float = 0.01
    piecewise_split_candidates: tuple[int, ...] = (120, 140, 160, 180)
    top_k: int = 20


@dataclass
class ComboCandidate:
    name: str
    kind: str
    config: dict[str, Any]
    prediction: np.ndarray
    metrics: dict[str, float]

    def to_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "config": dict(self.config),
            "metrics": dict(self.metrics),
        }


def search_task1_combinations(
    *,
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    config: ComboSearchConfig | None = None,
    initial: np.ndarray | None = None,
) -> list[ComboCandidate]:
    cfg = config or ComboSearchConfig()
    if not predictions:
        raise ValueError("predictions must not be empty")
    truth = np.asarray(target, dtype=np.float32)
    arrays = {name: _checked_prediction(name, prediction, truth.shape) for name, prediction in predictions.items()}
    base_name = cfg.base_name if cfg.base_name in arrays else next(iter(arrays))
    candidates: list[ComboCandidate] = []

    if cfg.include_single_models:
        for name, prediction in arrays.items():
            candidates.append(
                ComboCandidate(
                    name=name,
                    kind="single_model",
                    config={"kind": "single_model", "model": name},
                    prediction=_with_initial(prediction.copy(), truth),
                    metrics=_metrics_safe(prediction, truth),
                )
            )

    if cfg.include_global and len(arrays) >= 2:
        global_result = fit_global_convex_ensemble(arrays, truth, grid_step=cfg.grid_step)
        candidates.append(
            ComboCandidate(
                name="global_convex_ensemble",
                kind="global_convex_ensemble",
                config={"kind": "global_convex_ensemble", "weights": global_result.weights, "grid_step": cfg.grid_step},
                prediction=global_result.prediction,
                metrics=global_result.metrics,
            )
        )

    if cfg.include_cluster and len(arrays) >= 2:
        cluster_initial = initial if initial is not None else truth[:, :10, :]
        cluster_result = fit_cluster_em_ensemble(cluster_initial, arrays, truth, n_clusters=min(3, truth.shape[0]))
        candidates.append(
            ComboCandidate(
                name="cluster_gated_ensemble",
                kind="cluster_gated_ensemble",
                config={
                    "kind": "cluster_gated_ensemble",
                    "cluster_weights": cluster_result.cluster_weights,
                    "n_clusters": int(len(cluster_result.cluster_weights)),
                },
                prediction=cluster_result.prediction,
                metrics=cluster_result.metrics,
            )
        )

    tail_names = [name for name in arrays if name != base_name]
    for tail_name in tail_names:
        if cfg.include_temporal:
            candidates.append(_dense_temporal_candidate(arrays, truth, cfg=cfg, base_name=base_name, tail_name=tail_name))
        if cfg.include_piecewise:
            candidates.append(_piecewise_temporal_candidate(arrays, truth, cfg=cfg, base_name=base_name, tail_name=tail_name))
    if cfg.include_cross_piecewise and len(tail_names) >= 2:
        for early_tail_name in tail_names:
            for late_tail_name in tail_names:
                if early_tail_name == late_tail_name:
                    continue
                candidates.append(
                    _cross_piecewise_temporal_candidate(
                        arrays,
                        truth,
                        cfg=cfg,
                        base_name=base_name,
                        early_tail_name=early_tail_name,
                        late_tail_name=late_tail_name,
                    )
                )

    ranked = sorted(candidates, key=_rank_key)
    return ranked[: max(1, int(cfg.top_k))]


def _checked_prediction(name: str, prediction: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(prediction, dtype=np.float32)
    if values.shape != expected_shape:
        raise ValueError(f"prediction {name!r} shape {values.shape} does not match target shape {expected_shape}")
    return values


def _dense_temporal_candidate(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    cfg: ComboSearchConfig,
    base_name: str,
    tail_name: str,
) -> ComboCandidate:
    base = predictions[base_name]
    tail = predictions[tail_name]
    cut_min = _clamp_cut(cfg.temporal_cut_min, target.shape[1])
    cut_max = _clamp_cut(cfg.temporal_cut_max, target.shape[1])
    if cut_max < cut_min:
        cut_max = cut_min
    cuts = range(cut_min, cut_max + 1, max(1, int(cfg.temporal_cut_stride)))
    weights = _weight_grid(cfg.temporal_weight_step)
    best = _search_temporal_screen(base, tail, target, cuts=cuts, weights=weights)
    cut = int(best["cut"])
    weight = float(best["weight"])
    prediction = base.copy()
    prediction[:, cut:, :] = (1.0 - weight) * base[:, cut:, :] + weight * tail[:, cut:, :]
    prediction = _with_initial(prediction, target)
    return ComboCandidate(
        name=f"temporal_tail_blend_{tail_name}_cut{cut}_w{_weight_label(weight)}",
        kind="temporal_tail_blend",
        config={
            "kind": "temporal_tail_blend",
            "base_name": base_name,
            "tail_name": tail_name,
            "cut": cut,
            "tail_weight": weight,
        },
        prediction=prediction,
        metrics=_metrics_safe(prediction, target),
    )


def _piecewise_temporal_candidate(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    cfg: ComboSearchConfig,
    base_name: str,
    tail_name: str,
) -> ComboCandidate:
    base = predictions[base_name]
    tail = predictions[tail_name]
    start = _clamp_cut(cfg.temporal_cut_min, target.shape[1])
    splits = [cut for cut in cfg.piecewise_split_candidates if start < int(cut) < target.shape[1]]
    if not splits:
        splits = [min(target.shape[1] - 1, max(start + 1, 140))]
    weights = _weight_grid(cfg.temporal_weight_step)
    best = _search_piecewise_screen(base, tail, target, start=start, splits=splits, weights=weights)
    split = int(best["split"])
    early_weight = float(best["early_weight"])
    late_weight = float(best["late_weight"])
    prediction = base.copy()
    prediction[:, start:split, :] = (1.0 - early_weight) * base[:, start:split, :] + early_weight * tail[:, start:split, :]
    prediction[:, split:, :] = (1.0 - late_weight) * base[:, split:, :] + late_weight * tail[:, split:, :]
    prediction = _with_initial(prediction, target)
    return ComboCandidate(
        name=(
            f"piecewise_tail_blend_{tail_name}_start{start}_split{split}"
            f"_w{_weight_label(early_weight)}_{_weight_label(late_weight)}"
        ),
        kind="piecewise_temporal_blend",
        config={
            "kind": "piecewise_temporal_blend",
            "base_name": base_name,
            "tail_name": tail_name,
            "start": start,
            "split": split,
            "early_tail_weight": early_weight,
            "late_tail_weight": late_weight,
        },
        prediction=prediction,
        metrics=_metrics_safe(prediction, target),
    )


def _cross_piecewise_temporal_candidate(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    cfg: ComboSearchConfig,
    base_name: str,
    early_tail_name: str,
    late_tail_name: str,
) -> ComboCandidate:
    base = predictions[base_name]
    early_tail = predictions[early_tail_name]
    late_tail = predictions[late_tail_name]
    start = _clamp_cut(cfg.temporal_cut_min, target.shape[1])
    splits = [cut for cut in cfg.piecewise_split_candidates if start < int(cut) < target.shape[1]]
    if not splits:
        splits = [min(target.shape[1] - 1, max(start + 1, 140))]
    weights = _weight_grid(cfg.temporal_weight_step)
    best = _search_cross_piecewise_screen(
        base,
        early_tail,
        late_tail,
        target,
        start=start,
        splits=splits,
        weights=weights,
    )
    split = int(best["split"])
    early_weight = float(best["early_weight"])
    late_weight = float(best["late_weight"])
    prediction = base.copy()
    prediction[:, start:split, :] = (1.0 - early_weight) * base[:, start:split, :] + early_weight * early_tail[:, start:split, :]
    prediction[:, split:, :] = (1.0 - late_weight) * base[:, split:, :] + late_weight * late_tail[:, split:, :]
    prediction = _with_initial(prediction, target)
    return ComboCandidate(
        name=(
            f"cross_piecewise_tail_blend_{early_tail_name}_then_{late_tail_name}"
            f"_start{start}_split{split}_w{_weight_label(early_weight)}_{_weight_label(late_weight)}"
        ),
        kind="cross_piecewise_temporal_blend",
        config={
            "kind": "cross_piecewise_temporal_blend",
            "base_name": base_name,
            "early_tail_name": early_tail_name,
            "late_tail_name": late_tail_name,
            "start": start,
            "split": split,
            "early_tail_weight": early_weight,
            "late_tail_weight": late_weight,
        },
        prediction=prediction,
        metrics=_metrics_safe(prediction, target),
    )


def _search_temporal_screen(
    base: np.ndarray,
    tail: np.ndarray,
    target: np.ndarray,
    *,
    cuts: range,
    weights: np.ndarray,
) -> dict[str, float | int]:
    frame_terms = _frame_quadratic_terms(base, tail, target)
    prefix_a = _prefix(frame_terms["a"])
    prefix_b = _prefix(frame_terms["b"])
    prefix_c = _prefix(frame_terms["c"])
    total_a = prefix_a[-1]
    total_b = prefix_b[-1]
    total_c = prefix_c[-1]
    normalizer = float(np.prod(target.shape))
    best: dict[str, float | int] | None = None
    for cut in cuts:
        unaffected = prefix_a[cut]
        a_tail = total_a - prefix_a[cut]
        b_tail = total_b - prefix_b[cut]
        c_tail = total_c - prefix_c[cut]
        mse_values = (unaffected + a_tail + 2.0 * weights * b_tail + weights * weights * c_tail) / normalizer
        idx = int(np.argmin(mse_values))
        value = float(mse_values[idx])
        if best is None or value < float(best["screen_mse"]):
            best = {"cut": int(cut), "weight": float(weights[idx]), "screen_mse": value}
    assert best is not None
    return best


def _search_piecewise_screen(
    base: np.ndarray,
    tail: np.ndarray,
    target: np.ndarray,
    *,
    start: int,
    splits: list[int],
    weights: np.ndarray,
) -> dict[str, float | int]:
    terms = _frame_quadratic_terms(base, tail, target)
    prefix_a = _prefix(terms["a"])
    prefix_b = _prefix(terms["b"])
    prefix_c = _prefix(terms["c"])
    normalizer = float(np.prod(target.shape))
    best: dict[str, float | int] | None = None
    for split in splits:
        base_before = prefix_a[start]
        early = _quadratic_range(prefix_a, prefix_b, prefix_c, start, int(split), weights)
        late = _quadratic_range(prefix_a, prefix_b, prefix_c, int(split), target.shape[1], weights)
        grid = base_before + early[:, None] + late[None, :]
        idx = np.unravel_index(int(np.argmin(grid)), grid.shape)
        value = float(grid[idx] / normalizer)
        if best is None or value < float(best["screen_mse"]):
            best = {
                "split": int(split),
                "early_weight": float(weights[idx[0]]),
                "late_weight": float(weights[idx[1]]),
                "screen_mse": value,
            }
    assert best is not None
    return best


def _search_cross_piecewise_screen(
    base: np.ndarray,
    early_tail: np.ndarray,
    late_tail: np.ndarray,
    target: np.ndarray,
    *,
    start: int,
    splits: list[int],
    weights: np.ndarray,
) -> dict[str, float | int]:
    early_terms = _frame_quadratic_terms(base, early_tail, target)
    late_terms = _frame_quadratic_terms(base, late_tail, target)
    early_prefix_a = _prefix(early_terms["a"])
    early_prefix_b = _prefix(early_terms["b"])
    early_prefix_c = _prefix(early_terms["c"])
    late_prefix_a = _prefix(late_terms["a"])
    late_prefix_b = _prefix(late_terms["b"])
    late_prefix_c = _prefix(late_terms["c"])
    normalizer = float(np.prod(target.shape))
    best: dict[str, float | int] | None = None
    for split in splits:
        base_before = early_prefix_a[start]
        early = _quadratic_range(early_prefix_a, early_prefix_b, early_prefix_c, start, int(split), weights)
        late = _quadratic_range(late_prefix_a, late_prefix_b, late_prefix_c, int(split), target.shape[1], weights)
        grid = base_before + early[:, None] + late[None, :]
        idx = np.unravel_index(int(np.argmin(grid)), grid.shape)
        value = float(grid[idx] / normalizer)
        if best is None or value < float(best["screen_mse"]):
            best = {
                "split": int(split),
                "early_weight": float(weights[idx[0]]),
                "late_weight": float(weights[idx[1]]),
                "screen_mse": value,
            }
    assert best is not None
    return best


def _frame_quadratic_terms(base: np.ndarray, tail: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    error = base.astype(np.float64) - target.astype(np.float64)
    delta = tail.astype(np.float64) - base.astype(np.float64)
    return {
        "a": np.sum(error * error, axis=(0, 2)),
        "b": np.sum(error * delta, axis=(0, 2)),
        "c": np.sum(delta * delta, axis=(0, 2)),
    }


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(values.astype(np.float64))])


def _quadratic_range(
    prefix_a: np.ndarray,
    prefix_b: np.ndarray,
    prefix_c: np.ndarray,
    start: int,
    end: int,
    weights: np.ndarray,
) -> np.ndarray:
    a = prefix_a[end] - prefix_a[start]
    b = prefix_b[end] - prefix_b[start]
    c = prefix_c[end] - prefix_c[start]
    return a + 2.0 * weights * b + weights * weights * c


def _weight_grid(step: float) -> np.ndarray:
    if step <= 0.0 or step > 1.0:
        raise ValueError("temporal_weight_step must be inside (0, 1]")
    count = int(round(1.0 / step))
    values = np.linspace(0.0, 1.0, count + 1, dtype=np.float64)
    return np.clip(values, 0.0, 1.0)


def _clamp_cut(cut: int, total_steps: int) -> int:
    return max(10, min(int(cut), int(total_steps)))


def _with_initial(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction[:, :10, :] = target[:, :10, :]
    return prediction


def _metrics_safe(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    try:
        return compute_task1_metrics(prediction, target)
    except ValueError:
        mse = float(np.mean((np.asarray(prediction, dtype=np.float32) - np.asarray(target, dtype=np.float32)) ** 2))
        return {
            "mse": mse,
            "forecast_mse": mse,
            "long_horizon_mse": mse,
            "segment3_rmse": float(np.sqrt(mse)),
            "competition_score_proxy": 100.0 / (1.0 + mse),
        }


def _rank_key(candidate: ComboCandidate) -> tuple[float, float, str]:
    metrics = candidate.metrics
    return (
        -float(metrics.get("competition_score_proxy", -1.0e18)),
        float(metrics.get("mse", 1.0e18)),
        candidate.name,
    )


def _weight_label(weight: float) -> str:
    return f"{weight:.4f}".rstrip("0").rstrip(".").replace(".", "p")
