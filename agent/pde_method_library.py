from __future__ import annotations

from copy import deepcopy
from typing import Any


METHOD_LIBRARY: list[dict[str, Any]] = [
    {
        "name": "PINO-style physics/data hybrid loss",
        "source": "Physics-Informed Neural Operator, 2024",
        "problem_signal": ["rollout drift", "physics inconsistency", "long_horizon_mse high"],
        "applicable_to": ["task1", "task2", "FNO"],
        "implementation_knobs": {
            "gradient_loss_weight": [0.001, 0.005, 0.01],
            "spectral_loss_weight": [0.0, 0.0001, 0.001],
            "physics_loss_weight": [0.0001, 0.0005, 0.001],
            "physics_nu": [0.001],
            "horizon_loss_gamma": [1.02, 1.05, 1.1],
        },
        "risk": "Physics-style regularization can over-smooth Burgers shocks when too large.",
    },
    {
        "name": "Localized kernel / local differential regularization",
        "source": "Neural Operators with Localized Integral and Differential Kernels, 2024",
        "problem_signal": ["local gradients", "shock-like structure", "high-frequency detail loss"],
        "applicable_to": ["task1", "FNO"],
        "implementation_knobs": {
            "gradient_loss_weight": [0.001, 0.005, 0.01, 0.02],
            "trainable": ["last-block-head", "all"],
        },
        "risk": "Local-detail emphasis can hurt smooth-region accuracy if overweighted.",
    },
    {
        "name": "DPOT-style autoregressive rollout stability",
        "source": "DPOT: Auto-Regressive Denoising Operator Transformer, 2024",
        "problem_signal": ["accumulated rollout drift", "long trajectory instability"],
        "applicable_to": ["task1", "FNO"],
        "implementation_knobs": {
            "rollout_steps": [2, 3],
            "horizon_loss_gamma": [1.02, 1.05, 1.1],
            "lr": [1e-6, 2e-6, 3e-6],
        },
        "risk": "Too many rollout steps can destabilize fine-tuning from a one-step checkpoint.",
    },
    {
        "name": "CoDA-NO multiphysics conditioning",
        "source": "Pretraining Codomain Attention Neural Operators, 2024",
        "problem_signal": ["multi-nu generalization", "multiphysics transfer", "limited data"],
        "applicable_to": ["task2"],
        "implementation_knobs": {
            "condition_on_nu": [True],
            "latent_dim": [16, 32],
            "normalization": ["per-sample", "global"],
        },
        "risk": "Task2 test does not expose Nu, so conditioning must be inferred or marginalized.",
    },
    {
        "name": "Transolver physics-attention state tokens",
        "source": "Transolver, ICML 2024",
        "problem_signal": ["architecture bottleneck", "nonlocal physical correlation"],
        "applicable_to": ["task1", "task2"],
        "implementation_knobs": {
            "prototype": ["correction-head", "attention-refiner"],
            "architecture": ["residual-corrected-fno"],
            "trainable": ["residual-head", "last-block-head"],
            "validation_scope": ["small"],
        },
        "risk": "Full architecture replacement is expensive; start with a small refiner.",
    },
    {
        "name": "Flow-Marching-inspired long-horizon stabilization",
        "source": "Flow Marching, 2025",
        "problem_signal": ["long-term error accumulation", "trajectory uncertainty"],
        "applicable_to": ["task1"],
        "implementation_knobs": {
            "candidate_rollouts": [2, 4],
            "horizon_loss_gamma": [1.02, 1.05, 1.1],
            "postprocess_search": ["segment-persistence", "candidate-selection"],
        },
        "risk": "Generative flow matching is too heavy for the current baseline; use the stability idea first.",
    },
]


def _score_method(method: dict[str, Any], *, task: str, metrics: dict[str, float]) -> tuple[int, str]:
    if task and task not in {str(item).lower() for item in method.get("applicable_to", [])}:
        return (0, "not applicable")
    if task == "task2" and "task2" in {str(item).lower() for item in method.get("applicable_to", [])}:
        return (4, "Task2 needs multi-physics / multi-Nu generalization support.")
    segment1 = float(metrics.get("segment1_rel_mse", 0.0) or 0.0)
    segment2 = float(metrics.get("segment2_rel_mse", 0.0) or 0.0)
    forecast = float(metrics.get("forecast_mse", 0.0) or 0.0)
    long_horizon = float(metrics.get("long_horizon_mse", 0.0) or 0.0)
    seg3_rmse = float(metrics.get("segment3_rmse", 0.0) or 0.0)
    signals = {str(item).lower() for item in method.get("problem_signal", [])}
    if segment1 and segment2 > 2.0 * segment1 and any("rollout" in item or "long" in item for item in signals):
        return (6, "Segment-2 error is much larger than segment-1; accumulated rollout drift dominates.")
    if forecast and long_horizon > 1.1 * forecast and any("long" in item or "rollout" in item for item in signals):
        return (5, "Long-horizon MSE is higher than average forecast MSE.")
    if seg3_rmse > 0.025 and any("local" in item or "high-frequency" in item or "shock" in item for item in signals):
        return (4, "Late segment RMSE suggests local detail or high-frequency degradation.")
    if not metrics and task == "task1":
        return (2, "Default Task1 method candidate before metrics are available.")
    return (1, "General low-cost method candidate.")


def select_method_candidates(
    *,
    task: str = "task1",
    metrics: dict[str, float] | None = None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Return compact method cards ranked by current PDE error signals."""
    task_key = task.lower()
    metric_values = dict(metrics or {})
    ranked: list[tuple[int, dict[str, Any]]] = []
    for method in METHOD_LIBRARY:
        score, reason = _score_method(method, task=task_key, metrics=metric_values)
        if score <= 0:
            continue
        candidate = deepcopy(method)
        candidate["reason"] = reason
        candidate["score"] = score
        ranked.append((score, candidate))
    ranked.sort(key=lambda item: (-item[0], item[1]["name"]))
    return [candidate for _, candidate in ranked[:limit]]
