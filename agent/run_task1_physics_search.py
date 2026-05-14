from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from .pde_physics import (
    burgers_residual_mse,
    estimate_viscosity_from_initial,
    select_physics_rerank_candidate,
)
from .pde_search import Candidate, WeightedEnsembleSearch
from .pde_workflow import TASK1_FNO_CHECKPOINTS, Task1FNOWorkflow
from .run_task1_weight_search import (
    CachedFNOPredictionProvider,
    _candidate_grid,
    _quadratic_coefficients,
    _rank_candidates,
    _read_target,
    _weighted_average,
)


NU_VALUES: dict[str, float] = {
    "nu0.001": 0.001,
    "unet_pf20_nu0.001": 0.001,
}


def _coords(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        return h5["x-coordinate"][:], h5["t-coordinate"][:]


def _effective_nu(weights: Mapping[str, float]) -> float:
    total = sum(float(value) for value in weights.values())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return float(sum(float(weights.get(name, 0.0)) * nu for name, nu in NU_VALUES.items()) / total)


def _mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2))


def score_physics_candidates(
    single_predictions: dict[str, np.ndarray],
    target: np.ndarray,
    x_coords: np.ndarray,
    t_coords: np.ndarray,
    ranked_candidates: list[tuple[float, dict[str, float]]],
    *,
    physics_top_k: int,
    continuation: int,
    time_stride: int,
    nu_estimates: np.ndarray | None = None,
    spatial_margin: int = 4,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for rank, (quadratic_mse, weights) in enumerate(ranked_candidates[:physics_top_k], start=1):
        names = []
        arrays = []
        values = []
        for name, weight in weights.items():
            if float(weight) <= 0.0:
                continue
            if name not in single_predictions:
                continue
            names.append(name)
            arrays.append(single_predictions[name])
            values.append(float(weight))
        prediction = _weighted_average(arrays, values).astype(np.float32)
        nu = nu_estimates if nu_estimates is not None else _effective_nu(weights)
        physics_mse = burgers_residual_mse(
            prediction,
            x_coords,
            t_coords,
            nu=nu,
            continuation=continuation,
            time_stride=time_stride,
            spatial_margin=spatial_margin,
        )
        records.append(
            {
                "rank": rank,
                "name": f"rank-{rank:02d}-mse-{quadratic_mse:.8g}",
                "weights": dict(weights),
                "active_models": names,
                "mse": _mse(prediction, target),
                "quadratic_mse": float(quadratic_mse),
                "physics_mse": float(physics_mse),
                "effective_nu": _effective_nu(weights),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 FC-PINO-lite physics reranking for Task 1.")
    parser.add_argument("--search-name", default="task1-physics-rerank")
    parser.add_argument("--step", type=float, default=0.0025)
    parser.add_argument("--mse-top-k", type=int, default=128)
    parser.add_argument("--physics-top-k", type=int, default=48)
    parser.add_argument("--mse-tolerance", type=float, default=5.0e-4)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--continuation", type=int, default=16)
    parser.add_argument("--time-stride", type=int, default=4)
    parser.add_argument("--nu-source", choices=["effective", "estimated"], default="effective")
    parser.add_argument("--w0-max", type=float, default=0.025)
    parser.add_argument("--w1-min", type=float, default=0.24)
    parser.add_argument("--w1-max", type=float, default=0.39)
    parser.add_argument("--w3-max", type=float, default=0.08)
    args = parser.parse_args()

    project_root = Path(".").resolve()
    workflow = Task1FNOWorkflow(project_root=project_root)
    provider = CachedFNOPredictionProvider(project_root=project_root, batch_size=args.batch_size)

    val_predictions = provider.get_single_predictions(workflow.spec.validation_target_path)
    target = _read_target(workflow.spec.validation_target_path)
    x_coords, t_coords = _coords(workflow.spec.validation_target_path)
    names, gram, linear, constant = _quadratic_coefficients(val_predictions, target)
    raw_candidates = _candidate_grid(
        args.step,
        w0_max=args.w0_max,
        w1_min=args.w1_min,
        w1_max=args.w1_max,
        w3_max=args.w3_max,
    )
    ranked = _rank_candidates(raw_candidates, names, gram, linear, constant, top_k=args.mse_top_k)

    nu_estimates = None
    if args.nu_source == "estimated":
        nu_estimates = estimate_viscosity_from_initial(
            target,
            x_coords,
            t_coords,
            frames=10,
            continuation=args.continuation,
            spatial_margin=4,
        )

    physics_records = score_physics_candidates(
        val_predictions,
        target,
        x_coords,
        t_coords,
        ranked,
        physics_top_k=args.physics_top_k,
        continuation=args.continuation,
        time_stride=args.time_stride,
        nu_estimates=nu_estimates,
    )
    selected = select_physics_rerank_candidate(physics_records, mse_tolerance=args.mse_tolerance)
    best_by_mse = min(physics_records, key=lambda record: float(record["mse"]))

    run_root = workflow.run_root / args.search_name
    run_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "search_name": args.search_name,
        "step": args.step,
        "mse_top_k": args.mse_top_k,
        "physics_top_k": args.physics_top_k,
        "mse_tolerance": args.mse_tolerance,
        "nu_source": args.nu_source,
        "nu_estimate_summary": None
        if nu_estimates is None
        else {
            "min": float(np.min(nu_estimates)),
            "mean": float(np.mean(nu_estimates)),
            "max": float(np.max(nu_estimates)),
        },
        "best_by_mse": best_by_mse,
        "selected_by_physics": selected,
        "records": physics_records,
    }
    (run_root / "physics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    workflow.prediction_provider = provider
    candidate = Candidate(name=str(selected["name"]), weights=dict(selected["weights"]))
    search = WeightedEnsembleSearch(workflow=workflow, candidates=[candidate], search_name=args.search_name)
    validation = search.run(make_submission=False)
    if validation.best_candidate is None:
        raise RuntimeError("Physics-selected candidate failed validation")

    provider.drop_cache(workflow.spec.validation_target_path)
    submission = workflow.run_test_submission(candidate.weights, run_name=str(Path(args.search_name) / "best"))
    best_summary = {
        "selected_candidate": asdict(candidate),
        "physics_record": selected,
        "validation_result": validation.best_validation_result.to_json_dict(),
        "submission_result": submission.to_json_dict(),
    }
    (run_root / "best_summary.json").write_text(
        json.dumps(best_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(best_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
