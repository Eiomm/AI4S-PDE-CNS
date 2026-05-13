from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from .pde_search import Candidate, WeightedEnsembleSearch
from .pde_tasks import DEFAULT_TASK1_FNO_WEIGHTS
from .pde_workflow import TASK1_FNO_CHECKPOINTS, Task1FNOWorkflow


def parse_checkpoint_overrides(values: list[str] | None) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"checkpoint override must use KEY=PATH format, got {value!r}")
        key, path = value.split("=", 1)
        key = key.strip()
        if key not in TASK1_FNO_CHECKPOINTS:
            raise ValueError(f"unknown checkpoint key {key!r}; expected one of {sorted(TASK1_FNO_CHECKPOINTS)}")
        if not path.strip():
            raise ValueError(f"checkpoint override for {key!r} has an empty path")
        overrides[key] = Path(path)
    return overrides


class CachedFNOPredictionProvider:
    def __init__(
        self,
        *,
        project_root: Path,
        batch_size: int,
        checkpoint_paths: Mapping[str, str | Path] | None = None,
    ):
        self.project_root = project_root
        self.batch_size = batch_size
        paths: dict[str, Path] = {key: Path(value) for key, value in TASK1_FNO_CHECKPOINTS.items()}
        if checkpoint_paths is not None:
            paths.update({key: Path(value) for key, value in checkpoint_paths.items()})
        self.checkpoint_paths = paths
        self._cache: dict[Path, dict[str, np.ndarray]] = {}

    def __call__(self, input_path: Path, weights: Mapping[str, float], output_steps: int) -> np.ndarray:
        predictions = self.get_single_predictions(input_path)
        arrays = []
        values = []
        for name in self.checkpoint_paths:
            weight = float(weights.get(name, 0.0))
            if weight <= 0.0:
                continue
            arrays.append(predictions[name])
            values.append(weight)
        if not arrays:
            raise ValueError("At least one positive checkpoint weight is required")
        return _weighted_average(arrays, values).astype(np.float32)

    def get_single_predictions(self, input_path: str | Path) -> dict[str, np.ndarray]:
        path = Path(input_path).resolve()
        if path not in self._cache:
            self._cache[path] = self._run_all_checkpoints(path)
        return self._cache[path]

    def drop_cache(self, input_path: str | Path) -> None:
        self._cache.pop(Path(input_path).resolve(), None)

    def _run_all_checkpoints(self, input_path: Path) -> dict[str, np.ndarray]:
        import torch

        code_dir = self.project_root / "code"
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        from fno_inference import load_fno_checkpoint, run_autoregressive_inference

        tensor, x_coords, t_coords_full = _load_task1_input(input_path)
        initial = tensor[:, :10, :]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"caching {input_path} on {device}, samples={tensor.shape[0]}")
        predictions: dict[str, np.ndarray] = {}
        for name, rel_checkpoint in self.checkpoint_paths.items():
            checkpoint = self.project_root / rel_checkpoint
            t0 = time.perf_counter()
            model = load_fno_checkpoint(str(checkpoint), device)
            pred = run_autoregressive_inference(
                model,
                initial,
                x_coords,
                t_coords_full,
                device,
                batch_size=self.batch_size,
            )
            predictions[name] = pred.astype(np.float32)
            print(f"  {name}: {time.perf_counter() - t0:.3f}s")
        return predictions


def _load_task1_input(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        tensor = h5["tensor"][:]
        x_coords = h5["x-coordinate"][:]
        t_coords = h5["t-coordinate"][:]
    if len(t_coords) == 200:
        return tensor, x_coords, t_coords
    val_path = path.with_name("task1_val.hdf5")
    if val_path.exists():
        with h5py.File(val_path, "r") as h5:
            return tensor, x_coords, h5["t-coordinate"][:]
    return tensor, x_coords, np.linspace(t_coords[0], t_coords[-1] * 20, 200).astype(np.float32)


def _read_target(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        return h5["tensor"][:].astype(np.float32)


def _weighted_average(predictions: list[np.ndarray], weights: list[float]) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    values = values / total
    combined = np.zeros_like(predictions[0], dtype=np.float64)
    for value, prediction in zip(values, predictions):
        combined += float(value) * prediction
    return combined


def _quadratic_coefficients(
    single_predictions: dict[str, np.ndarray],
    target: np.ndarray,
    checkpoint_names: list[str] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    names = checkpoint_names or list(TASK1_FNO_CHECKPOINTS)
    arrays = [single_predictions[name].astype(np.float64) for name in names]
    target64 = target.astype(np.float64)
    gram = np.zeros((len(names), len(names)), dtype=np.float64)
    linear = np.zeros(len(names), dtype=np.float64)
    for i, left in enumerate(arrays):
        linear[i] = float(np.mean(left * target64))
        for j in range(i, len(arrays)):
            value = float(np.mean(left * arrays[j]))
            gram[i, j] = value
            gram[j, i] = value
    constant = float(np.mean(target64 * target64))
    return names, gram, linear, constant


def _mse_from_quadratic(weights: np.ndarray, gram: np.ndarray, linear: np.ndarray, constant: float) -> float:
    return float(weights @ gram @ weights - 2.0 * linear @ weights + constant)


def _candidate_grid(step: float, *, w0_max: float, w1_min: float, w1_max: float, w3_max: float) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    w0_values = np.arange(0.0, w0_max + step / 2.0, step)
    w1_values = np.arange(w1_min, w1_max + step / 2.0, step)
    w3_values = np.arange(0.0, w3_max + step / 2.0, step)
    for w0 in w0_values:
        for w1 in w1_values:
            for w3 in w3_values:
                w2 = 1.0 - float(w0) - float(w1) - float(w3)
                if w2 < 0.0:
                    continue
                candidates.append(
                    {
                        "nu0.001": round(float(w0), 6),
                        "nu0.01": round(float(w1), 6),
                        "nu0.1": round(float(w2), 6),
                        "nu1.0": round(float(w3), 6),
                    }
                )
    candidates.append(dict(DEFAULT_TASK1_FNO_WEIGHTS))
    return candidates


def _rank_candidates(
    candidates: list[dict[str, float]],
    names: list[str],
    gram: np.ndarray,
    linear: np.ndarray,
    constant: float,
    *,
    top_k: int,
) -> list[tuple[float, dict[str, float]]]:
    ranked = []
    seen = set()
    for weights in candidates:
        key = tuple(float(weights.get(name, 0.0)) for name in names)
        if key in seen:
            continue
        seen.add(key)
        values = np.asarray(key, dtype=np.float64)
        values = values / float(values.sum())
        ranked.append((_mse_from_quadratic(values, gram, linear, constant), weights))
    ranked.sort(key=lambda item: item[0])
    return ranked[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cached Task 1 FNO ensemble weight search.")
    parser.add_argument("--search-name", default="task1-weight-search-fine")
    parser.add_argument("--step", type=float, default=0.0025)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--w0-max", type=float, default=0.025)
    parser.add_argument("--w1-min", type=float, default=0.24)
    parser.add_argument("--w1-max", type=float, default=0.39)
    parser.add_argument("--w3-max", type=float, default=0.08)
    parser.add_argument("--selection-metric", default="competition_score_proxy")
    parser.add_argument("--selection-direction", choices=["min", "max"], default=None)
    parser.add_argument(
        "--checkpoint-override",
        action="append",
        default=[],
        help="Override an FNO checkpoint path with KEY=PATH, e.g. nu0.1=runs/finetune/best.pt",
    )
    args = parser.parse_args()

    project_root = Path(".").resolve()
    checkpoint_overrides = parse_checkpoint_overrides(args.checkpoint_override)
    workflow = Task1FNOWorkflow(project_root=project_root, checkpoint_paths=checkpoint_overrides)
    provider = CachedFNOPredictionProvider(
        project_root=project_root,
        batch_size=args.batch_size,
        checkpoint_paths=checkpoint_overrides,
    )

    val_predictions = provider.get_single_predictions(workflow.spec.validation_target_path)
    target = _read_target(workflow.spec.validation_target_path)
    names, gram, linear, constant = _quadratic_coefficients(val_predictions, target, list(provider.checkpoint_paths))
    raw_candidates = _candidate_grid(
        args.step,
        w0_max=args.w0_max,
        w1_min=args.w1_min,
        w1_max=args.w1_max,
        w3_max=args.w3_max,
    )
    ranked = _rank_candidates(raw_candidates, names, gram, linear, constant, top_k=args.top_k)

    summary_path = workflow.run_root / args.search_name / "search_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "search_name": args.search_name,
        "step": args.step,
        "grid_candidates": len(raw_candidates),
        "top_k": args.top_k,
        "checkpoint_overrides": {key: str(value) for key, value in checkpoint_overrides.items()},
        "quadratic_top": [{"mse": mse, "weights": weights} for mse, weights in ranked],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    workflow.prediction_provider = provider
    candidates = [
        Candidate(name=f"rank-{index:02d}-mse-{mse:.8g}", weights=weights)
        for index, (mse, weights) in enumerate(ranked, start=1)
    ]
    selection_direction = args.selection_direction
    if selection_direction is None:
        selection_direction = "max" if args.selection_metric.endswith("score") or "score" in args.selection_metric else "min"
    search = WeightedEnsembleSearch(
        workflow=workflow,
        candidates=candidates,
        search_name=args.search_name,
        metric=args.selection_metric,
        maximize=selection_direction == "max",
    )
    search_result = search.run(make_submission=False)
    if search_result.best_candidate is None or search_result.best_validation_result is None:
        raise RuntimeError("No successful validation candidate")

    provider.drop_cache(workflow.spec.validation_target_path)
    best_submission = workflow.run_test_submission(
        search_result.best_candidate.weights,
        run_name=str(Path(args.search_name) / "best"),
    )

    final_summary = {
        "best_candidate": asdict(search_result.best_candidate),
        "best_validation_result": search_result.best_validation_result.to_json_dict(),
        "best_submission_result": best_submission.to_json_dict(),
    }
    final_path = workflow.run_root / args.search_name / "best_summary.json"
    final_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
