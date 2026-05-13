from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .logging import utc_now_iso
from .pde_metrics import compute_task1_metrics
from .run_task1_baseline_zoo import _read_prediction, run_validation_ensembles


ENSEMBLE_PREFIXES = ("global_ensemble", "cluster_em_ensemble", "temporal_tail_blend_")


def _is_ensemble_name(name: str) -> bool:
    return name in ENSEMBLE_PREFIXES[:2] or name.startswith("temporal_tail_blend_")


def collect_study_predictions(study_dir: str | Path, *, include_ensembles: bool = False) -> dict[str, np.ndarray]:
    study_path = Path(study_dir)
    predictions: dict[str, np.ndarray] = {}
    for path in sorted(study_path.glob("*/task1_val_pred.hdf5")):
        name = path.parent.name
        if not include_ensembles and _is_ensemble_name(name):
            continue
        predictions[name] = _read_prediction(path)
    return predictions


def _write_summary_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Task 1 Combo Search",
        "",
        f"- Updated: {payload['updated_at']}",
        f"- Study: {payload['study_dir']}",
        f"- Target: {payload['target_path']}",
        "",
        "| Rank | Name | Proxy | MSE | Forecast MSE | Long MSE | Segment3 RMSE |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, item in enumerate(payload["ranked_results"], start=1):
        metrics = item.get("metrics", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(item["name"]),
                    _format_float(metrics.get("competition_score_proxy")),
                    _format_float(metrics.get("mse")),
                    _format_float(metrics.get("forecast_mse")),
                    _format_float(metrics.get("long_horizon_mse")),
                    _format_float(metrics.get("segment3_rmse")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.10g}"


def _metrics_for_inputs(study_dir: Path, predictions: dict[str, np.ndarray], target: np.ndarray) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name, prediction in predictions.items():
        metrics_path = study_dir / name / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metrics = compute_task1_metrics(prediction, target)
        else:
            metrics = compute_task1_metrics(prediction, target)
        results.append(
            {
                "name": name,
                "kind": "input_prediction",
                "run_dir": str(study_dir / name),
                "prediction_path": str(study_dir / name / "task1_val_pred.hdf5"),
                "metrics": metrics,
            }
        )
    return results


def _rank_key(item: dict[str, Any]) -> tuple[float, float, str]:
    metrics = item.get("metrics", {})
    return (
        -float(metrics.get("competition_score_proxy", -1.0e18)),
        float(metrics.get("mse", 1.0e18)),
        str(item.get("name", "")),
    )


def run_task1_combo_search(
    *,
    study_dir: str | Path,
    target_path: str | Path = "data/Task1/task1_val.hdf5",
    grid_step: float = 0.05,
    include_ensembles: bool = False,
) -> Path:
    study_path = Path(study_dir)
    target = _read_prediction(target_path)
    predictions = collect_study_predictions(study_path, include_ensembles=include_ensembles)
    if len(predictions) < 2:
        raise ValueError(f"need at least two validation predictions in {study_path}")

    input_results = _metrics_for_inputs(study_path, predictions, target)
    ensemble_results = run_validation_ensembles(
        study_dir=study_path,
        target=target,
        predictions=predictions,
        grid_step=grid_step,
    )
    ranked = sorted(input_results + ensemble_results, key=_rank_key)
    payload = {
        "version": 1,
        "updated_at": utc_now_iso(),
        "study_dir": str(study_path),
        "target_path": str(target_path),
        "grid_step": grid_step,
        "include_ensembles": include_ensembles,
        "input_models": sorted(predictions),
        "best": ranked[0],
        "ranked_results": ranked,
    }
    summary_path = study_path / "combo_search_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_summary_markdown(study_path / "combo_search_summary.md", payload)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Task 1 validation-only baseline combinations without packaging.")
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--target", default="data/Task1/task1_val.hdf5")
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--include-ensembles", action="store_true")
    args = parser.parse_args()
    path = run_task1_combo_search(
        study_dir=args.study_dir,
        target_path=args.target,
        grid_step=args.grid_step,
        include_ensembles=args.include_ensembles,
    )
    print(path)


if __name__ == "__main__":
    main()
