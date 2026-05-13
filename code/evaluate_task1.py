from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.pde_metrics import compute_task1_metrics  # noqa: E402


def _read_dataset(path: Path, preferred_key: str) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if preferred_key in h5:
            return h5[preferred_key][:]
        if len(h5.keys()) == 1:
            return h5[next(iter(h5.keys()))][:]
        raise KeyError(f"{path} must contain {preferred_key!r} or a single dataset")


def evaluate_prediction_file(
    prediction_path: str | Path,
    target_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    prediction = _read_dataset(Path(prediction_path), "prediction")
    target = _read_dataset(Path(target_path), "tensor")
    metrics = compute_task1_metrics(prediction, target)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task 1 predictions against validation targets.")
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    metrics = evaluate_prediction_file(args.prediction, args.target, args.output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
