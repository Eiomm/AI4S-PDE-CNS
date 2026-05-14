from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

from fno_inference import load_fno_checkpoint, run_autoregressive_inference  # noqa: E402
from agent.pde_finetune_data import spatial_indices  # noqa: E402


def _parse_weight(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise ValueError(f"expected NAME=WEIGHT, got {value!r}")
    name, raw_weight = value.split("=", 1)
    return name.strip(), float(raw_weight)


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return name.strip(), path


def _read_initial(path: Path, *, start: int, stop: int, spatial_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        tensor = h5["tensor"]
        indices = spatial_indices(source_size=int(tensor.shape[2]), target_size=spatial_size)
        initial = tensor[start:stop, :10, indices].astype(np.float32)
        if "x-coordinate" in h5:
            x_coords = h5["x-coordinate"][indices].astype(np.float32)
        else:
            x_coords = np.linspace(0.0, 1.0, spatial_size, endpoint=False, dtype=np.float32)
        if "t-coordinate" in h5:
            t_coords = h5["t-coordinate"][:200].astype(np.float32)
        else:
            t_coords = np.linspace(0.0, 1.0, 200, dtype=np.float32)
    return initial, x_coords, t_coords


def generate_cache(
    *,
    input_path: Path,
    output_path: Path,
    checkpoints: dict[str, Path],
    weights: dict[str, float],
    limit: int,
    chunk_size: int,
    batch_size: int,
    spatial_size: int,
    device_name: str,
) -> dict[str, object]:
    positive = {name: float(weight) for name, weight in weights.items() if float(weight) > 0.0}
    if not positive:
        raise ValueError("at least one positive weight is required")
    missing = [name for name in positive if name not in checkpoints]
    if missing:
        raise ValueError(f"missing checkpoints for weights: {missing}")
    total_weight = sum(positive.values())
    normalized = {name: weight / total_weight for name, weight in positive.items()}
    with h5py.File(input_path, "r") as h5:
        total_samples = int(h5["tensor"].shape[0])
    sample_count = min(int(limit), total_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    models = {name: load_fno_checkpoint(str(checkpoints[name]), device) for name in normalized}
    started = time.perf_counter()
    with h5py.File(output_path, "w") as h5:
        dataset = h5.create_dataset("prediction", shape=(sample_count, 200, spatial_size), dtype="float32")
        for start in range(0, sample_count, chunk_size):
            stop = min(start + chunk_size, sample_count)
            initial, x_coords, t_coords = _read_initial(input_path, start=start, stop=stop, spatial_size=spatial_size)
            combined = np.zeros((stop - start, 200, spatial_size), dtype=np.float32)
            for name, model in models.items():
                prediction = run_autoregressive_inference(
                    model,
                    initial,
                    x_coords,
                    t_coords,
                    device,
                    batch_size=batch_size,
                )
                combined += float(normalized[name]) * prediction.astype(np.float32)
            combined[:, :10, :] = initial
            dataset[start:stop] = combined
            print(json.dumps({"event": "chunk_done", "start": start, "stop": stop, "elapsed_seconds": time.perf_counter() - started}), flush=True)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "sample_count": sample_count,
        "spatial_size": spatial_size,
        "weights": normalized,
        "checkpoints": {name: str(path) for name, path in checkpoints.items() if name in normalized},
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FNO ensemble base prediction cache for Task 1 residual refiner training.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--weight", action="append", required=True, help="NAME=WEIGHT")
    parser.add_argument("--limit", type=int, default=4000)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--spatial-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    checkpoints = dict(_parse_checkpoint(value) for value in args.checkpoint)
    weights = dict(_parse_weight(value) for value in args.weight)
    summary = generate_cache(
        input_path=Path(args.input),
        output_path=Path(args.output),
        checkpoints=checkpoints,
        weights=weights,
        limit=args.limit,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        spatial_size=args.spatial_size,
        device_name=args.device,
    )
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
