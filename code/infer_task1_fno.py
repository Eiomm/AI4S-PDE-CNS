from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from model_task1_fno import load_task1_fno, rollout


def _read_tensor(path: Path, limit: int | None) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if "tensor" not in h5:
            raise KeyError(f"{path} must contain a 'tensor' dataset")
        data = h5["tensor"]
        if data.ndim != 3 or data.shape[1] < 10 or data.shape[2] != 256:
            raise ValueError(f"expected tensor shape (N, >=10, 256), got {data.shape}")
        stop = data.shape[0] if limit is None else min(limit, data.shape[0])
        return data[:stop, :10, :].astype(np.float32)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_inference(
    input_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    *,
    batch_size: int = 64,
    device_name: str = "auto",
    limit: int | None = None,
) -> float:
    started = time.perf_counter()
    initial = _read_tensor(input_path, limit)
    device = _device(device_name)
    model = load_task1_fno(checkpoint_path, device=device)
    predictions: list[np.ndarray] = []
    for start in range(0, initial.shape[0], batch_size):
        batch = torch.from_numpy(initial[start : start + batch_size]).to(device)
        pred = rollout(model, batch, total_steps=200)
        predictions.append(pred.cpu().numpy().astype(np.float32))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("prediction", data=np.concatenate(predictions, axis=0))
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 1 FNO checkpoint inference.")
    parser.add_argument("--input", required=True, help="Path to task1_test.hdf5 or task1_val.hdf5.")
    parser.add_argument("--checkpoint", required=True, help="Path to 1D_Burgers_Sols_Nu0.001_FNO.pt.")
    parser.add_argument("--output", required=True, help="Output HDF5 path with a prediction dataset.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    args = parser.parse_args()

    elapsed = run_inference(
        Path(args.input),
        Path(args.checkpoint),
        Path(args.output),
        batch_size=args.batch_size,
        device_name=args.device,
        limit=args.limit,
    )
    print(f"inference_seconds={elapsed:.6f}")


if __name__ == "__main__":
    main()
