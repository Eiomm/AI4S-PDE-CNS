"""Task 2 persistence baseline.

The official sample submission keeps the first 10 frames equal to task2_test.h5.
This baseline repeats the last observed frame for the remaining 190 frames.
It is a correctness scaffold, not a competitive Task 2 model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def read_tensor(path: str | Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if "tensor" not in h5:
            raise KeyError(f"{path} must contain a 'tensor' dataset")
        return h5["tensor"][:].astype(np.float32)


def persistence_prediction(initial: np.ndarray, *, output_steps: int = 200) -> np.ndarray:
    initial = np.asarray(initial, dtype=np.float32)
    if initial.ndim != 3 or initial.shape[1] != 10 or initial.shape[2] != 256:
        raise ValueError(f"Task 2 initial tensor must have shape (N, 10, 256), got {initial.shape}")
    prediction = np.zeros((initial.shape[0], output_steps, initial.shape[2]), dtype=np.float32)
    prediction[:, :10, :] = initial
    prediction[:, 10:, :] = initial[:, -1:, :]
    return prediction


def write_prediction(path: str | Path, prediction: np.ndarray, *, dataset_key: str = "prediction") -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with h5py.File(output, "w") as h5:
        h5.create_dataset(dataset_key, data=np.asarray(prediction, dtype=np.float32))
    return output


def run_task2_persistence(input_path: str | Path, output_path: str | Path, *, output_steps: int = 200) -> Path:
    initial = read_tensor(input_path)
    prediction = persistence_prediction(initial, output_steps=output_steps)
    return write_prediction(output_path, prediction)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 2 persistence baseline.")
    parser.add_argument("--input", default="data/Task2/task2_test.h5")
    parser.add_argument("--output", default="runs/task2-persistence/task2_pred.hdf5")
    parser.add_argument("--output-steps", type=int, default=200)
    args = parser.parse_args()
    print(run_task2_persistence(args.input, args.output, output_steps=args.output_steps))


if __name__ == "__main__":
    main()
