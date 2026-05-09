from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def copy_initial_condition_baseline(input_path: Path, output_path: Path, dataset_key: str = "input") -> None:
    """Create a shape-correct placeholder by repeating the last initial frame.

    This is only a pipeline smoke-test baseline. It is not intended to be competitive.
    """
    with h5py.File(input_path, "r") as src:
        data = src[dataset_key][:]
    if data.ndim != 3 or data.shape[1] < 10 or data.shape[2] != 256:
        raise ValueError(f"Expected input shape (N, >=10, 256), got {data.shape}")
    pred = np.zeros((data.shape[0], 200, 256), dtype=np.float32)
    pred[:, :10, :] = data[:, :10, :]
    pred[:, 10:, :] = data[:, 9:10, :]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as dst:
        dst.create_dataset("prediction", data=pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a non-competitive shape smoke-test baseline.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-key", default="input")
    args = parser.parse_args()
    copy_initial_condition_baseline(Path(args.input), Path(args.output), args.dataset_key)


if __name__ == "__main__":
    main()
