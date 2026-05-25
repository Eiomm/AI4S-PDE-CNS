from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def read_named_or_single(path: str | Path, preferred_key: str) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if preferred_key in h5:
            return h5[preferred_key][:]
        if "tensor" in h5:
            return h5["tensor"][:]
        if "prediction" in h5:
            return h5["prediction"][:]
        keys = list(h5.keys())
        if len(keys) == 1:
            return h5[keys[0]][:]
    raise KeyError(f"{path} must contain {preferred_key!r}, tensor, prediction, or exactly one dataset")


def read_task1_input(path: str | Path, *, full_t_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    input_path = Path(path)
    with h5py.File(input_path, "r") as h5:
        if "tensor" not in h5:
            raise KeyError(f"{input_path} must contain tensor")
        tensor = h5["tensor"][:].astype(np.float32)
        x_coords = h5["x-coordinate"][:] if "x-coordinate" in h5 else np.linspace(0.0, 1.0, tensor.shape[-1], dtype=np.float32)
        t_raw = h5["t-coordinate"][:] if "t-coordinate" in h5 else np.arange(tensor.shape[1], dtype=np.float32)

    if tensor.ndim != 3 or tensor.shape[1] < 10 or tensor.shape[2] != 256:
        raise ValueError(f"expected input tensor shape (N, >=10, 256), got {tensor.shape}")

    target = tensor if tensor.shape[1:] == (200, 256) else None
    t_coords = None
    if full_t_path is not None and Path(full_t_path).is_file():
        with h5py.File(full_t_path, "r") as h5:
            if "t-coordinate" in h5 and len(h5["t-coordinate"]) == 200:
                t_coords = h5["t-coordinate"][:]
    if t_coords is None and len(t_raw) == 200:
        t_coords = t_raw
    if t_coords is None:
        t_coords = np.linspace(float(t_raw[0]), float(t_raw[-1]) * 20.0, 200, dtype=np.float32)

    return tensor[:, :10, :].astype(np.float32), x_coords.astype(np.float32), t_coords.astype(np.float32), target


def write_prediction(path: str | Path, prediction: np.ndarray, *, dataset_key: str = "tensor") -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    if temp.exists():
        temp.unlink()
    data = np.asarray(prediction, dtype=np.float32)
    chunks = (min(10, data.shape[0]), data.shape[1], data.shape[2])
    with h5py.File(temp, "w") as h5:
        h5.create_dataset(
            dataset_key,
            data=data,
            compression="gzip",
            compression_opts=6,
            shuffle=True,
            chunks=chunks,
        )
    temp.replace(output)
