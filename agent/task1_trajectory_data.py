from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .pde_finetune_data import spatial_indices


@dataclass(frozen=True)
class Task1TrajectoryConfig:
    hdf5_paths: list[Path]
    spatial_size: int = 256
    initial_step: int = 10
    output_steps: int = 200
    max_samples_per_file: int | None = None
    sample_start: int = 0


@dataclass(frozen=True)
class Task1TrajectorySample:
    initial: np.ndarray
    target: np.ndarray
    x_coords: np.ndarray
    t_coords: np.ndarray
    source_path: Path
    sample_index: int
    nu: float | None


def parse_nu_from_path(path: str | Path) -> float | None:
    match = re.search(r"Nu([0-9.]+)", Path(path).name)
    if not match:
        return None
    try:
        return float(match.group(1).rstrip("."))
    except ValueError:
        return None


def _sample_count(path: Path, config: Task1TrajectoryConfig) -> int:
    with h5py.File(path, "r") as h5:
        if "tensor" not in h5:
            raise KeyError(f"{path} must contain a 'tensor' dataset")
        samples = int(h5["tensor"].shape[0])
    if config.sample_start < 0 or config.sample_start >= samples:
        raise ValueError(f"sample_start {config.sample_start} is outside sample count {samples}")
    available = samples - config.sample_start
    return min(available, config.max_samples_per_file) if config.max_samples_per_file is not None else available


def task1_trajectory_length(config: Task1TrajectoryConfig) -> int:
    if not config.hdf5_paths:
        return 0
    return sum(_sample_count(Path(path), config) for path in config.hdf5_paths)


def _locate(config: Task1TrajectoryConfig, index: int) -> tuple[Path, int]:
    if index < 0:
        raise IndexError(index)
    offset = int(index)
    for raw_path in config.hdf5_paths:
        path = Path(raw_path)
        count = _sample_count(path, config)
        if offset < count:
            return path, config.sample_start + offset
        offset -= count
    raise IndexError(index)


def read_task1_trajectory_sample(config: Task1TrajectoryConfig, index: int) -> Task1TrajectorySample:
    path, sample_index = _locate(config, index)
    with h5py.File(path, "r") as h5:
        tensor = h5["tensor"]
        _, time_steps, source_size = tensor.shape
        if time_steps < config.output_steps:
            raise ValueError(f"{path} has only {time_steps} time steps, need {config.output_steps}")
        indices = spatial_indices(source_size=source_size, target_size=config.spatial_size)
        target = tensor[sample_index, : config.output_steps, indices]
        if "x-coordinate" in h5:
            x_coords = h5["x-coordinate"][indices]
        else:
            x_coords = np.linspace(0.0, 1.0, config.spatial_size, endpoint=False, dtype=np.float32)
        if "t-coordinate" in h5:
            t_coords = h5["t-coordinate"][: config.output_steps]
        else:
            t_coords = np.linspace(0.0, 1.0, config.output_steps, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    return Task1TrajectorySample(
        initial=target[: config.initial_step].copy(),
        target=target,
        x_coords=np.asarray(x_coords, dtype=np.float32),
        t_coords=np.asarray(t_coords, dtype=np.float32),
        source_path=path,
        sample_index=int(sample_index),
        nu=parse_nu_from_path(path),
    )
