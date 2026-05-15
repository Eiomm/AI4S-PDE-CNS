from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class HDF5WindowDatasetConfig:
    hdf5_path: Path
    initial_step: int = 10
    spatial_size: int = 256
    max_samples: int | None = None
    sample_start: int = 0
    max_time_steps: int | None = 200
    temporal_stride: int = 1


@dataclass(frozen=True)
class TrainingWindow:
    input_frames: np.ndarray
    target_frame: np.ndarray
    x_coords: np.ndarray
    t_coords: np.ndarray
    sample_index: int
    target_time_index: int


@dataclass(frozen=True)
class TrainingRolloutWindow:
    input_frames: np.ndarray
    target_frames: np.ndarray
    x_coords: np.ndarray
    t_coords: np.ndarray
    sample_index: int
    start_time_index: int


def spatial_indices(*, source_size: int, target_size: int) -> np.ndarray:
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if source_size < target_size:
        raise ValueError(f"source_size {source_size} must be >= target_size {target_size}")
    if source_size == target_size:
        return np.arange(source_size, dtype=np.int64)
    if source_size % target_size == 0:
        stride = source_size // target_size
        return np.arange(0, source_size, stride, dtype=np.int64)[:target_size]
    return np.linspace(0, source_size - 1, target_size, dtype=np.int64)


def _dataset_shape(path: Path) -> tuple[int, int, int]:
    with h5py.File(path, "r") as h5:
        if "tensor" not in h5:
            raise KeyError(f"{path} must contain a 'tensor' dataset")
        shape = tuple(h5["tensor"].shape)
    if len(shape) != 3:
        raise ValueError(f"tensor must have shape (N, T, X), got {shape}")
    return shape


def dataset_length(config: HDF5WindowDatasetConfig) -> int:
    samples, time_steps, _ = _dataset_shape(Path(config.hdf5_path))
    if config.temporal_stride <= 0:
        raise ValueError("temporal_stride must be positive")
    if config.sample_start < 0 or config.sample_start >= samples:
        raise ValueError(f"sample_start {config.sample_start} is outside sample count {samples}")
    available_samples = samples - config.sample_start
    sample_count = min(available_samples, config.max_samples) if config.max_samples is not None else available_samples
    usable_time_steps = min(time_steps, config.max_time_steps) if config.max_time_steps is not None else time_steps
    reduced_time_steps = ((usable_time_steps - 1) // config.temporal_stride) + 1
    windows_per_sample = reduced_time_steps - config.initial_step
    if windows_per_sample <= 0:
        raise ValueError("not enough time steps for the requested initial_step")
    return sample_count * windows_per_sample


def index_to_sample_and_target(config: HDF5WindowDatasetConfig, index: int) -> tuple[int, int]:
    _, time_steps, _ = _dataset_shape(Path(config.hdf5_path))
    usable_time_steps = min(time_steps, config.max_time_steps) if config.max_time_steps is not None else time_steps
    reduced_time_steps = ((usable_time_steps - 1) // config.temporal_stride) + 1
    windows_per_sample = reduced_time_steps - config.initial_step
    if index < 0 or index >= dataset_length(config):
        raise IndexError(index)
    sample_index = index // windows_per_sample
    sample_index += config.sample_start
    reduced_target_index = config.initial_step + (index % windows_per_sample)
    target_time_index = reduced_target_index * config.temporal_stride
    return int(sample_index), int(target_time_index)


def rollout_dataset_length(config: HDF5WindowDatasetConfig, rollout_steps: int) -> int:
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if config.temporal_stride <= 0:
        raise ValueError("temporal_stride must be positive")
    samples, time_steps, _ = _dataset_shape(Path(config.hdf5_path))
    if config.sample_start < 0 or config.sample_start >= samples:
        raise ValueError(f"sample_start {config.sample_start} is outside sample count {samples}")
    available_samples = samples - config.sample_start
    sample_count = min(available_samples, config.max_samples) if config.max_samples is not None else available_samples
    usable_time_steps = min(time_steps, config.max_time_steps) if config.max_time_steps is not None else time_steps
    reduced_time_steps = ((usable_time_steps - 1) // config.temporal_stride) + 1
    windows_per_sample = reduced_time_steps - config.initial_step - int(rollout_steps) + 1
    if windows_per_sample <= 0:
        raise ValueError("not enough time steps for the requested initial_step and rollout_steps")
    return sample_count * windows_per_sample


def index_to_sample_and_rollout_start(
    config: HDF5WindowDatasetConfig,
    index: int,
    *,
    rollout_steps: int,
) -> tuple[int, int]:
    _, time_steps, _ = _dataset_shape(Path(config.hdf5_path))
    usable_time_steps = min(time_steps, config.max_time_steps) if config.max_time_steps is not None else time_steps
    reduced_time_steps = ((usable_time_steps - 1) // config.temporal_stride) + 1
    windows_per_sample = reduced_time_steps - config.initial_step - int(rollout_steps) + 1
    if index < 0 or index >= rollout_dataset_length(config, rollout_steps):
        raise IndexError(index)
    sample_index = index // windows_per_sample
    sample_index += config.sample_start
    reduced_start_index = config.initial_step + (index % windows_per_sample)
    start_time_index = reduced_start_index * config.temporal_stride
    return int(sample_index), int(start_time_index)


def read_training_window(
    config: HDF5WindowDatasetConfig,
    *,
    sample_index: int,
    target_time_index: int,
) -> TrainingWindow:
    path = Path(config.hdf5_path)
    with h5py.File(path, "r") as h5:
        tensor = h5["tensor"]
        _, time_steps, source_size = tensor.shape
        if target_time_index < config.initial_step * config.temporal_stride or target_time_index >= time_steps:
            raise IndexError(target_time_index)
        indices = spatial_indices(source_size=source_size, target_size=config.spatial_size)
        input_time_indices = np.arange(
            target_time_index - config.initial_step * config.temporal_stride,
            target_time_index,
            config.temporal_stride,
            dtype=np.int64,
        )
        if input_time_indices[0] < 0 or input_time_indices.shape[0] != config.initial_step:
            raise IndexError(target_time_index)
        input_frames = tensor[sample_index, input_time_indices, :][:, indices]
        target_frame = tensor[sample_index, target_time_index, indices]
        if "x-coordinate" in h5:
            x_coords = h5["x-coordinate"][indices]
        else:
            x_coords = np.linspace(0.0, 1.0, config.spatial_size, endpoint=False, dtype=np.float32)
        if "t-coordinate" in h5:
            t_coords = h5["t-coordinate"][:]
        else:
            t_coords = np.arange(time_steps, dtype=np.float32)
    return TrainingWindow(
        input_frames=np.asarray(input_frames, dtype=np.float32),
        target_frame=np.asarray(target_frame, dtype=np.float32),
        x_coords=np.asarray(x_coords, dtype=np.float32),
        t_coords=np.asarray(t_coords, dtype=np.float32),
        sample_index=sample_index,
        target_time_index=target_time_index,
    )


def read_rollout_window(
    config: HDF5WindowDatasetConfig,
    *,
    sample_index: int,
    start_time_index: int,
    rollout_steps: int,
) -> TrainingRolloutWindow:
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    path = Path(config.hdf5_path)
    with h5py.File(path, "r") as h5:
        tensor = h5["tensor"]
        _, time_steps, source_size = tensor.shape
        end_time_index = start_time_index + int(rollout_steps)
        raw_end_time_index = start_time_index + int(rollout_steps) * config.temporal_stride
        if start_time_index < config.initial_step * config.temporal_stride or raw_end_time_index > time_steps:
            raise IndexError(start_time_index)
        indices = spatial_indices(source_size=source_size, target_size=config.spatial_size)
        input_time_indices = np.arange(
            start_time_index - config.initial_step * config.temporal_stride,
            start_time_index,
            config.temporal_stride,
            dtype=np.int64,
        )
        target_time_indices = np.arange(
            start_time_index,
            raw_end_time_index,
            config.temporal_stride,
            dtype=np.int64,
        )
        input_frames = tensor[sample_index, input_time_indices, :][:, indices]
        target_frames = tensor[sample_index, target_time_indices, :][:, indices]
        if "x-coordinate" in h5:
            x_coords = h5["x-coordinate"][indices]
        else:
            x_coords = np.linspace(0.0, 1.0, config.spatial_size, endpoint=False, dtype=np.float32)
        if "t-coordinate" in h5:
            t_coords = h5["t-coordinate"][:]
        else:
            t_coords = np.arange(time_steps, dtype=np.float32)
    return TrainingRolloutWindow(
        input_frames=np.asarray(input_frames, dtype=np.float32),
        target_frames=np.asarray(target_frames, dtype=np.float32),
        x_coords=np.asarray(x_coords, dtype=np.float32),
        t_coords=np.asarray(t_coords, dtype=np.float32),
        sample_index=sample_index,
        start_time_index=start_time_index,
    )
