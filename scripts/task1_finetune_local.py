#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai4sv2_task1.hdf5_io import read_task1_input
from ai4sv2_task1.metrics import compute_task1_metrics
from ai4sv2_task1.models.fno import FNO1d, load_fno_checkpoint, rollout_fno
from ai4sv2_task1.paths import resolve_path

RUN_ID_PATTERN = re.compile(r"^(?:agent_)?\d{8}T\d{6}\d{6}Z(?:__tool\d{2})?$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def canonical_run_dir(raw_run_dir: str | None) -> str:
    if not raw_run_dir:
        return f"runs/task1/{utc_run_id()}"
    path = Path(raw_run_dir)
    parts = path.parts
    if not path.is_absolute() and len(parts) >= 3 and parts[0] == "runs" and parts[1] == "task1":
        if len(parts) != 3:
            return f"runs/task1/{utc_run_id()}"
        if RUN_ID_PATTERN.fullmatch(parts[2]):
            return path.as_posix()
        return f"runs/task1/{utc_run_id()}"
    if path.is_absolute():
        for index in range(len(parts) - 1):
            if parts[index] == "runs" and parts[index + 1] == "task1":
                task1_runs = Path(*parts[: index + 2])
                if len(parts) == index + 3 and RUN_ID_PATTERN.fullmatch(parts[index + 2]):
                    return str(path)
                return str(task1_runs / utc_run_id())
    return raw_run_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_time_csv(path: Path, *, train_time: float, inference_time: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["train_time", "inference_time"])
        writer.writeheader()
        writer.writerow({"train_time": f"{float(train_time):.6f}", "inference_time": f"{float(inference_time):.6f}"})


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def spatial_indices(source_size: int, target_size: int, downsample: int) -> np.ndarray:
    if source_size == target_size:
        return np.arange(target_size, dtype=np.int64)
    expected = source_size // downsample
    if expected != target_size:
        raise ValueError(f"spatial scale mismatch: source={source_size}, downsample={downsample}, target={target_size}")
    indices = np.arange(0, source_size, downsample, dtype=np.int64)
    if len(indices) != target_size:
        raise ValueError(f"bad spatial indices length: {len(indices)} != {target_size}")
    return indices


class ReducedBurgersWindowDataset(Dataset):
    """Raw PDEBench -> official Task1 reduced-scale training windows.

    Official Task1 checkpoints were trained with reduced_resolution_t=5 and
    reduced_resolution=4. This dataset enforces that alignment so one model
    training step corresponds to five raw PDEBench time indices.
    """

    def __init__(
        self,
        hdf5_path: Path,
        *,
        initial_step: int = 10,
        rollout_steps: int = 1,
        temporal_stride: int = 5,
        spatial_downsample: int = 4,
        spatial_size: int = 256,
        max_samples: int | None = None,
        sample_start: int = 0,
        late_window_prob: float = 0.0,
        late_window_fraction: float = 1.0 / 3.0,
    ):
        if temporal_stride != 5:
            raise ValueError("Task1 official-checkpoint fine-tune must use temporal_stride=5")
        if spatial_downsample != 4:
            raise ValueError("Task1 official-checkpoint fine-tune must use spatial_downsample=4")
        if rollout_steps < 1:
            raise ValueError("rollout_steps must be positive")
        self.hdf5_path = Path(hdf5_path)
        self.initial_step = int(initial_step)
        self.rollout_steps = int(rollout_steps)
        self.temporal_stride = int(temporal_stride)
        self.spatial_downsample = int(spatial_downsample)
        self.spatial_size = int(spatial_size)
        self.sample_start = int(sample_start)
        self.late_window_prob = float(late_window_prob)
        self.late_window_fraction = float(late_window_fraction)
        self._h5: h5py.File | None = None
        self._tensor: h5py.Dataset | None = None

        with h5py.File(self.hdf5_path, "r") as h5:
            tensor = h5["tensor"]
            self.total_samples = int(tensor.shape[0])
            self.raw_time_steps = int(tensor.shape[1])
            self.raw_spatial_size = int(tensor.shape[2])
            x_coord = np.asarray(h5["x-coordinate"], dtype=np.float64)
            t_coord = np.asarray(h5["t-coordinate"], dtype=np.float64)
        if self.sample_start < 0 or self.sample_start >= self.total_samples:
            raise ValueError(f"invalid sample_start={self.sample_start}")
        if not 0.0 <= self.late_window_prob <= 1.0:
            raise ValueError("late_window_prob must be in [0, 1]")
        if not 0.0 < self.late_window_fraction <= 1.0:
            raise ValueError("late_window_fraction must be in (0, 1]")
        available_samples = self.total_samples - self.sample_start
        self.num_samples = min(available_samples, int(max_samples)) if max_samples is not None else available_samples
        self.space_idx = spatial_indices(self.raw_spatial_size, self.spatial_size, self.spatial_downsample)
        self.reduced_time_indices = np.arange(0, self.raw_time_steps, self.temporal_stride, dtype=np.int64)
        self.reduced_x_coords = x_coord[self.space_idx]
        self.reduced_t_coords = t_coord[self.reduced_time_indices]
        self.dx = float(np.median(np.diff(self.reduced_x_coords)))
        self.dt = float(np.median(np.diff(self.reduced_t_coords)))
        self.windows_per_sample = len(self.reduced_time_indices) - self.initial_step - self.rollout_steps + 1
        if self.windows_per_sample <= 0:
            raise ValueError(
                "not enough reduced time steps: "
                f"raw_time={self.raw_time_steps}, stride={self.temporal_stride}, initial={self.initial_step}, rollout={self.rollout_steps}"
            )

    def __len__(self) -> int:
        return self.num_samples * self.windows_per_sample

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_tensor"] = None
        return state

    def _get_tensor(self) -> h5py.Dataset:
        if self._h5 is None or self._tensor is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
            self._tensor = self._h5["tensor"]
        return self._tensor

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._tensor = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_offset = index // self.windows_per_sample
        window_start = index % self.windows_per_sample
        if self.late_window_prob > 0.0 and np.random.random() < self.late_window_prob:
            late_count = max(1, int(math.ceil(self.windows_per_sample * self.late_window_fraction)))
            late_start = max(0, self.windows_per_sample - late_count)
            window_start = int(np.random.randint(late_start, self.windows_per_sample))
        sample_index = self.sample_start + sample_offset
        reduced_slice = self.reduced_time_indices[window_start : window_start + self.initial_step + self.rollout_steps]
        raw = self._get_tensor()[sample_index, reduced_slice, :]
        reduced = raw[:, self.space_idx].astype(np.float32)
        inputs = reduced[: self.initial_step]
        targets = reduced[self.initial_step : self.initial_step + self.rollout_steps]
        return torch.from_numpy(inputs), torch.from_numpy(targets)

    def metadata(self) -> dict[str, Any]:
        return {
            "hdf5_path": str(self.hdf5_path),
            "total_samples": self.total_samples,
            "num_samples": self.num_samples,
            "sample_start": self.sample_start,
            "raw_time_steps": self.raw_time_steps,
            "raw_spatial_size": self.raw_spatial_size,
            "reduced_time_steps": int(len(self.reduced_time_indices)),
            "windows_per_sample": int(self.windows_per_sample),
            "initial_step": self.initial_step,
            "rollout_steps": self.rollout_steps,
            "temporal_stride": self.temporal_stride,
            "spatial_downsample": self.spatial_downsample,
            "spatial_size": self.spatial_size,
            "dx": self.dx,
            "dt": self.dt,
            "late_window_prob": self.late_window_prob,
            "late_window_fraction": self.late_window_fraction,
            "first_observed_raw_indices": self.reduced_time_indices[: self.initial_step].tolist(),
            "first_supervised_target_raw_index": int(self.reduced_time_indices[self.initial_step]),
        }


def grid_for_batch(features: torch.Tensor) -> torch.Tensor:
    grid = torch.linspace(0.0, 1.0, features.shape[1], device=features.device, dtype=features.dtype)
    return grid.view(1, features.shape[1], 1).expand(features.shape[0], -1, -1)


def predict_next(model: FNO1d, window: torch.Tensor) -> torch.Tensor:
    features = window.permute(0, 2, 1).contiguous()
    return model(features, grid_for_batch(features)).squeeze(-1).squeeze(-1)


def parse_horizon_weights(raw: str | None, rollout_steps: int) -> list[float] | None:
    if not raw:
        return None
    weights = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(weights) != rollout_steps:
        raise ValueError(f"--horizon-weights must contain {rollout_steps} values, got {len(weights)}")
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("--horizon-weights must be positive")
    return weights


def default_horizon_weights(mode: str, rollout_steps: int, gamma: float, explicit: list[float] | None) -> torch.Tensor | None:
    if explicit is not None:
        return torch.tensor(explicit, dtype=torch.float32)
    if mode == "gamma":
        return None
    if mode == "uniform":
        return torch.ones(rollout_steps, dtype=torch.float32)
    if mode == "official":
        if rollout_steps == 1:
            return torch.ones(1, dtype=torch.float32)
        return torch.linspace(0.6, 1.5, rollout_steps, dtype=torch.float32)
    raise ValueError(f"unsupported horizon_weight_mode={mode!r}")


def burgers_residual_loss(seq: torch.Tensor, *, dx: float, dt: float, nu: float, eps: float = 1.0e-8) -> torch.Tensor:
    ux = (torch.roll(seq, shifts=-1, dims=-1) - torch.roll(seq, shifts=1, dims=-1)) / (2.0 * dx)
    uxx = (torch.roll(seq, shifts=-1, dims=-1) - 2.0 * seq + torch.roll(seq, shifts=1, dims=-1)) / (dx * dx)
    ut = (seq[:, 1:, :] - seq[:, :-1, :]) / dt
    conv = seq[:, :-1, :] * ux[:, :-1, :]
    diff = nu * uxx[:, :-1, :]
    residual = ut + conv - diff
    denom = ut.pow(2).mean() + conv.pow(2).mean() + diff.pow(2).mean() + eps
    return residual.pow(2).mean() / denom.detach()


def rollout_loss(
    model: FNO1d,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float,
    horizon_weights: torch.Tensor | None = None,
    physics_weight: float = 0.0,
    physics_dx: float = 1.0,
    physics_dt: float = 1.0,
    physics_nu: float = 1.0e-3,
) -> tuple[torch.Tensor, dict[str, float]]:
    current = inputs
    losses = []
    predictions = []
    for step in range(targets.shape[1]):
        prediction = predict_next(model, current)
        predictions.append(prediction)
        target = targets[:, step, :]
        losses.append(torch.mean((prediction - target) ** 2))
        current = torch.cat([current[:, 1:, :], prediction.unsqueeze(1)], dim=1)
    step_losses = torch.stack(losses)
    if horizon_weights is None:
        mse_loss = torch.stack([(float(gamma) ** step) * loss for step, loss in enumerate(step_losses)]).mean()
    else:
        weights = horizon_weights.to(device=step_losses.device, dtype=step_losses.dtype)
        mse_loss = (step_losses * weights).sum() / weights.sum()
    phys_loss = torch.zeros((), device=inputs.device, dtype=inputs.dtype)
    if physics_weight > 0.0:
        pred_stack = torch.stack(predictions, dim=1)
        physics_seq = torch.cat([inputs[:, -1:, :], pred_stack], dim=1)
        phys_loss = burgers_residual_loss(physics_seq, dx=physics_dx, dt=physics_dt, nu=physics_nu)
    total_loss = mse_loss + float(physics_weight) * phys_loss
    return total_loss, {
        "mse_loss": float(mse_loss.detach().cpu()),
        "physics_loss": float(phys_loss.detach().cpu()),
        "physics_weight": float(physics_weight),
    }


TRAINABLE_MODULES = ("fc0", "conv0", "w0", "conv1", "w1", "conv2", "w2", "conv3", "w3", "fc1", "fc2")


def parse_trainable_modules(raw_modules: list[str] | None) -> list[str]:
    modules: list[str] = []
    for raw in raw_modules or []:
        for item in str(raw).split(","):
            module = item.strip()
            if module:
                modules.append(module)
    unknown = sorted(set(modules) - set(TRAINABLE_MODULES))
    if unknown:
        raise ValueError(f"unknown trainable modules: {unknown}; allowed={list(TRAINABLE_MODULES)}")
    return list(dict.fromkeys(modules))


def preset_trainable_modules(mode: str) -> list[str] | None:
    if mode == "head":
        return ["fc1", "fc2"]
    if mode == "last-block-head":
        return ["conv3", "w3", "fc1", "fc2"]
    if mode == "all":
        return None
    if mode == "custom":
        return []
    raise ValueError(f"unsupported trainable mode: {mode}")


def set_trainable(model: FNO1d, mode: str, modules: list[str] | None = None) -> list[str]:
    selected_modules = parse_trainable_modules(modules)
    if selected_modules:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for module_name in selected_modules:
            for parameter in getattr(model, module_name).parameters():
                parameter.requires_grad = True
    elif mode == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        preset_modules = preset_trainable_modules(mode)
        if not preset_modules:
            raise ValueError("custom trainable mode requires at least one --trainable-module")
        for parameter in model.parameters():
            parameter.requires_grad = False
        for module_name in preset_modules:
            for parameter in getattr(model, module_name).parameters():
                parameter.requires_grad = True
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


@torch.no_grad()
def evaluate(model: FNO1d, val_hdf5: Path, device: torch.device, *, batch_size: int, max_samples: int | None) -> dict[str, float]:
    initial, x_coords, t_coords, target = read_task1_input(val_hdf5, full_t_path=val_hdf5)
    if target is None:
        raise ValueError("validation hdf5 must contain full target tensor")
    if max_samples is not None:
        initial = initial[:max_samples]
        target = target[:max_samples]
    prediction = rollout_fno(model, initial, x_coords, t_coords, device, batch_size)
    return {key: float(value) for key, value in compute_task1_metrics(prediction, target).items()}


def is_better(candidate: dict[str, float], best: dict[str, float] | None, metric: str, direction: str, min_delta: float) -> bool:
    if best is None:
        return True
    value = float(candidate[metric])
    best_value = float(best[metric])
    if direction == "max":
        return value > best_value + min_delta
    return value < best_value - min_delta


def clone_state_dict(model: FNO1d) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


@torch.no_grad()
def update_ema_state(ema_state: dict[str, torch.Tensor], model: FNO1d, decay: float) -> None:
    for name, value in model.state_dict().items():
        source = value.detach()
        if torch.is_floating_point(source) or torch.is_complex(source):
            ema_state[name].mul_(float(decay)).add_(source, alpha=1.0 - float(decay))
        else:
            ema_state[name].copy_(source)


def save_checkpoint(
    path: Path,
    model: FNO1d,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    metrics: dict[str, float] | None,
    model_state_dict: dict[str, torch.Tensor] | None = None,
    source: str = "raw",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model_state_dict or model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": int(step),
            "metrics": metrics or {},
            "checkpoint_source": source,
            "scale_alignment": {
                "temporal_stride": 5,
                "spatial_downsample": 4,
                "model_step_meaning": "one model step equals 5 raw PDEBench time indices",
            },
        },
        path,
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    requested_run_dir = args.run_dir
    run_dir_arg = canonical_run_dir(args.run_dir)
    run_dir = resolve_path(run_dir_arg) if not Path(run_dir_arg).is_absolute() else Path(run_dir_arg)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "finetune_train.jsonl"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    dataset = ReducedBurgersWindowDataset(
        resolve_path(args.train_hdf5),
        initial_step=args.initial_step,
        rollout_steps=args.rollout_steps,
        temporal_stride=args.temporal_stride,
        spatial_downsample=args.spatial_downsample,
        spatial_size=args.spatial_size,
        max_samples=args.max_samples,
        sample_start=args.sample_start,
        late_window_prob=args.late_window_prob,
        late_window_fraction=args.late_window_fraction,
    )
    explicit_horizon_weights = parse_horizon_weights(args.horizon_weights, args.rollout_steps)
    horizon_weights = default_horizon_weights(args.horizon_weight_mode, args.rollout_steps, args.horizon_gamma, explicit_horizon_weights)
    physics_dx = args.physics_dx if args.physics_dx is not None else dataset.dx
    physics_dt = args.physics_dt if args.physics_dt is not None else dataset.dt
    pin_memory = args.pin_memory == "true" or (args.pin_memory == "auto" and device.type == "cuda")
    persistent_workers = args.persistent_workers == "true" or (args.persistent_workers == "auto" and args.num_workers > 0)
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "drop_last": True,
        "generator": torch.Generator().manual_seed(args.seed),
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
    model = load_fno_checkpoint(resolve_path(args.base_checkpoint), device)
    trainable_modules = parse_trainable_modules(args.trainable_module)
    trainable_names = set_trainable(model, args.trainable, trainable_modules)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in [0, 1)")
    ema_state = clone_state_dict(model) if args.ema_decay > 0.0 else None
    best_metrics: dict[str, float] | None = None
    base_metrics: dict[str, float] | None = None
    best_source = "raw"
    history: list[dict[str, Any]] = []
    start = time.perf_counter()

    run_meta = {
        "timestamp": utc_now(),
        "event": "finetune_start",
        "run_dir": str(run_dir),
        "requested_run_dir": requested_run_dir,
        "device": str(device),
        "base_checkpoint": str(resolve_path(args.base_checkpoint)),
        "dataset": dataset.metadata(),
        "trainable": args.trainable,
        "trainable_modules": trainable_modules or preset_trainable_modules(args.trainable),
        "trainable_parameter_names": trainable_names,
        "dataloader": {
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "pin_memory": bool(pin_memory),
            "persistent_workers": bool(persistent_workers),
            "prefetch_factor": int(args.prefetch_factor) if args.num_workers > 0 else None,
        },
        "args": vars(args),
        "loss_config": {
            "horizon_weight_mode": args.horizon_weight_mode,
            "horizon_weights": explicit_horizon_weights if explicit_horizon_weights is not None else (horizon_weights.tolist() if horizon_weights is not None else None),
            "physics_weight": float(args.physics_weight),
            "physics_dx": float(physics_dx),
            "physics_dt": float(physics_dt),
            "physics_nu": float(args.physics_nu),
            "ema_decay": float(args.ema_decay),
        },
    }
    append_jsonl(log_path, run_meta)

    if not args.skip_base_eval:
        model.eval()
        base_metrics = evaluate(model, resolve_path(args.val_hdf5), device, batch_size=args.eval_batch_size, max_samples=args.val_max_samples)
        best_metrics = dict(base_metrics)
        save_checkpoint(run_dir / "best.pt", model, optimizer, step=0, metrics=best_metrics)
        record = {"timestamp": utc_now(), "event": "base_eval", "step": 0, "metrics": base_metrics}
        history.append(record)
        append_jsonl(log_path, record)

    step = 0
    model.train()
    while step < args.steps:
        for inputs, targets in loader:
            step += 1
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
            targets = targets.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
            loss, loss_parts = rollout_loss(
                model,
                inputs,
                targets,
                gamma=args.horizon_gamma,
                horizon_weights=horizon_weights,
                physics_weight=args.physics_weight,
                physics_dx=physics_dx,
                physics_dt=physics_dt,
                physics_nu=args.physics_nu,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if ema_state is not None:
                update_ema_state(ema_state, model, args.ema_decay)

            if step == 1 or step % args.log_every == 0:
                record = {
                    "timestamp": utc_now(),
                    "event": "train_step",
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    **loss_parts,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                history.append(record)
                append_jsonl(log_path, record)

            if step == args.steps or step % args.val_every == 0:
                model.eval()
                metrics = evaluate(model, resolve_path(args.val_hdf5), device, batch_size=args.eval_batch_size, max_samples=args.val_max_samples)
                improved = is_better(metrics, best_metrics, args.selection_metric, args.selection_direction, args.min_delta)
                record = {
                    "timestamp": utc_now(),
                    "event": "validation",
                    "step": step,
                    "source": "raw",
                    "improved": improved,
                    "metrics": metrics,
                }
                history.append(record)
                append_jsonl(log_path, record)
                if improved:
                    best_metrics = metrics
                    best_source = "raw"
                    save_checkpoint(run_dir / "best.pt", model, optimizer, step=step, metrics=best_metrics, source=best_source)
                if ema_state is not None:
                    raw_state = clone_state_dict(model)
                    model.load_state_dict(ema_state)
                    ema_metrics = evaluate(model, resolve_path(args.val_hdf5), device, batch_size=args.eval_batch_size, max_samples=args.val_max_samples)
                    model.load_state_dict(raw_state)
                    ema_improved = is_better(ema_metrics, best_metrics, args.selection_metric, args.selection_direction, args.min_delta)
                    ema_record = {
                        "timestamp": utc_now(),
                        "event": "validation",
                        "step": step,
                        "source": "ema",
                        "ema_decay": float(args.ema_decay),
                        "improved": ema_improved,
                        "metrics": ema_metrics,
                    }
                    history.append(ema_record)
                    append_jsonl(log_path, ema_record)
                    if ema_improved:
                        best_metrics = ema_metrics
                        best_source = "ema"
                        save_checkpoint(
                            run_dir / "best.pt",
                            model,
                            optimizer,
                            step=step,
                            metrics=best_metrics,
                            model_state_dict=ema_state,
                            source=best_source,
                        )
                model.train()

            if step >= args.steps:
                break

    elapsed = time.perf_counter() - start
    save_checkpoint(run_dir / "last.pt", model, optimizer, step=step, metrics=best_metrics)
    if ema_state is not None:
        save_checkpoint(run_dir / "ema_last.pt", model, optimizer, step=step, metrics=best_metrics, model_state_dict=ema_state, source="ema")
    time_path = run_dir / "task1_time.csv"
    write_time_csv(time_path, train_time=elapsed)
    result = {
        "success": True,
        "run_dir": str(run_dir),
        "requested_run_dir": requested_run_dir,
        "elapsed_seconds": elapsed,
        "steps": step,
        "base_metrics": base_metrics,
        "best_metrics": best_metrics,
        "best_source": best_source,
        "history": history,
        "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
        "log_path": str(log_path),
        "time_path": str(time_path),
        "scale_alignment": dataset.metadata(),
        "loss_config": {
            "horizon_weight_mode": args.horizon_weight_mode,
            "horizon_weights": explicit_horizon_weights if explicit_horizon_weights is not None else (horizon_weights.tolist() if horizon_weights is not None else None),
            "physics_weight": float(args.physics_weight),
            "physics_dx": float(physics_dx),
            "physics_dt": float(physics_dt),
            "physics_nu": float(args.physics_nu),
            "ema_decay": float(args.ema_decay),
        },
    }
    write_json(run_dir / "finetune_result.json", result)
    checkpoint_records = [
        {
            "kind": "fno_base",
            "path": str(resolve_path(args.base_checkpoint)),
            "sha256": file_sha256(resolve_path(args.base_checkpoint)),
            "weight": 1.0,
        },
        {
            "kind": "fno_finetuned_best",
            "path": str(run_dir / "best.pt"),
            "sha256": file_sha256(run_dir / "best.pt"),
            "weight": 1.0,
        },
        {
            "kind": "fno_finetuned_last",
            "path": str(run_dir / "last.pt"),
            "sha256": file_sha256(run_dir / "last.pt"),
            "weight": 1.0,
        },
    ]
    if ema_state is not None:
        checkpoint_records.append(
            {
                "kind": "fno_finetuned_ema_last",
                "path": str(run_dir / "ema_last.pt"),
                "sha256": file_sha256(run_dir / "ema_last.pt"),
                "weight": 1.0,
            }
        )
    metadata = {
        "task": "task1",
        "route": "finetune_fno",
        "split": "train_val",
        "run_name": run_dir.name,
        "requested_run_name": requested_run_dir,
        "run_dir": str(run_dir),
        "config_path": "finetune_local_tool",
        "input_path": str(resolve_path(args.train_hdf5)),
        "prediction_path": None,
        "metrics_path": None,
        "time_path": str(time_path),
        "log_path": str(log_path),
        "checkpoints": checkpoint_records,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers),
        "prefetch_factor": int(args.prefetch_factor) if args.num_workers > 0 else None,
        "device": str(device),
        "train_time": float(elapsed),
        "inference_time": 0.0,
        "validation": {
            "shape": None,
            "first_ten_match": None,
            "finite": True,
            "max_initial_error": None,
        },
        "metrics": best_metrics or {},
        "best_source": best_source,
        "base_metrics": base_metrics or {},
        "scale_alignment": dataset.metadata(),
        "trainable": args.trainable,
        "trainable_modules": trainable_modules or preset_trainable_modules(args.trainable),
        "trainable_parameter_names": trainable_names,
        "steps": int(step),
        "loss_config": result["loss_config"],
    }
    write_json(run_dir / "metadata.json", metadata)
    append_jsonl(log_path, {"timestamp": utc_now(), "event": "finetune_done", "elapsed_seconds": elapsed, "best_metrics": best_metrics})
    dataset.close()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Task1 FNO fine-tune with official reduced-scale alignment.")
    parser.add_argument("--train-hdf5", default="data/pdebench_burgers/raw/1D_Burgers_Sols_Nu0.001.hdf5")
    parser.add_argument("--val-hdf5", default="data/task1_val.hdf5")
    parser.add_argument("--base-checkpoint", default="checkpoints/official/nu0.001_fno.pt")
    parser.add_argument("--run-dir", default=None, help="默认写入 runs/task1/<UTC timestamp>；Agent loop 会显式传入当前 experiment/runs/<tool>。")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--initial-step", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--horizon-gamma", type=float, default=0.9)
    parser.add_argument(
        "--horizon-weight-mode",
        choices=["gamma", "uniform", "official"],
        default="gamma",
        help="gamma 保持原行为；official 使用后几步更高的归一化权重。",
    )
    parser.add_argument("--horizon-weights", default=None, help="逗号分隔的显式 horizon 权重，长度必须等于 --rollout-steps。")
    parser.add_argument("--physics-weight", type=float, default=0.0, help="tiny normalized Burgers residual weight；0 表示关闭。")
    parser.add_argument("--physics-nu", type=float, default=1.0e-3)
    parser.add_argument("--physics-dx", type=float, default=None)
    parser.add_argument("--physics-dt", type=float, default=None)
    parser.add_argument("--ema-decay", type=float, default=0.0, help="0 关闭 EMA；常用 0.995 或 0.999。")
    parser.add_argument("--temporal-stride", type=int, default=5)
    parser.add_argument("--spatial-downsample", type=int, default=4)
    parser.add_argument("--spatial-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--late-window-prob", type=float, default=0.0, help="每个样本重采样到 late window 的概率。")
    parser.add_argument("--late-window-fraction", type=float, default=1.0 / 3.0, help="late window 使用最后多少比例的训练窗口。")
    parser.add_argument("--val-max-samples", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--persistent-workers", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--trainable", choices=["all", "head", "last-block-head", "custom"], default="last-block-head")
    parser.add_argument(
        "--trainable-module",
        action="append",
        default=[],
        help=(
            "可重复传入或用逗号分隔，覆盖 --trainable preset。"
            "允许模块：fc0, conv0, w0, conv1, w1, conv2, w2, conv3, w3, fc1, fc2。"
        ),
    )
    parser.add_argument("--selection-metric", default="competition_score_proxy")
    parser.add_argument("--selection-direction", choices=["max", "min"], default="max")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    result = train(build_parser().parse_args())
    best = result.get("best_metrics") or {}
    print(
        json.dumps(
            {
                "run_dir": result["run_dir"],
                "steps": result["steps"],
                "elapsed_seconds": result["elapsed_seconds"],
                "best_checkpoint": result["best_checkpoint"],
                "best_score": best.get("competition_score_proxy"),
                "best_forecast_mse": best.get("forecast_mse"),
                "scale_alignment": {
                    "temporal_stride": result["scale_alignment"]["temporal_stride"],
                    "spatial_downsample": result["scale_alignment"]["spatial_downsample"],
                    "first_supervised_target_raw_index": result["scale_alignment"]["first_supervised_target_raw_index"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
