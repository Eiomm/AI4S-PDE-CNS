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

        with h5py.File(self.hdf5_path, "r") as h5:
            tensor = h5["tensor"]
            self.total_samples = int(tensor.shape[0])
            self.raw_time_steps = int(tensor.shape[1])
            self.raw_spatial_size = int(tensor.shape[2])
        if self.sample_start < 0 or self.sample_start >= self.total_samples:
            raise ValueError(f"invalid sample_start={self.sample_start}")
        available_samples = self.total_samples - self.sample_start
        self.num_samples = min(available_samples, int(max_samples)) if max_samples is not None else available_samples
        self.space_idx = spatial_indices(self.raw_spatial_size, self.spatial_size, self.spatial_downsample)
        self.reduced_time_indices = np.arange(0, self.raw_time_steps, self.temporal_stride, dtype=np.int64)
        self.windows_per_sample = len(self.reduced_time_indices) - self.initial_step - self.rollout_steps + 1
        if self.windows_per_sample <= 0:
            raise ValueError(
                "not enough reduced time steps: "
                f"raw_time={self.raw_time_steps}, stride={self.temporal_stride}, initial={self.initial_step}, rollout={self.rollout_steps}"
            )

    def __len__(self) -> int:
        return self.num_samples * self.windows_per_sample

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_offset = index // self.windows_per_sample
        window_start = index % self.windows_per_sample
        sample_index = self.sample_start + sample_offset
        reduced_slice = self.reduced_time_indices[window_start : window_start + self.initial_step + self.rollout_steps]
        with h5py.File(self.hdf5_path, "r") as h5:
            raw = h5["tensor"][sample_index, reduced_slice, :]
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
            "first_observed_raw_indices": self.reduced_time_indices[: self.initial_step].tolist(),
            "first_supervised_target_raw_index": int(self.reduced_time_indices[self.initial_step]),
        }


def grid_for_batch(features: torch.Tensor) -> torch.Tensor:
    grid = torch.linspace(0.0, 1.0, features.shape[1], device=features.device, dtype=features.dtype)
    return grid.view(1, features.shape[1], 1).expand(features.shape[0], -1, -1)


def predict_next(model: FNO1d, window: torch.Tensor) -> torch.Tensor:
    features = window.permute(0, 2, 1).contiguous()
    return model(features, grid_for_batch(features)).squeeze(-1).squeeze(-1)


def rollout_loss(model: FNO1d, inputs: torch.Tensor, targets: torch.Tensor, *, gamma: float) -> torch.Tensor:
    current = inputs
    losses = []
    for step in range(targets.shape[1]):
        prediction = predict_next(model, current)
        target = targets[:, step, :]
        losses.append((float(gamma) ** step) * torch.mean((prediction - target) ** 2))
        current = torch.cat([current[:, 1:, :], prediction.unsqueeze(1)], dim=1)
    return torch.stack(losses).mean()


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


def save_checkpoint(path: Path, model: FNO1d, optimizer: torch.optim.Optimizer, *, step: int, metrics: dict[str, float] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": int(step),
            "metrics": metrics or {},
            "scale_alignment": {
                "temporal_stride": 5,
                "spatial_downsample": 4,
                "model_step_meaning": "one model step equals 5 raw PDEBench time indices",
            },
        },
        path,
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    requested_run_dir = args.run_dir
    run_dir_arg = canonical_run_dir(args.run_dir)
    run_dir = resolve_path(run_dir_arg) if not Path(run_dir_arg).is_absolute() else Path(run_dir_arg)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "finetune_train.jsonl"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = ReducedBurgersWindowDataset(
        resolve_path(args.train_hdf5),
        initial_step=args.initial_step,
        rollout_steps=args.rollout_steps,
        temporal_stride=args.temporal_stride,
        spatial_downsample=args.spatial_downsample,
        spatial_size=args.spatial_size,
        max_samples=args.max_samples,
        sample_start=args.sample_start,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = load_fno_checkpoint(resolve_path(args.base_checkpoint), device)
    trainable_modules = parse_trainable_modules(args.trainable_module)
    trainable_names = set_trainable(model, args.trainable, trainable_modules)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_metrics: dict[str, float] | None = None
    base_metrics: dict[str, float] | None = None
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
        "args": vars(args),
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
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            loss = rollout_loss(model, inputs, targets, gamma=args.horizon_gamma)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if step == 1 or step % args.log_every == 0:
                record = {
                    "timestamp": utc_now(),
                    "event": "train_step",
                    "step": step,
                    "loss": float(loss.detach().cpu()),
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
                    "improved": improved,
                    "metrics": metrics,
                }
                history.append(record)
                append_jsonl(log_path, record)
                if improved:
                    best_metrics = metrics
                    save_checkpoint(run_dir / "best.pt", model, optimizer, step=step, metrics=best_metrics)
                model.train()

            if step >= args.steps:
                break

    elapsed = time.perf_counter() - start
    save_checkpoint(run_dir / "last.pt", model, optimizer, step=step, metrics=best_metrics)
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
        "history": history,
        "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
        "log_path": str(log_path),
        "time_path": str(time_path),
        "scale_alignment": dataset.metadata(),
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
        "base_metrics": base_metrics or {},
        "scale_alignment": dataset.metadata(),
        "trainable": args.trainable,
        "trainable_modules": trainable_modules or preset_trainable_modules(args.trainable),
        "trainable_parameter_names": trainable_names,
        "steps": int(step),
    }
    write_json(run_dir / "metadata.json", metadata)
    append_jsonl(log_path, {"timestamp": utc_now(), "event": "finetune_done", "elapsed_seconds": elapsed, "best_metrics": best_metrics})
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Task1 FNO fine-tune with official reduced-scale alignment.")
    parser.add_argument("--train-hdf5", default="data/pdebench_burgers/raw/1D_Burgers_Sols_Nu0.001.hdf5")
    parser.add_argument("--val-hdf5", default="data/task1_val.hdf5")
    parser.add_argument("--base-checkpoint", default="checkpoints/official/nu0.001_fno.pt")
    parser.add_argument("--run-dir", default=None, help="默认写入 runs/task1/<UTC timestamp>。")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--initial-step", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--horizon-gamma", type=float, default=0.9)
    parser.add_argument("--temporal-stride", type=int, default=5)
    parser.add_argument("--spatial-downsample", type=int, default=4)
    parser.add_argument("--spatial-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--val-max-samples", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
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
