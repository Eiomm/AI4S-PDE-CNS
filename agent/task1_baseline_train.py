from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .pde_baseline_losses import initial_consistency_mse, spectral_mse
from .pde_baselines import BaselineSpec, write_baseline_artifacts
from .pde_finetune_data import spatial_indices
from .pde_metrics import compute_task1_metrics
from .pde_results import RunResult
from .task1_trajectory_data import Task1TrajectoryConfig, read_task1_trajectory_sample, task1_trajectory_length


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("torch is required for train_task1_baseline; run this in the Hwpytorch/GPU environment") from exc
    return torch, nn, DataLoader, Dataset


def _write_prediction(path: Path, prediction: np.ndarray, *, dataset_key: str = "prediction") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with h5py.File(path, "w") as h5:
        h5.create_dataset(dataset_key, data=np.asarray(prediction, dtype=np.float32))
    return path


def _read_target(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if "tensor" in h5:
            return h5["tensor"][:]
        if "prediction" in h5:
            return h5["prediction"][:]
        return h5[next(iter(h5.keys()))][:]


def _default_train_paths(project_root: Path) -> list[Path]:
    raw = project_root / "data" / "pdebench_burgers" / "raw"
    return [
        raw / "1D_Burgers_Sols_Nu0.001.hdf5",
    ]


def normalize_loss_window(output_steps: int, loss_start_step: int = 10, loss_end_step: int | None = None) -> tuple[int, int]:
    end = output_steps if loss_end_step is None else int(loss_end_step)
    start = int(loss_start_step)
    if start < 10 or start >= output_steps:
        raise ValueError(f"loss_start_step must be inside [10, {output_steps - 1}], got {start}")
    if end <= start or end > output_steps:
        raise ValueError(f"loss_end_step must be inside ({start}, {output_steps}], got {end}")
    return start, end


def torch_spectral_mse(
    prediction,
    target,
    *,
    low_weight: float = 1.0,
    mid_weight: float = 1.0,
    high_weight: float = 1.0,
):
    torch, _, _, _ = _require_torch()
    pred_fft = torch.fft.rfft(prediction, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    error = (pred_fft - target_fft).abs().pow(2)
    weights = torch.ones(error.shape[-1], dtype=prediction.dtype, device=prediction.device)
    bands = torch.tensor_split(torch.arange(error.shape[-1], device=prediction.device), 3)
    for band, weight in zip(bands, (low_weight, mid_weight, high_weight)):
        weights[band] = float(weight)
    return (error * weights.reshape((1,) * (error.ndim - 1) + (-1,))).mean()


def torch_burgers_residual_mse(
    trajectory,
    *,
    nu: float = 0.001,
    dx: float = 1.0 / 256.0,
    dt: float = 0.05,
    start_step: int = 10,
    end_step: int | None = None,
):
    torch, _, _, _ = _require_torch()
    end = trajectory.shape[1] if end_step is None else int(end_step)
    start = int(start_step)
    if end - start < 2:
        return trajectory.new_tensor(0.0)
    u = trajectory[:, start:end, :]
    current = u[:, :-1, :]
    next_u = u[:, 1:, :]
    u_t = (next_u - current) / float(dt)
    u_x = (torch.roll(current, shifts=-1, dims=-1) - torch.roll(current, shifts=1, dims=-1)) / (2.0 * float(dx))
    u_xx = (
        torch.roll(current, shifts=-1, dims=-1)
        - 2.0 * current
        + torch.roll(current, shifts=1, dims=-1)
    ) / (float(dx) ** 2)
    residual = u_t + current * u_x - float(nu) * u_xx
    return residual.pow(2).mean()


def build_model(model_name: str, *, spatial_size: int = 256, output_steps: int = 200, hidden: int = 64, rank: int = 64):
    torch, nn, _, _ = _require_torch()

    class UNet1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Conv1d(10, hidden, 5, padding=2), nn.GELU(), nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU())
            self.enc2 = nn.Sequential(nn.Conv1d(hidden, hidden * 2, 5, padding=2), nn.GELU(), nn.Conv1d(hidden * 2, hidden * 2, 3, padding=1), nn.GELU())
            self.out = nn.Conv1d(hidden * 2, output_steps, 1)

        def forward(self, initial, base=None):
            pred = self.out(self.enc2(self.enc1(initial)))
            pred[:, :10, :] = initial
            return pred

    class DeepONetLite(nn.Module):
        def __init__(self):
            super().__init__()
            self.branch = nn.Sequential(
                nn.Conv1d(10, hidden, 5, padding=2),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(hidden, rank),
            )
            self.trunk = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, rank))
            t = torch.linspace(0.0, 1.0, output_steps)
            x = torch.linspace(0.0, 1.0, spatial_size)
            tt, xx = torch.meshgrid(t, x, indexing="ij")
            self.register_buffer("coords", torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=1))

        def forward(self, initial, base=None):
            branch = self.branch(initial)
            trunk = self.trunk(self.coords)
            pred = torch.einsum("br,pr->bp", branch, trunk).reshape(initial.shape[0], output_steps, spatial_size)
            pred[:, :10, :] = initial
            return pred

    class ResidualRefiner(nn.Module):
        def __init__(self):
            super().__init__()
            final = nn.Conv1d(hidden, output_steps, 1)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            self.net = nn.Sequential(
                nn.Conv1d(10 + output_steps, hidden, 5, padding=2),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, 5, padding=2),
                nn.GELU(),
                final,
            )

        def forward(self, initial, base=None):
            if base is None:
                raise ValueError("residual_refiner requires a base prediction")
            correction = self.net(torch.cat([initial, base], dim=1))
            pred = base + correction
            pred[:, :10, :] = initial
            return pred

    if model_name in {"unet1d", "pino_fno"}:
        return UNet1D()
    if model_name == "deeponet_lite":
        return DeepONetLite()
    if model_name == "residual_refiner":
        return ResidualRefiner()
    if model_name == "tfno":
        raise RuntimeError("tfno training requires neuralop integration; use fno/pino_fno prototypes first")
    raise ValueError(f"unknown baseline model: {model_name}")


def _make_dataset_class():
    _, _, _, Dataset = _require_torch()

    class TrajectoryDataset(Dataset):
        def __init__(self, config: Task1TrajectoryConfig):
            self.config = config
            self.length = task1_trajectory_length(config)

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            sample = read_task1_trajectory_sample(self.config, int(index))
            return sample.initial, sample.target

    return TrajectoryDataset


def _make_residual_dataset_class(base_hdf5_paths: list[Path]):
    _, _, _, Dataset = _require_torch()

    class ResidualTrajectoryDataset(Dataset):
        def __init__(self, config: Task1TrajectoryConfig):
            if len(base_hdf5_paths) != len(config.hdf5_paths):
                raise ValueError("base_train_hdf5 must have the same number of files as train_hdf5")
            self.config = config
            self.base_paths = [Path(path) for path in base_hdf5_paths]
            initial_blocks = []
            target_blocks = []
            base_blocks = []
            for train_path, base_path in zip(config.hdf5_paths, self.base_paths):
                start = int(config.sample_start)
                count = _sample_count_for_path(Path(train_path), config)
                stop = start + count
                target = _read_target_block(Path(train_path), start=start, stop=stop, config=config)
                base = _read_base_block(
                    base_path,
                    start=start,
                    stop=stop,
                    spatial_size=config.spatial_size,
                    output_steps=config.output_steps,
                )
                initial_blocks.append(target[:, : config.initial_step, :].copy())
                target_blocks.append(target)
                base_blocks.append(base)
            self.initial = np.concatenate(initial_blocks, axis=0).astype(np.float32, copy=False)
            self.target = np.concatenate(target_blocks, axis=0).astype(np.float32, copy=False)
            self.base = np.concatenate(base_blocks, axis=0).astype(np.float32, copy=False)
            self.length = int(self.target.shape[0])

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            idx = int(index)
            return self.initial[idx], self.target[idx], self.base[idx]

    return ResidualTrajectoryDataset


def _predict_validation(
    model,
    target: np.ndarray,
    *,
    device: str,
    batch_size: int,
    base_prediction: np.ndarray | None = None,
) -> np.ndarray:
    torch, _, _, _ = _require_torch()
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, target.shape[0], batch_size):
            batch = torch.from_numpy(target[start : start + batch_size, :10, :].astype(np.float32)).to(device)
            if base_prediction is not None:
                base = torch.from_numpy(base_prediction[start : start + batch_size].astype(np.float32)).to(device)
                pred = model(batch, base=base).detach().cpu().numpy()
            else:
                pred = model(batch).detach().cpu().numpy()
            pred[:, :10, :] = target[start : start + batch_size, :10, :]
            predictions.append(pred.astype(np.float32))
    return np.concatenate(predictions, axis=0)


def _read_base_sample(path: Path, sample_index: int, *, spatial_size: int, output_steps: int) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        key = "prediction" if "prediction" in h5 else "tensor"
        if key not in h5:
            key = next(iter(h5.keys()))
        dataset = h5[key]
        if sample_index >= dataset.shape[0]:
            raise IndexError(f"base sample index {sample_index} is outside {path}")
        source_size = int(dataset.shape[2])
        indices = spatial_indices(source_size=source_size, target_size=spatial_size)
        base = dataset[sample_index, :output_steps, indices]
    return np.asarray(base, dtype=np.float32)


def _sample_count_for_path(path: Path, config: Task1TrajectoryConfig) -> int:
    with h5py.File(path, "r") as h5:
        samples = int(h5["tensor"].shape[0])
    if config.sample_start < 0 or config.sample_start >= samples:
        raise ValueError(f"sample_start {config.sample_start} is outside sample count {samples}")
    available = samples - config.sample_start
    return min(available, config.max_samples_per_file) if config.max_samples_per_file is not None else available


def _read_target_block(path: Path, *, start: int, stop: int, config: Task1TrajectoryConfig) -> np.ndarray:
    chunks = []
    with h5py.File(path, "r") as h5:
        tensor = h5["tensor"]
        if int(tensor.shape[1]) < config.output_steps:
            raise ValueError(f"{path} has only {tensor.shape[1]} time steps, need {config.output_steps}")
        indices = spatial_indices(source_size=int(tensor.shape[2]), target_size=config.spatial_size)
        for cursor in range(start, stop, 256):
            end = min(cursor + 256, stop)
            if int(tensor.shape[2]) == config.spatial_size:
                chunk = tensor[cursor:end, : config.output_steps, :]
            else:
                chunk = tensor[cursor:end, : config.output_steps, :][:, :, indices]
            chunks.append(np.asarray(chunk, dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def _read_base_block(path: Path, *, start: int, stop: int, spatial_size: int, output_steps: int) -> np.ndarray:
    chunks = []
    with h5py.File(path, "r") as h5:
        key = "prediction" if "prediction" in h5 else "tensor"
        if key not in h5:
            key = next(iter(h5.keys()))
        dataset = h5[key]
        if stop > int(dataset.shape[0]):
            raise IndexError(f"base sample stop {stop} is outside {path}")
        indices = spatial_indices(source_size=int(dataset.shape[2]), target_size=spatial_size)
        for cursor in range(start, stop, 256):
            end = min(cursor + 256, stop)
            if int(dataset.shape[2]) == spatial_size:
                chunk = dataset[cursor:end, :output_steps, :]
            else:
                chunk = dataset[cursor:end, :output_steps, :][:, :, indices]
            chunks.append(np.asarray(chunk, dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def train_task1_baseline(
    *,
    model_name: str,
    run_dir: str | Path,
    project_root: str | Path = ".",
    train_hdf5: list[str | Path] | None = None,
    validation_hdf5: str | Path | None = None,
    max_samples: int = 1024,
    steps: int = 200,
    batch_size: int = 4,
    lr: float = 1.0e-3,
    hidden: int = 64,
    device: str | None = None,
    loss_start_step: int = 10,
    loss_end_step: int | None = None,
    base_train_hdf5: list[str | Path] | None = None,
    base_validation_prediction_path: str | Path | None = None,
    initial_loss_weight: float = 0.05,
    spectral_loss_weight: float = 0.0,
    spectral_high_weight: float = 2.0,
    physics_loss_weight: float = 0.0,
    physics_nu: float = 0.001,
    physics_dt: float = 0.05,
    physics_dx: float = 1.0 / 256.0,
) -> RunResult:
    torch, nn, DataLoader, _ = _require_torch()
    root = Path(project_root)
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_paths = [Path(path) for path in (train_hdf5 or _default_train_paths(root))]
    train_paths = [path for path in train_paths if path.exists()]
    if not train_paths:
        raise FileNotFoundError("no training HDF5 files found")
    validation_path = Path(validation_hdf5) if validation_hdf5 is not None else root / "data" / "Task1" / "task1_val.hdf5"
    dataset_config = Task1TrajectoryConfig(
        hdf5_paths=train_paths,
        max_samples_per_file=max(1, max_samples // len(train_paths)),
    )
    base_train_paths = [Path(path) for path in (base_train_hdf5 or [])]
    if model_name == "residual_refiner" and not base_train_paths:
        raise ValueError("residual_refiner requires base_train_hdf5 with FNO base predictions")
    if model_name == "residual_refiner" and base_validation_prediction_path is None:
        raise ValueError("residual_refiner requires base_validation_prediction_path")
    DatasetClass = _make_residual_dataset_class(base_train_paths) if model_name == "residual_refiner" else _make_dataset_class()
    loader = DataLoader(DatasetClass(dataset_config), batch_size=batch_size, shuffle=True, num_workers=0)
    loss_start, loss_end = normalize_loss_window(200, loss_start_step, loss_end_step)
    model = build_model(model_name, hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    start_time = time.perf_counter()
    iterator = iter(loader)
    model.train()
    last_loss = 0.0
    for _ in range(int(steps)):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        if model_name == "residual_refiner":
            initial, target, base = batch
            base = base.to(device)
        else:
            initial, target = batch
            base = None
        initial = initial.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(initial, base=base)
        loss = mse_loss(prediction[:, loss_start:loss_end, :], target[:, loss_start:loss_end, :])
        if initial_loss_weight > 0.0 or model_name == "pino_fno":
            loss = loss + float(initial_loss_weight) * mse_loss(prediction[:, :10, :], initial)
        if spectral_loss_weight > 0.0:
            loss = loss + float(spectral_loss_weight) * torch_spectral_mse(
                prediction[:, loss_start:loss_end, :],
                target[:, loss_start:loss_end, :],
                high_weight=spectral_high_weight,
            )
        if physics_loss_weight > 0.0:
            loss = loss + float(physics_loss_weight) * torch_burgers_residual_mse(
                prediction,
                nu=physics_nu,
                dx=physics_dx,
                dt=physics_dt,
                start_step=loss_start,
                end_step=loss_end,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    train_time = time.perf_counter() - start_time
    checkpoint_path = run_path / "best.pt"
    torch.save({"model_name": model_name, "state_dict": model.state_dict(), "hidden": hidden}, checkpoint_path)

    target = _read_target(validation_path).astype(np.float32)
    base_validation = None
    if base_validation_prediction_path is not None:
        base_validation = _read_target(Path(base_validation_prediction_path)).astype(np.float32)
        if base_validation.shape != target.shape:
            raise ValueError(f"base validation prediction shape {base_validation.shape} does not match target shape {target.shape}")
    inference_start = time.perf_counter()
    prediction = _predict_validation(model, target, device=device, batch_size=batch_size, base_prediction=base_validation)
    inference_time = time.perf_counter() - inference_start
    prediction_path = _write_prediction(run_path / "task1_val_pred.hdf5", prediction)
    metrics = compute_task1_metrics(prediction, target)
    metrics["train_loss"] = last_loss
    metrics["spectral_mse"] = spectral_mse(prediction, target)
    metrics["initial_consistency_mse"] = initial_consistency_mse(prediction, target[:, :10, :])
    metrics["physics_loss_weight"] = float(physics_loss_weight)
    metrics["spectral_loss_weight"] = float(spectral_loss_weight)
    (run_path / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = RunResult(
        task_id="task1",
        run_dir=run_path,
        metrics=metrics,
        prediction_path=prediction_path,
        zip_path=None,
        train_time=train_time,
        inference_time=inference_time,
        success=True,
        weights={},
        command=[sys.executable, "code/train_task1_baseline.py", "--model", model_name],
    )
    write_baseline_artifacts(
        run_path,
        BaselineSpec(name=model_name, family=model_name, trainable=True),
        {
            "model": model_name,
            "train_hdf5": [str(path) for path in train_paths],
            "validation_hdf5": str(validation_path),
            "max_samples": max_samples,
            "steps": steps,
            "batch_size": batch_size,
            "lr": lr,
            "hidden": hidden,
            "device": device,
            "loss_start_step": loss_start,
            "loss_end_step": loss_end,
            "checkpoint_path": str(checkpoint_path),
            "base_train_hdf5": [str(path) for path in base_train_paths],
            "base_validation_prediction_path": str(base_validation_prediction_path) if base_validation_prediction_path else None,
            "initial_loss_weight": initial_loss_weight,
            "spectral_loss_weight": spectral_loss_weight,
            "spectral_high_weight": spectral_high_weight,
            "physics_loss_weight": physics_loss_weight,
            "physics_nu": physics_nu,
            "physics_dt": physics_dt,
            "physics_dx": physics_dx,
        },
        result,
        conclusion=f"prototype validation competition_score_proxy={metrics['competition_score_proxy']:.6g}",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight Task 1 Baseline Zoo model.")
    parser.add_argument("--model", required=True, choices=["unet1d", "deeponet_lite", "residual_refiner", "pino_fno", "tfno"])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--train-hdf5", action="append", default=None)
    parser.add_argument("--validation-hdf5", default=None)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--loss-start-step", type=int, default=10)
    parser.add_argument("--loss-end-step", type=int, default=None)
    parser.add_argument("--base-train-hdf5", action="append", default=None)
    parser.add_argument("--base-validation-prediction-path", default=None)
    parser.add_argument("--initial-loss-weight", type=float, default=0.05)
    parser.add_argument("--spectral-loss-weight", type=float, default=0.0)
    parser.add_argument("--spectral-high-weight", type=float, default=2.0)
    parser.add_argument("--physics-loss-weight", type=float, default=0.0)
    parser.add_argument("--physics-nu", type=float, default=0.001)
    parser.add_argument("--physics-dt", type=float, default=0.05)
    parser.add_argument("--physics-dx", type=float, default=1.0 / 256.0)
    args = parser.parse_args()
    result = train_task1_baseline(
        model_name=args.model,
        run_dir=args.run_dir,
        project_root=args.project_root,
        train_hdf5=args.train_hdf5,
        validation_hdf5=args.validation_hdf5,
        max_samples=args.max_samples,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        device=args.device,
        loss_start_step=args.loss_start_step,
        loss_end_step=args.loss_end_step,
        base_train_hdf5=args.base_train_hdf5,
        base_validation_prediction_path=args.base_validation_prediction_path,
        initial_loss_weight=args.initial_loss_weight,
        spectral_loss_weight=args.spectral_loss_weight,
        spectral_high_weight=args.spectral_high_weight,
        physics_loss_weight=args.physics_loss_weight,
        physics_nu=args.physics_nu,
        physics_dt=args.physics_dt,
        physics_dx=args.physics_dx,
    )
    print(json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
