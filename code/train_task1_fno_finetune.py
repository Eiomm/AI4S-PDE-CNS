"""Fine-tune a PDEBench Task 1 FNO checkpoint on Burgers HDF5 data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.pde_finetune_data import (  # noqa: E402
    HDF5WindowDatasetConfig,
    dataset_length,
    index_to_sample_and_target,
    index_to_sample_and_rollout_start,
    read_rollout_window,
    read_training_window,
    rollout_dataset_length,
)
from agent.pde_finetune import is_better_metric  # noqa: E402
from evaluate_task1 import compute_task1_metrics  # noqa: E402
from fno_inference import ResidualCorrectedFNO1d, load_fno_checkpoint, run_autoregressive_inference  # noqa: E402


class HDF5OneStepDataset(Dataset):
    def __init__(self, config: HDF5WindowDatasetConfig, *, rollout_steps: int = 1):
        self.config = config
        self.rollout_steps = int(rollout_steps)
        if self.rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        self.length = dataset_length(config) if self.rollout_steps == 1 else rollout_dataset_length(config, self.rollout_steps)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        if self.rollout_steps > 1:
            sample_index, start_time_index = index_to_sample_and_rollout_start(
                self.config,
                index,
                rollout_steps=self.rollout_steps,
            )
            window = read_rollout_window(
                self.config,
                sample_index=sample_index,
                start_time_index=start_time_index,
                rollout_steps=self.rollout_steps,
            )
            return (
                torch.from_numpy(window.input_frames),
                torch.from_numpy(window.target_frames),
            )
        sample_index, target_time_index = index_to_sample_and_target(self.config, index)
        window = read_training_window(
            self.config,
            sample_index=sample_index,
            target_time_index=target_time_index,
        )
        return (
            torch.from_numpy(window.input_frames),
            torch.from_numpy(window.target_frame),
        )


def _collate(batch):
    inputs, targets = zip(*batch)
    return torch.stack(list(inputs), dim=0), torch.stack(list(targets), dim=0)


def _load_val(path: Path, spatial_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from agent.pde_finetune_data import spatial_indices

    with h5py.File(path, "r") as h5:
        tensor = h5["tensor"][:]
        source_size = tensor.shape[-1]
        indices = spatial_indices(source_size=source_size, target_size=spatial_size)
        tensor = tensor[:, :, indices]
        x_coords = h5["x-coordinate"][indices] if "x-coordinate" in h5 else np.linspace(0.0, 1.0, spatial_size, endpoint=False)
        t_coords = h5["t-coordinate"][:]
    return tensor.astype(np.float32), x_coords.astype(np.float32), t_coords.astype(np.float32)


def _evaluate_rollout(model, val_path: Path, device: torch.device, spatial_size: int, batch_size: int, max_samples: int | None):
    target, x_coords, t_coords = _load_val(val_path, spatial_size)
    if max_samples is not None:
        target = target[:max_samples]
    initial = target[:, :10, :]
    prediction = run_autoregressive_inference(
        model,
        initial,
        x_coords,
        t_coords,
        device,
        batch_size=batch_size,
    )
    return compute_task1_metrics(prediction, target)


def _grid_for_batch(features: torch.Tensor) -> torch.Tensor:
    grid = torch.linspace(0.0, 1.0, features.shape[1], device=features.device, dtype=torch.float32)
    return grid.view(1, features.shape[1], 1).expand(features.shape[0], -1, -1)


def _predict_next(model, inputs: torch.Tensor) -> torch.Tensor:
    features = inputs.permute(0, 2, 1).contiguous()
    return model(features, _grid_for_batch(features)).squeeze(-1).squeeze(-1)


def _gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((torch.diff(prediction, dim=-1) - torch.diff(target, dim=-1)) ** 2)


def _spectral_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_fft = torch.fft.rfft(prediction, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    return torch.mean((torch.abs(pred_fft) - torch.abs(target_fft)) ** 2)


def _burgers_residual_loss(
    previous_frames: torch.Tensor,
    prediction: torch.Tensor,
    *,
    nu: float,
    dt: float,
    dx: float,
) -> torch.Tensor:
    if previous_frames.shape[1] < 2:
        return prediction.new_tensor(0.0)
    previous = previous_frames[:, -2, :]
    current = previous_frames[:, -1, :]
    next_frame = prediction
    u_t = (next_frame - previous) / (2.0 * float(dt))
    u_x = (torch.roll(current, shifts=-1, dims=-1) - torch.roll(current, shifts=1, dims=-1)) / (2.0 * float(dx))
    u_xx = (
        torch.roll(current, shifts=-1, dims=-1)
        - 2.0 * current
        + torch.roll(current, shifts=1, dims=-1)
    ) / (float(dx) ** 2)
    residual = u_t + current * u_x - float(nu) * u_xx
    return torch.mean(residual**2)


def _frame_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    gradient_loss_weight: float,
    spectral_loss_weight: float,
) -> torch.Tensor:
    loss = torch.mean((prediction - target) ** 2)
    if gradient_loss_weight:
        loss = loss + float(gradient_loss_weight) * _gradient_loss(prediction, target)
    if spectral_loss_weight:
        loss = loss + float(spectral_loss_weight) * _spectral_loss(prediction, target)
    return loss


def _training_loss(
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    rollout_steps: int,
    gradient_loss_weight: float = 0.0,
    spectral_loss_weight: float = 0.0,
    physics_loss_weight: float = 0.0,
    physics_nu: float = 0.001,
    physics_dt: float = 0.025,
    physics_dx: float = 1.0 / 256.0,
    horizon_loss_gamma: float = 1.0,
) -> torch.Tensor:
    if rollout_steps == 1:
        prediction = _predict_next(model, inputs)
        loss = _frame_loss(
            prediction,
            targets,
            gradient_loss_weight=gradient_loss_weight,
            spectral_loss_weight=spectral_loss_weight,
        )
        if physics_loss_weight:
            loss = loss + float(physics_loss_weight) * _burgers_residual_loss(
                inputs,
                prediction,
                nu=physics_nu,
                dt=physics_dt,
                dx=physics_dx,
            )
        return loss
    current = inputs
    losses = []
    for horizon in range(rollout_steps):
        prediction = _predict_next(model, current)
        frame_loss = _frame_loss(
            prediction,
            targets[:, horizon, :],
            gradient_loss_weight=gradient_loss_weight,
            spectral_loss_weight=spectral_loss_weight,
        )
        if physics_loss_weight:
            frame_loss = frame_loss + float(physics_loss_weight) * _burgers_residual_loss(
                current,
                prediction,
                nu=physics_nu,
                dt=physics_dt,
                dx=physics_dx,
            )
        losses.append((float(horizon_loss_gamma) ** horizon) * frame_loss)
        current = torch.cat([current[:, 1:, :], prediction.unsqueeze(1)], dim=1)
    return torch.stack(losses).mean()


def _save_checkpoint(path: Path, model, optimizer, *, step: int, metrics: dict[str, float] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, ResidualCorrectedFNO1d):
        payload = {
            "model_kind": "residual_corrected_fno",
            "base_model_state_dict": model.base_model.state_dict(),
            "residual_head_state_dict": model.residual_head.state_dict(),
            "residual_head_hidden": model.residual_head_hidden,
            "residual_head_scale": model.residual_head_scale,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "metrics": metrics or {},
        }
    else:
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "metrics": metrics or {},
        }
    torch.save(payload, path)


def _configure_trainable_parameters(model: torch.nn.Module, trainable: str) -> list[str]:
    base_model = model.base_model if isinstance(model, ResidualCorrectedFNO1d) else model
    if trainable == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif trainable == "head":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for module_name in ("fc1", "fc2"):
            module = getattr(base_model, module_name)
            for parameter in module.parameters():
                parameter.requires_grad = True
    elif trainable == "last-block-head":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for module_name in ("conv3", "w3", "fc1", "fc2"):
            module = getattr(base_model, module_name)
            for parameter in module.parameters():
                parameter.requires_grad = True
        if isinstance(model, ResidualCorrectedFNO1d):
            for parameter in model.residual_head.parameters():
                parameter.requires_grad = True
    elif trainable == "residual-head":
        if not isinstance(model, ResidualCorrectedFNO1d):
            raise ValueError("trainable=residual-head requires --architecture residual-corrected-fno")
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.residual_head.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"unsupported trainable mode: {trainable}")
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def _apply_architecture(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    if args.architecture == "fno":
        return model
    if args.architecture == "residual-corrected-fno":
        if isinstance(model, ResidualCorrectedFNO1d):
            return model
        return ResidualCorrectedFNO1d(
            model,
            hidden_channels=args.residual_head_hidden,
            scale=args.residual_head_scale,
        )
    raise ValueError(f"unsupported architecture: {args.architecture}")


def train(args: argparse.Namespace) -> dict[str, object]:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_fno_checkpoint(str(args.base_checkpoint), device)
    model = _apply_architecture(model, args).to(device)
    trainable_parameter_names = _configure_trainable_parameters(model, args.trainable)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("no trainable parameters were selected")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)

    config = HDF5WindowDatasetConfig(
        hdf5_path=Path(args.train_hdf5),
        initial_step=args.initial_step,
        spatial_size=args.spatial_size,
        max_samples=args.max_samples,
        sample_start=args.sample_start,
        max_time_steps=args.max_time_steps,
        temporal_stride=args.temporal_stride,
    )
    dataset = HDF5OneStepDataset(config, rollout_steps=args.rollout_steps)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
        generator=generator,
    )

    best_metrics: dict[str, float] | None = None
    base_metrics: dict[str, float] | None = None
    history: list[dict[str, object]] = []
    step = 0
    start = time.perf_counter()

    if args.val_hdf5 and not args.skip_base_eval:
        model.eval()
        with torch.no_grad():
            metrics = _evaluate_rollout(
                model,
                Path(args.val_hdf5),
                device,
                args.spatial_size,
                args.eval_batch_size,
                args.val_max_samples,
            )
        base_metrics = {key: float(value) for key, value in metrics.items()}
        best_metrics = dict(base_metrics)
        history.append({"step": 0, "phase": "base", **base_metrics})
        _save_checkpoint(run_dir / "best.pt", model, optimizer, step=0, metrics=best_metrics)
        print(json.dumps({"step": 0, "phase": "base", **base_metrics}, ensure_ascii=False))

    model.train()
    while step < args.steps:
        for inputs, targets in loader:
            step += 1
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            loss = _training_loss(
                model,
                inputs,
                targets,
                rollout_steps=args.rollout_steps,
                gradient_loss_weight=args.gradient_loss_weight,
                spectral_loss_weight=args.spectral_loss_weight,
                physics_loss_weight=args.physics_loss_weight,
                physics_nu=args.physics_nu,
                physics_dt=args.physics_dt,
                physics_dx=args.physics_dx,
                horizon_loss_gamma=args.horizon_loss_gamma,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if step == 1 or step % args.log_every == 0:
                record = {"step": step, "loss": float(loss.detach().cpu())}
                history.append(record)
                print(json.dumps(record, ensure_ascii=False))

            if args.val_hdf5 and (step == args.steps or step % args.val_every == 0):
                model.eval()
                with torch.no_grad():
                    metrics = _evaluate_rollout(
                        model,
                        Path(args.val_hdf5),
                        device,
                        args.spatial_size,
                        args.eval_batch_size,
                        args.val_max_samples,
                )
                history.append({"step": step, **{key: float(value) for key, value in metrics.items()}})
                candidate_metrics = {key: float(value) for key, value in metrics.items()}
                if is_better_metric(
                    candidate_metrics,
                    best_metrics,
                    metric=args.selection_metric,
                    min_improvement=args.min_improvement,
                    maximize=args.selection_direction == "max",
                ):
                    best_metrics = candidate_metrics
                    _save_checkpoint(run_dir / "best.pt", model, optimizer, step=step, metrics=best_metrics)
                model.train()

            if step >= args.steps:
                break

    elapsed = time.perf_counter() - start
    _save_checkpoint(run_dir / "last.pt", model, optimizer, step=step, metrics=best_metrics)
    payload = {
        "success": True,
        "run_dir": str(run_dir),
        "train_hdf5": str(args.train_hdf5),
        "base_checkpoint": str(args.base_checkpoint),
        "steps": step,
        "sample_start": args.sample_start,
        "max_samples": args.max_samples,
        "max_time_steps": args.max_time_steps,
        "temporal_stride": args.temporal_stride,
        "rollout_steps": args.rollout_steps,
        "trainable": args.trainable,
        "architecture": args.architecture,
        "residual_head_hidden": args.residual_head_hidden,
        "residual_head_scale": args.residual_head_scale,
        "trainable_parameter_names": trainable_parameter_names,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "gradient_loss_weight": args.gradient_loss_weight,
        "spectral_loss_weight": args.spectral_loss_weight,
        "physics_loss_weight": args.physics_loss_weight,
        "physics_nu": args.physics_nu,
        "physics_dt": args.physics_dt,
        "physics_dx": args.physics_dx,
        "horizon_loss_gamma": args.horizon_loss_gamma,
        "elapsed_seconds": elapsed,
        "min_improvement": args.min_improvement,
        "selection_metric": args.selection_metric,
        "selection_direction": args.selection_direction,
        "base_metrics": base_metrics,
        "best_metrics": best_metrics,
        "history": history,
    }
    (run_dir / "finetune_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a Task 1 FNO checkpoint on PDEBench Burgers data.")
    parser.add_argument("--train-hdf5", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--val-hdf5", default="data/Task1/task1_val.hdf5")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gradient-loss-weight", type=float, default=0.0)
    parser.add_argument("--spectral-loss-weight", type=float, default=0.0)
    parser.add_argument("--physics-loss-weight", type=float, default=0.0)
    parser.add_argument("--physics-nu", type=float, default=0.001)
    parser.add_argument("--physics-dt", type=float, default=0.025)
    parser.add_argument("--physics-dx", type=float, default=1.0 / 256.0)
    parser.add_argument("--horizon-loss-gamma", type=float, default=1.0)
    parser.add_argument("--initial-step", type=int, default=10)
    parser.add_argument("--spatial-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--max-time-steps", type=int, default=200)
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument("--trainable", choices=["all", "head", "last-block-head", "residual-head"], default="all")
    parser.add_argument("--architecture", choices=["fno", "residual-corrected-fno"], default="fno")
    parser.add_argument("--residual-head-hidden", type=int, default=32)
    parser.add_argument("--residual-head-scale", type=float, default=1.0)
    parser.add_argument("--val-max-samples", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--selection-metric", default="competition_score_proxy")
    parser.add_argument("--selection-direction", choices=["min", "max"], default="max")
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    result = train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
