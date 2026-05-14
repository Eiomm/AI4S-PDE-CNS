"""Train-from-scratch Task 2 MiniFNO and Temporal U-Net baselines.

This module is intentionally Task2-only. It validates all data paths against the
official Task2 file names and rejects Task1 checkpoint paths.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

try:  # Optional in the base project environment.
    import torch
    from torch import nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised in dependency-light CI.
    torch = None
    nn = None
    F = None


INPUT_STEPS = 10
OUTPUT_STEPS = 200
FORECAST_STEPS = OUTPUT_STEPS - INPUT_STEPS
SPATIAL_SIZE = 256
TASK2_ALLOWED_FILES = frozenset(
    {
        "task2_part0_train.h5",
        "task2_part1_train.h5",
        "task2_part2_train.h5",
        "task2_val.h5",
        "task2_test.h5",
    }
)
TASK2_TRAIN_FILES = (
    Path("data/Task2/task2_part0_train.h5"),
    Path("data/Task2/task2_part1_train.h5"),
    Path("data/Task2/task2_part2_train.h5"),
)


def require_torch():
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for Task2 model training/inference. "
            "Install a CPU or CUDA build of torch, then rerun this command."
        )
    return torch


def validate_task2_data_path(path: str | Path) -> Path:
    candidate = Path(path)
    parts = {part.lower() for part in candidate.parts}
    if candidate.name not in TASK2_ALLOWED_FILES or "task2" not in parts:
        raise ValueError(
            f"Task2 training/inference may only use data/Task2 files "
            f"{sorted(TASK2_ALLOWED_FILES)}, got {candidate}"
        )
    return candidate


def validate_task2_checkpoint_path(path: str | Path) -> Path:
    candidate = Path(path)
    lowered = candidate.as_posix().lower()
    if "task1" in lowered:
        raise ValueError(f"Task1 checkpoints are forbidden for Task2: {candidate}")
    if "task2" not in lowered:
        raise ValueError(f"Task2 checkpoint paths must be clearly Task2-specific: {candidate}")
    return candidate


def load_task2_tensor(path: str | Path, *, require_target: bool = True) -> np.ndarray:
    source = validate_task2_data_path(path)
    with h5py.File(source, "r") as h5:
        if "tensor" not in h5:
            raise KeyError(f"{source} must contain a 'tensor' dataset")
        tensor = h5["tensor"][:].astype(np.float32)
    if tensor.ndim != 3 or tensor.shape[2] != SPATIAL_SIZE:
        raise ValueError(f"{source} tensor must have shape (N, T, 256), got {tensor.shape}")
    min_steps = OUTPUT_STEPS if require_target else INPUT_STEPS
    if tensor.shape[1] < min_steps:
        raise ValueError(f"{source} needs at least {min_steps} time steps, got {tensor.shape[1]}")
    return tensor


class Task2TrajectoryDataset:
    """Small array-backed dataset exposing first-10-frame Task2 samples."""

    def __init__(self, paths: Iterable[str | Path], *, sample_limit: int | None = None):
        arrays = [load_task2_tensor(path, require_target=True)[:, :OUTPUT_STEPS, :] for path in paths]
        if not arrays:
            raise ValueError("At least one Task2 training file is required")
        tensor = np.concatenate(arrays, axis=0).astype(np.float32)
        if sample_limit is not None:
            tensor = tensor[: int(sample_limit)]
        self.tensor = tensor

    def __len__(self) -> int:
        return int(self.tensor.shape[0])

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        target = self.tensor[index]
        initial = target[:INPUT_STEPS]
        return initial.astype(np.float32), target.astype(np.float32)


if torch is not None:

    class SpectralConv1d(nn.Module):
        def __init__(self, channels: int, modes: int):
            super().__init__()
            self.channels = int(channels)
            self.modes = int(modes)
            scale = 1.0 / max(1, channels * channels)
            real = scale * torch.randn(channels, channels, self.modes)
            imag = scale * torch.randn(channels, channels, self.modes)
            self.weight = nn.Parameter(torch.complex(real, imag))

        def forward(self, x):
            batch_size, channels, size = x.shape
            x_ft = torch.fft.rfft(x, dim=-1)
            out_ft = torch.zeros(
                batch_size,
                channels,
                x_ft.shape[-1],
                dtype=torch.cfloat,
                device=x.device,
            )
            modes = min(self.modes, x_ft.shape[-1])
            out_ft[:, :, :modes] = torch.einsum("bim,iom->bom", x_ft[:, :, :modes], self.weight[:, :, :modes])
            return torch.fft.irfft(out_ft, n=size, dim=-1)


    class Task2MiniFNO(nn.Module):
        def __init__(self, hidden_channels: int = 48, modes: int = 32, layers: int = 4):
            super().__init__()
            self.hidden_channels = int(hidden_channels)
            self.modes = int(modes)
            self.layers = int(layers)
            self.lift = nn.Conv1d(INPUT_STEPS, self.hidden_channels, kernel_size=1)
            self.spectral = nn.ModuleList([SpectralConv1d(self.hidden_channels, self.modes) for _ in range(layers)])
            self.pointwise = nn.ModuleList(
                [nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=1) for _ in range(layers)]
            )
            self.head = nn.Sequential(
                nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(self.hidden_channels, FORECAST_STEPS, kernel_size=1),
            )

        def forward(self, initial):
            x = self.lift(initial)
            for spectral, pointwise in zip(self.spectral, self.pointwise):
                x = F.gelu(spectral(x) + pointwise(x))
            future = self.head(x) + initial[:, -1:, :]
            return _with_initial_frames(initial, future)


    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
                nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)


    class Task2TemporalUNet(nn.Module):
        def __init__(self, hidden_channels: int = 48, modes: int = 32):
            super().__init__()
            _ = modes
            h = int(hidden_channels)
            self.enc1 = ConvBlock(INPUT_STEPS, h)
            self.down1 = nn.Conv1d(h, h * 2, kernel_size=4, stride=2, padding=1)
            self.enc2 = ConvBlock(h * 2, h * 2)
            self.down2 = nn.Conv1d(h * 2, h * 4, kernel_size=4, stride=2, padding=1)
            self.mid = ConvBlock(h * 4, h * 4)
            self.up2 = nn.ConvTranspose1d(h * 4, h * 2, kernel_size=4, stride=2, padding=1)
            self.dec2 = ConvBlock(h * 4, h * 2)
            self.up1 = nn.ConvTranspose1d(h * 2, h, kernel_size=4, stride=2, padding=1)
            self.dec1 = ConvBlock(h * 2, h)
            self.head = nn.Conv1d(h, FORECAST_STEPS, kernel_size=1)

        def forward(self, initial):
            e1 = self.enc1(initial)
            e2 = self.enc2(self.down1(e1))
            mid = self.mid(self.down2(e2))
            d2 = self.up2(mid)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            future = self.head(d1) + initial[:, -1:, :]
            return _with_initial_frames(initial, future)


def _with_initial_frames(initial, future):
    return torch.cat([initial, future], dim=1)


def build_model(model_name: str, *, hidden_channels: int = 48, modes: int = 32):
    require_torch()
    normalized = model_name.lower()
    if normalized == "minifno":
        return Task2MiniFNO(hidden_channels=hidden_channels, modes=modes)
    if normalized == "unet":
        return Task2TemporalUNet(hidden_channels=hidden_channels, modes=modes)
    raise ValueError(f"Unknown Task2 model {model_name!r}; expected 'minifno' or 'unet'")


def _to_device(batch, device: str):
    initial, target = batch
    return initial.to(device=device, dtype=torch.float32), target.to(device=device, dtype=torch.float32)


def _mse_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    diff = prediction.astype(np.float64) - target.astype(np.float64)
    future = diff[:, INPUT_STEPS:, :]
    return {
        "mse": float(np.mean(diff * diff)),
        "forecast_mse": float(np.mean(future * future)),
        "initial_mse": float(np.mean(diff[:, :INPUT_STEPS, :] * diff[:, :INPUT_STEPS, :])),
    }


def persistence_prediction(initial: np.ndarray) -> np.ndarray:
    initial = np.asarray(initial, dtype=np.float32)
    prediction = np.empty((initial.shape[0], OUTPUT_STEPS, SPATIAL_SIZE), dtype=np.float32)
    prediction[:, :INPUT_STEPS, :] = initial
    prediction[:, INPUT_STEPS:, :] = initial[:, -1:, :]
    return prediction


def persistence_metrics(val_file: str | Path, *, sample_limit: int | None = None) -> dict[str, float]:
    target = load_task2_tensor(val_file, require_target=True)[:, :OUTPUT_STEPS, :]
    if sample_limit is not None:
        target = target[: int(sample_limit)]
    pred = persistence_prediction(target[:, :INPUT_STEPS, :])
    return _mse_metrics(pred, target)


def _selection_metric(metrics: dict[str, float]) -> float:
    return float(metrics.get("forecast_mse", metrics.get("mse", math.inf)))


def select_best_candidate(
    candidates: Iterable[dict[str, object]],
    persistence: dict[str, float],
) -> dict[str, object] | None:
    """Return the best trained Task2 candidate only if it beats persistence."""

    baseline = _selection_metric(persistence)
    best: dict[str, object] | None = None
    best_score = baseline
    for candidate in candidates:
        metrics = candidate.get("metrics")
        if not isinstance(metrics, dict):
            continue
        score = _selection_metric(metrics)
        if score < best_score:
            best = candidate
            best_score = score
    return best


def evaluate_model(model, val_file: str | Path, *, batch_size: int, device: str, sample_limit: int | None = None):
    require_torch()
    dataset = Task2TrajectoryDataset([val_file], sample_limit=sample_limit)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            initial, target = _to_device(batch, device)
            prediction = model(initial)
            predictions.append(prediction.cpu().numpy())
            targets.append(target.cpu().numpy())
    return _mse_metrics(np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0))


def train_one_model(
    *,
    model_name: str,
    train_files: Iterable[str | Path] = TASK2_TRAIN_FILES,
    val_file: str | Path = "data/Task2/task2_val.h5",
    checkpoint_path: str | Path = "runs/task2-models/task2_model.pt",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1.0e-3,
    hidden_channels: int = 48,
    modes: int = 32,
    sample_limit: int | None = None,
    val_sample_limit: int | None = None,
    device: str = "cpu",
    seed: int = 13,
) -> dict[str, object]:
    require_torch()
    torch.manual_seed(seed)
    train_paths = [validate_task2_data_path(path) for path in train_files]
    val_path = validate_task2_data_path(val_file)
    checkpoint = validate_task2_checkpoint_path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    dataset = Task2TrajectoryDataset(train_paths, sample_limit=sample_limit)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = build_model(model_name, hidden_channels=hidden_channels, modes=modes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(int(epochs)):
        model.train()
        losses: list[float] = []
        for batch in loader:
            initial, target = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(initial)
            loss = F.mse_loss(prediction[:, INPUT_STEPS:, :], target[:, INPUT_STEPS:, :])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics = evaluate_model(
            model,
            val_path,
            batch_size=batch_size,
            device=device,
            sample_limit=val_sample_limit,
        )
        history.append({"epoch": float(epoch + 1), "train_loss": float(np.mean(losses)), **val_metrics})

    train_time = time.perf_counter() - started
    final_metrics = history[-1] if history else evaluate_model(
        model,
        val_path,
        batch_size=batch_size,
        device=device,
        sample_limit=val_sample_limit,
    )
    torch.save(
        {
            "model_name": model_name,
            "hidden_channels": hidden_channels,
            "modes": modes,
            "state_dict": model.state_dict(),
            "input_steps": INPUT_STEPS,
            "output_steps": OUTPUT_STEPS,
            "spatial_size": SPATIAL_SIZE,
            "train_files": [str(path) for path in train_paths],
            "val_file": str(val_path),
            "history": history,
        },
        checkpoint,
    )
    return {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint),
        "train_time": train_time,
        "metrics": final_metrics,
        "history": history,
        "persistence": persistence_metrics(val_path, sample_limit=val_sample_limit),
    }


def select_best_candidate(
    candidates: Iterable[dict[str, object]],
    persistence: dict[str, float],
    *,
    primary_metric: str = "forecast_mse",
    min_relative_improvement: float = 0.0,
) -> dict[str, object] | None:
    """Return the best candidate only if it beats persistence on validation."""

    candidate_list = list(candidates)
    if not candidate_list:
        return None
    persistence_value = float(persistence[primary_metric])
    threshold = persistence_value * (1.0 - float(min_relative_improvement))

    def score(item: dict[str, object]) -> tuple[float, float]:
        metrics = item["metrics"]
        if not isinstance(metrics, dict):
            raise TypeError("candidate metrics must be a dict")
        return float(metrics[primary_metric]), float(metrics.get("mse", metrics[primary_metric]))

    best = min(candidate_list, key=score)
    best_value, _ = score(best)
    if best_value < threshold:
        return best
    return None


def train_candidates(
    *,
    model_names: Iterable[str],
    train_files: Iterable[str | Path] = TASK2_TRAIN_FILES,
    val_file: str | Path = "data/Task2/task2_val.h5",
    output_dir: str | Path = "runs/task2-models",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1.0e-3,
    hidden_channels: int = 48,
    modes: int = 32,
    sample_limit: int | None = None,
    val_sample_limit: int | None = None,
    device: str = "cpu",
    seed: int = 13,
    min_relative_improvement: float = 0.0,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    persistence = persistence_metrics(val_file, sample_limit=val_sample_limit)
    candidates = []
    for offset, model_name in enumerate(model_names):
        checkpoint = output / f"task2_{model_name}.pt"
        result = train_one_model(
            model_name=model_name,
            train_files=train_files,
            val_file=val_file,
            checkpoint_path=checkpoint,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            hidden_channels=hidden_channels,
            modes=modes,
            sample_limit=sample_limit,
            val_sample_limit=val_sample_limit,
            device=device,
            seed=seed + offset,
        )
        candidates.append(result)
    selected = select_best_candidate(
        candidates,
        persistence,
        min_relative_improvement=min_relative_improvement,
    )
    return {"persistence": persistence, "candidates": candidates, "selected": selected}


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task2 MiniFNO or Temporal U-Net from scratch.")
    parser.add_argument("--model", choices=["minifno", "unet", "all"], required=True)
    parser.add_argument("--train-files", nargs="+", default=[str(path) for path in TASK2_TRAIN_FILES])
    parser.add_argument("--val-file", default="data/Task2/task2_val.h5")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="runs/task2-models")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--val-sample-limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-relative-improvement", type=float, default=0.0)
    parser.add_argument("--promote-if-better", action="store_true")
    parser.add_argument("--test-file", default="data/Task2/task2_test.h5")
    parser.add_argument("--prediction-output", default="runs/task2-models/task2_pred.hdf5")
    args = parser.parse_args()

    if args.model == "all":
        result = train_candidates(
            model_names=["minifno", "unet"],
            train_files=args.train_files,
            val_file=args.val_file,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_channels=args.hidden_channels,
            modes=args.modes,
            sample_limit=args.sample_limit,
            val_sample_limit=args.val_sample_limit,
            device=args.device,
            seed=args.seed,
            min_relative_improvement=args.min_relative_improvement,
        )
    else:
        checkpoint = args.checkpoint
        if checkpoint is None:
            checkpoint = Path(args.output_dir) / f"task2_{args.model}.pt"
        candidate = train_one_model(
            model_name=args.model,
            train_files=args.train_files,
            val_file=args.val_file,
            checkpoint_path=checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_channels=args.hidden_channels,
            modes=args.modes,
            sample_limit=args.sample_limit,
            val_sample_limit=args.val_sample_limit,
            device=args.device,
            seed=args.seed,
        )
        persistence = candidate["persistence"]
        result = {
            "persistence": persistence,
            "candidates": [candidate],
            "selected": select_best_candidate(
                [candidate],
                persistence,
                min_relative_improvement=args.min_relative_improvement,
            ),
        }

    if args.promote_if_better and result["selected"] is not None:
        from infer_task2_model import run_inference

        promoted = run_inference(
            checkpoint_path=result["selected"]["checkpoint_path"],
            input_path=args.test_file,
            output_path=args.prediction_output,
            batch_size=args.batch_size,
            device=args.device,
        )
        result["promoted_prediction"] = promoted
    elif args.promote_if_better:
        result["promoted_prediction"] = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"task2_{args.model}_metrics.json"
    metrics_path.write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
