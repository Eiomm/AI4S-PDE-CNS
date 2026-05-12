"""PDEBench FNO inference for AI4S CNS Task 1 (1D Burgers equation).

Loads a PDEBench FNO checkpoint and runs autoregressive prediction on test data.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D Fourier layer from PDEBench."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1) // 2 + 1,
            device=x.device, dtype=torch.cfloat,
        )
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNO1d(nn.Module):
    """PDEBench 1D Fourier Neural Operator."""

    def __init__(self, num_channels: int, modes: int = 16, width: int = 64, initial_step: int = 10):
        super().__init__()
        self.modes1 = modes
        self.width = width
        self.padding = 2
        self.fc0 = nn.Linear(initial_step * num_channels + 1, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = F.pad(x, [0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.unsqueeze(-2)


def load_fno_checkpoint(checkpoint_path: str, device: torch.device) -> FNO1d:
    """Load PDEBench FNO model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Infer model config from state dict
    width = state_dict["fc0.weight"].shape[0]
    input_features = state_dict["fc0.weight"].shape[1]
    modes = state_dict["conv0.weights1"].shape[2]
    initial_step = input_features - 1  # subtract 1 for grid coordinate

    model = FNO1d(num_channels=1, modes=modes, width=width, initial_step=initial_step)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_autoregressive_inference(
    model: FNO1d,
    initial_conditions: np.ndarray,
    x_coords: np.ndarray,
    t_coords: np.ndarray,
    device: torch.device,
    batch_size: int = 50,
) -> np.ndarray:
    """Run autoregressive FNO inference.

    Args:
        model: Trained FNO1d model.
        initial_conditions: Shape (N, initial_step, spatial) — first 10 time steps.
        x_coords: Shape (spatial,) — x coordinates.
        t_coords: Shape (total_steps,) — time coordinates for all 200 steps.
        device: Torch device.
        batch_size: Inference batch size.

    Returns:
        predictions: Shape (N, 200, 256).
    """
    n_samples = initial_conditions.shape[0]
    initial_step = initial_conditions.shape[1]
    spatial_size = initial_conditions.shape[2]
    total_steps = len(t_coords)

    # Normalize x to [0, 1]
    x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
    grid = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

    predictions = np.zeros((n_samples, total_steps, spatial_size), dtype=np.float32)
    predictions[:, :initial_step, :] = initial_conditions

    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_data = initial_conditions[batch_start:batch_end]  # (B, 10, 256)

        # Prepare input: (B, spatial, initial_step)
        xx = torch.tensor(batch_data, dtype=torch.float32).permute(0, 2, 1).to(device)
        batch_grid = grid.expand(batch_end - batch_start, -1, -1)

        batch_preds = [batch_data[:, i, :] for i in range(initial_step)]

        with torch.no_grad():
            for t in range(initial_step, total_steps):
                inp = xx.reshape(xx.shape[0], xx.shape[1], -1)  # (B, spatial, initial_step*1)
                pred = model(inp, batch_grid)  # (B, spatial, 1, 1)
                pred_np = pred.squeeze(-1).squeeze(-1).cpu().numpy()  # (B, spatial)
                batch_preds.append(pred_np)

                # Shift window: drop oldest, append prediction
                xx = torch.cat((xx[:, :, 1:], pred.squeeze(-1).squeeze(-1).unsqueeze(-1)), dim=2)

        batch_predictions = np.stack(batch_preds, axis=1)  # (B, 200, 256)
        predictions[batch_start:batch_end] = batch_predictions

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="FNO inference for Task 1.")
    parser.add_argument("--checkpoint", default="checkpoints/extracted/1D_Burgers_Sols_Nu0.001_FNO.pt")
    parser.add_argument("--input", default="data/data_and_sample_submission/data_and_sample_submission/train_val_test_init/task1_test.hdf5")
    parser.add_argument("--output", default="runs/task1-fno-pred/task1_pred.hdf5")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_fno_checkpoint(args.checkpoint, device)

    # Load data
    print(f"Loading input: {args.input}")
    with h5py.File(args.input, "r") as f:
        tensor = f["tensor"][:]  # (N, T, 256)
        x_coords = f["x-coordinate"][:]
        t_coords_raw = f["t-coordinate"][:]

    # Only use first 10 time steps as initial conditions
    initial_step = 10
    initial_conditions = tensor[:, :initial_step, :]
    print(f"Full data shape: {tensor.shape}, using first {initial_step} steps as IC")
    print(f"x coords: {x_coords.shape}, range [{x_coords.min():.4f}, {x_coords.max():.4f}]")
    print(f"t coords (input): {t_coords_raw.shape}, range [{t_coords_raw.min():.4f}, {t_coords_raw.max():.4f}]")

    # Build full time coordinate array (200 steps)
    # Use validation t_coords as reference if available, otherwise interpolate
    val_path = args.input.replace("task1_test.hdf5", "task1_val.hdf5")
    if Path(val_path).exists():
        with h5py.File(val_path, "r") as f:
            t_coords_full = f["t-coordinate"][:]
        print(f"Using val t-coordinates: {t_coords_full.shape}")
    else:
        t_coords_full = np.linspace(t_coords_raw[0], t_coords_raw[-1] * 20, 200).astype(np.float32)
        print(f"Interpolated t-coordinates: {t_coords_full.shape}")

    # Run inference
    print("Running autoregressive inference...")
    t0 = time.time()
    predictions = run_autoregressive_inference(
        model, initial_conditions, x_coords, t_coords_full, device, args.batch_size
    )
    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s, output shape: {predictions.shape}")

    # Save predictions
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("prediction", data=predictions)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
