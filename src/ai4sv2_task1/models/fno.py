from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            x.shape[0],
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = torch.einsum("bix,iox->box", x_ft[:, :, : self.modes1], self.weights1)
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNO1d(nn.Module):
    def __init__(self, num_channels: int = 1, modes: int = 16, width: int = 64, initial_step: int = 10):
        super().__init__()
        self.padding = 2
        self.fc0 = nn.Linear(initial_step * num_channels + 1, width)
        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = F.pad(x, [0, self.padding])
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)
        x = x[..., : -self.padding]
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.unsqueeze(-2)


def load_fno_checkpoint(checkpoint_path: str | Path, device: torch.device | str) -> FNO1d:
    ckpt = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    width = int(state_dict["fc0.weight"].shape[0])
    input_features = int(state_dict["fc0.weight"].shape[1])
    modes = int(state_dict["conv0.weights1"].shape[2])
    model = FNO1d(num_channels=1, modes=modes, width=width, initial_step=input_features - 1)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def rollout_fno(
    model: FNO1d,
    initial: np.ndarray,
    x_coords: np.ndarray,
    t_coords: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    n_samples, initial_step, spatial_size = initial.shape
    total_steps = len(t_coords)
    x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
    grid = torch.tensor(x_norm, dtype=torch.float32, device=device).view(1, spatial_size, 1)
    prediction = np.zeros((n_samples, total_steps, spatial_size), dtype=np.float32)
    prediction[:, :initial_step, :] = initial
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = initial[start:end]
        xx = torch.tensor(batch, dtype=torch.float32, device=device).permute(0, 2, 1).contiguous()
        batch_grid = grid.expand(end - start, -1, -1)
        frames = [batch[:, i, :] for i in range(initial_step)]
        for _ in range(initial_step, total_steps):
            pred = model(xx.reshape(xx.shape[0], xx.shape[1], -1), batch_grid).squeeze(-1).squeeze(-1)
            frames.append(pred.cpu().numpy())
            xx = torch.cat((xx[:, :, 1:], pred.unsqueeze(-1)), dim=2)
        prediction[start:end] = np.stack(frames, axis=1)
    return prediction
