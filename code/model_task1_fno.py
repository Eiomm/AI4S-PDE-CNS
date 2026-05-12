from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D Fourier layer matching the official PDEBench Burgers FNO checkpoint."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.weights1 = nn.Parameter(
            torch.empty(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            x.shape[0],
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes] = torch.einsum(
            "bix,iox->box",
            x_ft[:, :, : self.modes],
            self.weights1,
        )
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNO1d(nn.Module):
    """Minimal FNO1d architecture for `1D_Burgers_Sols_Nu0.001_FNO.pt`.

    The checkpoint expects 10 input frames plus one spatial grid coordinate,
    hence the first linear layer has 11 input channels.
    """

    def __init__(self, modes: int = 12, width: int = 20, initial_step: int = 10):
        super().__init__()
        self.initial_step = initial_step
        self.fc0 = nn.Linear(initial_step + 1, width)
        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.initial_step:
            raise ValueError(f"expected input shape (batch, x_points, {self.initial_step}), got {tuple(x.shape)}")
        grid = torch.linspace(0, 1, x.shape[1], device=x.device, dtype=x.dtype)
        grid = grid.view(1, x.shape[1], 1).repeat(x.shape[0], 1, 1)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


def load_task1_fno(checkpoint_path: str | Path, device: torch.device | str = "cpu") -> FNO1d:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model = FNO1d().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def rollout(model: FNO1d, initial: torch.Tensor, total_steps: int = 200) -> torch.Tensor:
    """Autoregressively extend `(batch, 10, 256)` initial conditions to 200 frames."""

    if initial.ndim != 3 or initial.shape[1:] != (10, 256):
        raise ValueError(f"expected initial shape (batch, 10, 256), got {tuple(initial.shape)}")
    if total_steps < initial.shape[1]:
        raise ValueError("total_steps must be at least the number of initial frames")

    frames = [initial[:, idx, :] for idx in range(initial.shape[1])]
    window = initial.permute(0, 2, 1).contiguous()
    for _ in range(initial.shape[1], total_steps):
        next_frame = model(window).squeeze(-1)
        frames.append(next_frame)
        window = torch.cat((window[:, :, 1:], next_frame.unsqueeze(-1)), dim=-1)
    return torch.stack(frames, dim=1)
