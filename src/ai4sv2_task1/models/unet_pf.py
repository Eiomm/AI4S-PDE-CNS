from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn


class UNet1d(nn.Module):
    def __init__(self, in_channels: int = 10, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        features = init_features
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")
        self.upconv4 = nn.ConvTranspose1d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")
        self.conv = nn.Conv1d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.decoder4(torch.cat((self.upconv4(bottleneck), enc4), dim=1))
        dec3 = self.decoder3(torch.cat((self.upconv3(dec4), enc3), dim=1))
        dec2 = self.decoder2(torch.cat((self.upconv2(dec3), enc2), dim=1))
        dec1 = self.decoder1(torch.cat((self.upconv1(dec2), enc1), dim=1))
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels: int, features: int, name: str) -> nn.Sequential:
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", nn.Conv1d(in_channels, features, kernel_size=3, padding=1, bias=False)),
                    (name + "norm1", nn.BatchNorm1d(features)),
                    (name + "tanh1", nn.Tanh()),
                    (name + "conv2", nn.Conv1d(features, features, kernel_size=3, padding=1, bias=False)),
                    (name + "norm2", nn.BatchNorm1d(features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )


def load_unet_pf_checkpoint(checkpoint_path: str | Path, device: torch.device | str) -> UNet1d:
    ckpt = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    in_channels = int(state_dict["encoder1.enc1conv1.weight"].shape[1])
    init_features = int(state_dict["encoder1.enc1conv1.weight"].shape[0])
    out_channels = int(state_dict["conv.weight"].shape[0])
    model = UNet1d(in_channels=in_channels, out_channels=out_channels, init_features=init_features)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def rollout_unet_pf(
    model: UNet1d,
    initial: np.ndarray,
    t_coords: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    n_samples, initial_step, spatial_size = initial.shape
    total_steps = len(t_coords)
    prediction = np.zeros((n_samples, total_steps, spatial_size), dtype=np.float32)
    prediction[:, :initial_step, :] = initial
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = initial[start:end]
        window = torch.tensor(batch, dtype=torch.float32, device=device)
        frames = [batch[:, i, :] for i in range(initial_step)]
        for _ in range(initial_step, total_steps):
            pred = model(window).squeeze(1)
            frames.append(pred.cpu().numpy())
            window = torch.cat((window[:, 1:, :], pred.unsqueeze(1)), dim=1)
        prediction[start:end] = np.stack(frames, axis=1)
    return prediction
