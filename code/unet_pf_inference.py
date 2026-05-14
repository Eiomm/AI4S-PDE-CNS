"""PDEBench U-Net push-forward inference for Task 1 Burgers checkpoints."""

from __future__ import annotations

import argparse
import time
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn


class UNet1d(nn.Module):
    """1D U-Net architecture matching PDEBench `Unet-PF-20` checkpoints."""

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

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels: int, features: int, name: str) -> nn.Sequential:
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )


def load_unet_pf_checkpoint(checkpoint_path: str | Path, device: torch.device) -> UNet1d:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    in_channels = int(state_dict["encoder1.enc1conv1.weight"].shape[1])
    init_features = int(state_dict["encoder1.enc1conv1.weight"].shape[0])
    out_channels = int(state_dict["conv.weight"].shape[0])
    model = UNet1d(in_channels=in_channels, out_channels=out_channels, init_features=init_features)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_autoregressive_unet_inference(
    model: UNet1d,
    initial_conditions: np.ndarray,
    t_coords: np.ndarray,
    device: torch.device,
    batch_size: int = 50,
) -> np.ndarray:
    n_samples = initial_conditions.shape[0]
    initial_step = initial_conditions.shape[1]
    spatial_size = initial_conditions.shape[2]
    total_steps = len(t_coords)
    predictions = np.zeros((n_samples, total_steps, spatial_size), dtype=np.float32)
    predictions[:, :initial_step, :] = initial_conditions

    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_data = initial_conditions[batch_start:batch_end]
        window = torch.tensor(batch_data, dtype=torch.float32, device=device)
        batch_preds = [batch_data[:, i, :] for i in range(initial_step)]

        with torch.no_grad():
            for _ in range(initial_step, total_steps):
                pred = model(window)
                pred_np = pred.squeeze(1).cpu().numpy()
                batch_preds.append(pred_np)
                window = torch.cat((window[:, 1:, :], pred), dim=1)

        predictions[batch_start:batch_end] = np.stack(batch_preds, axis=1)

    return predictions


def _load_input(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        tensor = h5["tensor"][:]
        t_coords_raw = h5["t-coordinate"][:]
    val_path = Path(path).with_name("task1_val.hdf5")
    if val_path.exists():
        with h5py.File(val_path, "r") as h5:
            return tensor, h5["t-coordinate"][:]
    return tensor, np.linspace(t_coords_raw[0], t_coords_raw[-1] * 20, 200).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unet-PF-20 inference for Task 1.")
    parser.add_argument("--checkpoint", default="checkpoints/extracted/1D_Burgers_Sols_Nu0.001_Unet-PF-20.pt")
    parser.add_argument("--input", default="data/Task1/task1_test.hdf5")
    parser.add_argument("--output", default="runs/task1-unet-pf20/task1_pred.hdf5")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_unet_pf_checkpoint(args.checkpoint, device)
    tensor, t_coords = _load_input(args.input)
    initial = tensor[:, :10, :]
    print(f"Input shape: {tensor.shape}, output steps: {len(t_coords)}")

    t0 = time.time()
    predictions = run_autoregressive_unet_inference(model, initial, t_coords, device, args.batch_size)
    print(f"Inference done in {time.time() - t0:.1f}s, output shape: {predictions.shape}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("prediction", data=predictions.astype(np.float32))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
