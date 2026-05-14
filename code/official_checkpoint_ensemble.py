"""Prediction-level ensemble over the official Task 1 FNO and Unet-PF checkpoints."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from fno_ensemble import combine_predictions  # noqa: E402


def _parse_model_specs(values: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"model spec must use KIND=PATH format, got {value!r}")
        kind, path = value.split("=", 1)
        kind = kind.strip()
        if kind not in {"fno", "unet_pf20"}:
            raise ValueError(f"unsupported official model kind {kind!r}")
        if not path.strip():
            raise ValueError(f"empty checkpoint path for {kind!r}")
        specs.append((kind, path))
    if not specs:
        raise ValueError("at least one model spec is required")
    return specs


def _load_task1_input(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_path = Path(path)
    with h5py.File(input_path, "r") as h5:
        tensor = h5["tensor"][:]
        x_coords = h5["x-coordinate"][:]
        t_coords_raw = h5["t-coordinate"][:]
    val_path = input_path.with_name("task1_val.hdf5")
    if val_path.exists():
        with h5py.File(val_path, "r") as h5:
            return tensor, x_coords, h5["t-coordinate"][:]
    return tensor, x_coords, np.linspace(t_coords_raw[0], t_coords_raw[-1] * 20, 200).astype(np.float32)


def run_official_ensemble(
    *,
    input_path: str | Path,
    model_specs: list[tuple[str, str]],
    weights: list[float] | None,
    batch_size: int,
) -> np.ndarray:
    import torch
    from fno_inference import load_fno_checkpoint, run_autoregressive_inference
    from unet_pf_inference import load_unet_pf_checkpoint, run_autoregressive_unet_inference

    if weights is not None and len(weights) != len(model_specs):
        raise ValueError("weights length must match model specs length")
    tensor, x_coords, t_coords = _load_task1_input(input_path)
    initial = tensor[:, :10, :]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Input shape: {tensor.shape}, output steps: {len(t_coords)}")

    predictions: list[np.ndarray] = []
    for kind, checkpoint_path in model_specs:
        print(f"\nLoading {kind}: {checkpoint_path}")
        started = time.time()
        if kind == "fno":
            model = load_fno_checkpoint(checkpoint_path, device)
            pred = run_autoregressive_inference(model, initial, x_coords, t_coords, device, batch_size)
        elif kind == "unet_pf20":
            model = load_unet_pf_checkpoint(checkpoint_path, device)
            pred = run_autoregressive_unet_inference(model, initial, t_coords, device, batch_size)
        else:
            raise ValueError(f"unsupported official model kind {kind!r}")
        predictions.append(pred.astype(np.float32))
        print(f"{kind} done in {time.time() - started:.1f}s")
    return combine_predictions(predictions, weights)


def main() -> None:
    parser = argparse.ArgumentParser(description="Official Task 1 checkpoint prediction ensemble.")
    parser.add_argument("--models", nargs="+", required=True, help="KIND=PATH entries, with KIND fno or unet_pf20.")
    parser.add_argument("--input", default="data/Task1/task1_test.hdf5")
    parser.add_argument("--output", default="runs/task1-official-ensemble/task1_pred.hdf5")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    args = parser.parse_args()

    model_specs = _parse_model_specs(args.models)
    prediction = run_official_ensemble(
        input_path=args.input,
        model_specs=model_specs,
        weights=args.weights,
        batch_size=args.batch_size,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("prediction", data=prediction.astype(np.float32))
    print(f"\nSaved to {output_path}; shape={prediction.shape}")


if __name__ == "__main__":
    main()
