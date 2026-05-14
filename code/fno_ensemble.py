"""FNO prediction helper for official Task 1 checkpoints."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import sys

# Import from fno_inference.py in same directory
sys.path.insert(0, str(Path(__file__).parent))


def combine_predictions(
    predictions: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Combine model predictions with optional normalized weights."""

    if not predictions:
        raise ValueError("At least one prediction array is required")
    if weights is None:
        return np.mean(predictions, axis=0).astype(np.float32)
    if len(weights) != len(predictions):
        raise ValueError("weights length must match checkpoints length")
    weights_array = np.asarray(weights, dtype=np.float64)
    total = float(weights_array.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    weights_array = weights_array / total
    combined = np.zeros_like(predictions[0], dtype=np.float64)
    for weight, pred in zip(weights_array, predictions):
        combined += float(weight) * pred
    return combined.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="FNO ensemble inference.")
    parser.add_argument("--checkpoints", nargs="+", default=[
        "checkpoints/extracted/1D_Burgers_Sols_Nu0.001_FNO.pt",
    ])
    parser.add_argument("--input", default="data/data_and_sample_submission/data_and_sample_submission/train_val_test_init/task1_test.hdf5")
    parser.add_argument("--output", default="runs/task1-fno-ensemble/task1_pred.hdf5")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    args = parser.parse_args()

    import torch
    from fno_inference import load_fno_checkpoint, run_autoregressive_inference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print(f"Loading input: {args.input}")
    with h5py.File(args.input, "r") as f:
        tensor = f["tensor"][:]
        x_coords = f["x-coordinate"][:]
        t_coords_raw = f["t-coordinate"][:]

    initial_step = 10
    initial_conditions = tensor[:, :initial_step, :]
    print(f"Full data shape: {tensor.shape}, using first {initial_step} steps as IC")

    val_path = args.input.replace("task1_test.hdf5", "task1_val.hdf5")
    if Path(val_path).exists():
        with h5py.File(val_path, "r") as f:
            t_coords_full = f["t-coordinate"][:]
    else:
        t_coords_full = np.linspace(t_coords_raw[0], t_coords_raw[-1] * 20, 200).astype(np.float32)

    # Run ensemble
    all_preds = []
    for ckpt_path in args.checkpoints:
        print(f"\nLoading: {ckpt_path}")
        model = load_fno_checkpoint(ckpt_path, device)
        print("Running inference...")
        t0 = time.time()
        preds = run_autoregressive_inference(
            model, initial_conditions, x_coords, t_coords_full, device, args.batch_size
        )
        print(f"Done in {time.time() - t0:.1f}s")
        all_preds.append(preds)

    # Average or weighted-average predictions
    ensemble_pred = combine_predictions(all_preds, args.weights)
    if args.weights is not None:
        print(f"Using weights: {args.weights}")
    print(f"\nEnsemble shape: {ensemble_pred.shape}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("prediction", data=ensemble_pred)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
