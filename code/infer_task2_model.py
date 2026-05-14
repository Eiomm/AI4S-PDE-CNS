"""Inference for Task2 train-from-scratch checkpoints.

Inference only reads task2_test.h5/tensor and a Task2-specific checkpoint. It
does not require or consume test viscosity/Nu values.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from train_task2_models import (  # noqa: E402
    INPUT_STEPS,
    OUTPUT_STEPS,
    SPATIAL_SIZE,
    build_model,
    load_task2_tensor,
    require_torch,
    validate_task2_checkpoint_path,
)


def load_test_initial(path: str | Path) -> np.ndarray:
    tensor = load_task2_tensor(path, require_target=False)
    if tensor.shape[1:] != (INPUT_STEPS, SPATIAL_SIZE):
        raise ValueError(f"Task2 test tensor must have shape (N, 10, 256), got {tensor.shape}")
    return tensor.astype(np.float32)


def write_prediction(path: str | Path, prediction: np.ndarray, *, dataset_key: str = "prediction") -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f"{output.name}.tmp")
    if temp.exists():
        temp.unlink()
    with h5py.File(temp, "w") as h5:
        h5.create_dataset(dataset_key, data=np.asarray(prediction, dtype=np.float32))
    temp.replace(output)
    return output


def load_checkpoint_model(checkpoint_path: str | Path, *, device: str):
    torch = require_torch()
    checkpoint_path = validate_task2_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        checkpoint["model_name"],
        hidden_channels=int(checkpoint["hidden_channels"]),
        modes=int(checkpoint["modes"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def run_inference(
    *,
    checkpoint_path: str | Path,
    input_path: str | Path = "data/Task2/task2_test.h5",
    output_path: str | Path = "runs/task2-models/task2_pred.hdf5",
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, object]:
    torch = require_torch()
    initial = load_test_initial(input_path)
    model, checkpoint = load_checkpoint_model(checkpoint_path, device=device)
    predictions: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, initial.shape[0], int(batch_size)):
            batch = torch.from_numpy(initial[start : start + int(batch_size)]).to(device=device, dtype=torch.float32)
            prediction = model(batch).cpu().numpy()
            predictions.append(prediction)
    prediction = np.concatenate(predictions, axis=0).astype(np.float32)
    if prediction.shape != (initial.shape[0], OUTPUT_STEPS, SPATIAL_SIZE):
        raise ValueError(f"Prediction must have shape (N, 200, 256), got {prediction.shape}")
    prediction[:, :INPUT_STEPS, :] = initial
    output = write_prediction(output_path, prediction)
    return {
        "checkpoint_path": str(validate_task2_checkpoint_path(checkpoint_path)),
        "model_name": checkpoint["model_name"],
        "input_path": str(input_path),
        "output_path": str(output),
        "num_samples": int(prediction.shape[0]),
        "inference_time": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task2 model checkpoint inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", default="data/Task2/task2_test.h5")
    parser.add_argument("--output", default="runs/task2-models/task2_pred.hdf5")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_inference(
        checkpoint_path=args.checkpoint,
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
