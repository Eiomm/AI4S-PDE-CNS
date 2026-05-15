from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module(relative_path: str, module_name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_task2_h5(path: Path, data: np.ndarray, nu: np.ndarray | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("tensor", data=data.astype(np.float32))
        if nu is not None:
            h5.create_dataset("nu", data=nu.astype(np.float32))
    return path


def test_task2_data_path_allowlist_rejects_task1_files_and_task1_checkpoints(tmp_path):
    train = _load_module("code/train_task2_models.py", "train_task2_models")

    allowed = tmp_path / "data" / "Task2" / "task2_part0_train.h5"
    task1_data = tmp_path / "data" / "Task1" / "task1_val.hdf5"
    task1_checkpoint = tmp_path / "checkpoints" / "task1_fno.pt"

    assert train.validate_task2_data_path(allowed) == allowed
    with pytest.raises(ValueError, match="Task2"):
        train.validate_task2_data_path(task1_data)
    with pytest.raises(ValueError, match="Task1"):
        train.validate_task2_checkpoint_path(task1_checkpoint)


def test_task2_dataset_reads_initial_and_target_shapes_from_allowlisted_files(tmp_path):
    train = _load_module("code/train_task2_models.py", "train_task2_models")
    rng = np.random.default_rng(42)
    tensor = rng.normal(size=(3, 200, 256)).astype(np.float32)
    path = _write_task2_h5(tmp_path / "data" / "Task2" / "task2_part1_train.h5", tensor)

    dataset = train.Task2TrajectoryDataset([path], sample_limit=2)
    initial, target = dataset[0]

    assert len(dataset) == 2
    assert initial.shape == (10, 256)
    assert target.shape == (200, 256)
    assert np.allclose(initial, tensor[0, :10])
    assert np.allclose(target, tensor[0, :200])


def test_minifno_and_temporal_unet_outputs_match_task2_contract():
    torch = pytest.importorskip("torch")
    train = _load_module("code/train_task2_models.py", "train_task2_models")

    initial = torch.randn(2, 10, 256)
    for model_name in ("minifno", "unet", "minifno_nu"):
        model = train.build_model(model_name, hidden_channels=8, modes=8)
        with torch.no_grad():
            prediction = model(initial)

        assert tuple(prediction.shape) == (2, 200, 256)
        assert torch.allclose(prediction[:, :10, :], initial)


def test_inference_reads_test_tensor_without_nu_dataset(tmp_path):
    infer = _load_module("code/infer_task2_model.py", "infer_task2_model")
    rng = np.random.default_rng(7)
    tensor = rng.normal(size=(4, 10, 256)).astype(np.float32)
    path = _write_task2_h5(tmp_path / "data" / "Task2" / "task2_test.h5", tensor)

    initial = infer.load_test_initial(path)

    assert initial.shape == (4, 10, 256)
    assert np.allclose(initial, tensor)


def test_select_best_candidate_only_promotes_validation_improvement():
    train = _load_module("code/train_task2_models.py", "train_task2_models")
    persistence = {"forecast_mse": 0.04277, "mse": 0.0408}
    worse = {
        "model_name": "minifno",
        "checkpoint_path": "runs/task2-models/task2_minifno.pt",
        "metrics": {"forecast_mse": 0.05, "mse": 0.048},
    }
    better = {
        "model_name": "unet",
        "checkpoint_path": "runs/task2-models/task2_unet.pt",
        "metrics": {"forecast_mse": 0.039, "mse": 0.037},
    }

    assert train.select_best_candidate([worse], persistence) is None
    selected = train.select_best_candidate([worse, better], persistence)

    assert selected["model_name"] == "unet"
    assert selected["checkpoint_path"].endswith("task2_unet.pt")


def test_best_history_record_selects_lowest_validation_forecast_mse():
    train = _load_module("code/train_task2_models.py", "train_task2_models")
    history = [
        {"epoch": 1.0, "forecast_mse": 0.004, "mse": 0.0038},
        {"epoch": 2.0, "forecast_mse": 0.002, "mse": 0.0021},
        {"epoch": 3.0, "forecast_mse": 0.003, "mse": 0.0027},
    ]

    best = train._best_history_record(history)

    assert best["epoch"] == 2.0
    assert best["forecast_mse"] == 0.002


def test_task2_dataset_can_return_nu_for_auxiliary_training(tmp_path):
    train = _load_module("code/train_task2_models.py", "train_task2_models")
    rng = np.random.default_rng(8)
    tensor = rng.normal(size=(3, 200, 256)).astype(np.float32)
    nu = np.array([0.001, 0.01, 0.1], dtype=np.float32)
    path = _write_task2_h5(tmp_path / "data" / "Task2" / "task2_part2_train.h5", tensor, nu=nu)

    dataset = train.Task2TrajectoryDataset([path], include_nu=True)
    initial, target, sample_nu = dataset[1]

    assert initial.shape == (10, 256)
    assert target.shape == (200, 256)
    assert sample_nu == pytest.approx(0.01)


def test_minifno_nu_exposes_auxiliary_nu_prediction_without_changing_inference_contract():
    torch = pytest.importorskip("torch")
    train = _load_module("code/train_task2_models.py", "train_task2_models")
    model = train.build_model("minifno_nu", hidden_channels=8, modes=8)
    initial = torch.randn(2, 10, 256)

    with torch.no_grad():
        prediction = model(initial)
        prediction_with_aux, nu_hat = model(initial, return_aux=True)

    assert tuple(prediction.shape) == (2, 200, 256)
    assert torch.allclose(prediction[:, :10, :], initial)
    assert tuple(prediction_with_aux.shape) == (2, 200, 256)
    assert tuple(nu_hat.shape) == (2,)
