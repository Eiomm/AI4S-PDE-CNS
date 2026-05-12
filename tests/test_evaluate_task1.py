import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np


def _load_eval_module():
    path = Path(__file__).resolve().parents[1] / "code" / "evaluate_task1.py"
    spec = importlib.util.spec_from_file_location("evaluate_task1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_ensemble_module():
    path = Path(__file__).resolve().parents[1] / "code" / "fno_ensemble.py"
    spec = importlib.util.spec_from_file_location("fno_ensemble", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compute_task1_metrics_splits_initial_forecast_and_long_horizon():
    evaluate = _load_eval_module()
    target = np.zeros((2, 200, 256), dtype=np.float32)
    pred = np.zeros_like(target)
    pred[:, :10, :] = 1.0
    pred[:, 10:105, :] = 2.0
    pred[:, 105:, :] = 3.0

    metrics = evaluate.compute_task1_metrics(pred, target)

    assert metrics["initial_mse"] == 1.0
    assert metrics["forecast_mse"] == 6.5
    assert metrics["long_horizon_mse"] == 9.0
    assert metrics["mse"] == 6.225


def test_evaluate_prediction_file_writes_json_metrics(tmp_path):
    evaluate = _load_eval_module()
    target = np.zeros((1, 200, 256), dtype=np.float32)
    pred = np.zeros_like(target)
    pred[:, 10:, :] = 2.0
    target_path = tmp_path / "task1_val.hdf5"
    pred_path = tmp_path / "task1_pred.hdf5"
    metrics_path = tmp_path / "metrics.json"
    with h5py.File(target_path, "w") as h5:
        h5.create_dataset("tensor", data=target)
    with h5py.File(pred_path, "w") as h5:
        h5.create_dataset("prediction", data=pred)

    metrics = evaluate.evaluate_prediction_file(pred_path, target_path, metrics_path)

    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics == saved
    assert saved["forecast_mse"] == 4.0
    assert saved["num_samples"] == 1


def test_combine_predictions_supports_normalized_weights_without_torch():
    ensemble = _load_ensemble_module()
    first = np.zeros((1, 2, 3), dtype=np.float32)
    second = np.full((1, 2, 3), 10.0, dtype=np.float32)

    combined = ensemble.combine_predictions([first, second], weights=[1.0, 3.0])

    np.testing.assert_allclose(combined, np.full((1, 2, 3), 7.5, dtype=np.float32))
