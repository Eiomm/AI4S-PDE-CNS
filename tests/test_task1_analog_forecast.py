import json

import h5py
import numpy as np

from agent.task1_analog_forecast import (
    AnalogSearchConfig,
    estimate_burgers_nu,
    extract_analog_features,
    run_task1_analog_validation,
)


def _write_hdf5(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("tensor", data=data.astype(np.float32))
        h5.create_dataset("x-coordinate", data=np.linspace(0.0, 1.0, data.shape[2], endpoint=False, dtype=np.float32))
        h5.create_dataset("t-coordinate", data=np.linspace(0.0, 1.0, data.shape[1], dtype=np.float32))


def test_analog_validation_uses_nearest_initial_condition(tmp_path):
    raw = np.zeros((3, 200, 256), dtype=np.float32)
    raw[0, :10, :] = 0.0
    raw[0, 10:, :] = 1.0
    raw[1, :10, :] = 2.0
    raw[1, 10:, :] = 5.0
    raw[2, :10, :] = -3.0
    raw[2, 10:, :] = -7.0
    query = raw[1:2].copy()
    raw_path = tmp_path / "raw" / "1D_Burgers_Sols_Nu0.1.hdf5"
    target_path = tmp_path / "task1_val.hdf5"
    _write_hdf5(raw_path, raw)
    _write_hdf5(target_path, query)

    summary_path = run_task1_analog_validation(
        run_dir=tmp_path / "runs" / "analog",
        target_hdf5=target_path,
        raw_hdf5=[raw_path],
        config=AnalogSearchConfig(top_k=1, max_candidates_per_file=3, spatial_size=256),
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["metrics"]["mse"] == 0.0
    assert summary["neighbor_records"][0][0]["sample_index"] == 1
    with h5py.File(tmp_path / "runs" / "analog" / "task1_val_pred.hdf5", "r") as h5:
        pred = h5["prediction"][:]
    assert np.allclose(pred, query)


def test_estimate_burgers_nu_returns_finite_values():
    x = np.linspace(0.0, 1.0, 64, endpoint=False, dtype=np.float32)
    t = np.linspace(0.0, 0.05, 10, dtype=np.float32)
    values = np.stack([0.1 * np.sin(2.0 * np.pi * (x - step)) for step in t], axis=0).astype(np.float32)

    estimated = estimate_burgers_nu(values, x, t)

    assert np.isfinite(estimated)


def test_analog_features_keep_amplitude_by_default():
    initial = np.zeros((2, 10, 256), dtype=np.float32)
    initial[1] = 2.0

    features = extract_analog_features(initial, "initial", normalize_per_sample=False)
    normalized = extract_analog_features(initial, "initial", normalize_per_sample=True)

    assert features[1].mean() > features[0].mean()
    assert np.isclose(normalized[1].mean(), normalized[0].mean(), atol=1.0e-5)
