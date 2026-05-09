import importlib.util
from pathlib import Path

import h5py
import numpy as np


def _load_baseline_module():
    path = Path(__file__).resolve().parents[1] / "code" / "baseline_stub.py"
    spec = importlib.util.spec_from_file_location("baseline_stub", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_baseline_autodetects_official_tensor_dataset(tmp_path):
    baseline = _load_baseline_module()
    init = np.random.default_rng(1).normal(size=(2, 10, 256)).astype(np.float32)
    input_path = tmp_path / "task1_test.hdf5"
    output_path = tmp_path / "task1_pred.hdf5"
    with h5py.File(input_path, "w") as h5:
        h5.create_dataset("tensor", data=init)

    baseline.copy_initial_condition_baseline(input_path, output_path)

    with h5py.File(output_path, "r") as h5:
        pred = h5["prediction"][:]
    assert pred.shape == (2, 200, 256)
    np.testing.assert_allclose(pred[:, :10, :], init)
    np.testing.assert_allclose(pred[:, 10:, :], np.repeat(init[:, 9:10, :], 190, axis=1))
