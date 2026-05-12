from pathlib import Path

import h5py
import numpy as np


def _write_hdf5(path: Path, key: str, data: np.ndarray) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset(key, data=data.astype(np.float32))


def test_check_prediction_file_reports_first_ten_match(tmp_path):
    from importlib.util import module_from_spec, spec_from_file_location

    module_path = Path(__file__).resolve().parents[1] / "code" / "check_pred_shape.py"
    spec = spec_from_file_location("check_pred_shape", module_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    initial = np.random.default_rng(1).normal(size=(2, 10, 256)).astype(np.float32)
    prediction = np.zeros((2, 200, 256), dtype=np.float32)
    prediction[:, :10, :] = initial
    prediction[:, 10:, :] = initial[:, 9:10, :]
    input_path = tmp_path / "task1_test.hdf5"
    pred_path = tmp_path / "task1_pred.hdf5"
    _write_hdf5(input_path, "tensor", initial)
    _write_hdf5(pred_path, "prediction", prediction)

    report = module.check_prediction_file(pred_path, input_path)

    assert report["shape"] == (2, 200, 256)
    assert report["first_ten_match"] is True
    assert report["max_initial_error"] == 0.0


def test_check_prediction_file_accepts_float_roundoff(tmp_path):
    from importlib.util import module_from_spec, spec_from_file_location

    module_path = Path(__file__).resolve().parents[1] / "code" / "check_pred_shape.py"
    spec = spec_from_file_location("check_pred_shape_roundoff", module_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    initial = np.ones((1, 10, 256), dtype=np.float32)
    prediction = np.zeros((1, 200, 256), dtype=np.float32)
    prediction[:, :10, :] = initial + np.float32(2e-7)
    input_path = tmp_path / "task1_test.hdf5"
    pred_path = tmp_path / "task1_pred.hdf5"
    _write_hdf5(input_path, "tensor", initial)
    _write_hdf5(pred_path, "prediction", prediction)

    report = module.check_prediction_file(pred_path, input_path)

    assert report["first_ten_match"] is True
    assert 0.0 < report["max_initial_error"] < 1e-6
