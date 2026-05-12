import json
from pathlib import Path

import h5py
import numpy as np

from agent.submission import validate_initial_condition, validate_submission
from agent.zero_submission import create_task1_zero_submission


def test_create_task1_zero_submission_writes_valid_bundle(tmp_path):
    init = np.random.default_rng(2).normal(size=(4, 10, 256)).astype(np.float32)
    input_path = tmp_path / "task1_test.hdf5"
    output_dir = tmp_path / "run"
    code_dir = Path(__file__).resolve().parents[1] / "code"
    with h5py.File(input_path, "w") as h5:
        h5.create_dataset("tensor", data=init)

    create_task1_zero_submission(input_path=input_path, output_dir=output_dir, code_dir=code_dir)

    report = validate_submission(output_dir)
    validate_initial_condition(output_dir / "task1_pred.hdf5", input_path)
    assert report.tasks == ["task1"]
    with h5py.File(output_dir / "task1_pred.hdf5", "r") as h5:
        assert list(h5.keys()) == ["tensor"]
    assert (output_dir / "code" / "baseline_stub.py").exists()
    assert (output_dir / "methodology.pdf").exists()
    time_csv = (output_dir / "task1_time.csv").read_text(encoding="utf-8")
    assert "train_time,inference_time" in time_csv
    log_record = json.loads((output_dir / "task1_logs.log").read_text(encoding="utf-8").splitlines()[0])
    assert log_record["provider"] == "local"
    assert log_record["model"] == "zero-train-baseline"
