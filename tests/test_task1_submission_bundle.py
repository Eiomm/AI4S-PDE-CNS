import json

import h5py
import numpy as np

from agent.submission import validate_submission
from agent.task1_submission import create_task1_submission_bundle


def _write_hdf5(path, key, data):
    with h5py.File(path, "w") as h5:
        h5.create_dataset(key, data=data.astype(np.float32))


def test_create_task1_submission_bundle_validates_outputs(tmp_path):
    initial = np.random.default_rng(0).normal(size=(2, 10, 256)).astype(np.float32)
    prediction = np.zeros((2, 200, 256), dtype=np.float32)
    prediction[:, :10, :] = initial
    prediction[:, 10:, :] = initial[:, 9:10, :]

    initial_path = tmp_path / "task1_test.hdf5"
    prediction_path = tmp_path / "candidate.hdf5"
    log_path = tmp_path / "task1_logs.log"
    methodology_path = tmp_path / "methodology.pdf"
    code_dir = tmp_path / "code-src"
    output_dir = tmp_path / "submission-run"
    _write_hdf5(initial_path, "tensor", initial)
    _write_hdf5(prediction_path, "prediction", prediction)
    log_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-12T19:00:00+08:00",
                "elapsed_seconds": 1.25,
                "response": {"content": "generated prediction"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    methodology_path.write_bytes(b"%PDF-1.4\n% methodology\n")
    code_dir.mkdir()
    (code_dir / "infer.py").write_text("print('ok')\n", encoding="utf-8")

    result = create_task1_submission_bundle(
        prediction_path=prediction_path,
        initial_path=initial_path,
        output_dir=output_dir,
        code_dir=code_dir,
        log_path=log_path,
        methodology_path=methodology_path,
        train_time=12.5,
        inference_time=3.25,
    )

    assert result == output_dir
    assert (output_dir / "task1_pred.hdf5").exists()
    with h5py.File(output_dir / "task1_pred.hdf5", "r") as h5:
        assert list(h5.keys()) == ["tensor"]
        np.testing.assert_allclose(h5["tensor"][:], prediction)
    assert (output_dir / "task1_logs.log").read_text(encoding="utf-8") == log_path.read_text(encoding="utf-8")
    assert (output_dir / "methodology.pdf").read_bytes() == methodology_path.read_bytes()
    assert (output_dir / "code" / "infer.py").read_text(encoding="utf-8") == "print('ok')\n"
    assert "12.500000,3.250000" in (output_dir / "task1_time.csv").read_text(encoding="utf-8")
    report = validate_submission(output_dir)
    assert report.valid is True
    assert report.tasks == ["task1"]
