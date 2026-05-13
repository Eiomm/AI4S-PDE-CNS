import csv
import json
import zipfile

import h5py
import numpy as np
import pytest

from agent.code_trace import append_code_trace_log
from agent.submission import (
    SubmissionError,
    default_pack_path,
    pack_submission,
    validate_initial_condition,
    validate_submission,
)


def _write_task_files(root, task="task1", samples=2):
    pred = np.zeros((samples, 200, 256), dtype=np.float32)
    with h5py.File(root / f"{task}_pred.hdf5", "w") as h5:
        h5.create_dataset("tensor", data=pred)
    with (root / f"{task}_time.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["train_time", "inference_time"])
        writer.writeheader()
        writer.writerow({"train_time": "12.5", "inference_time": "1.2"})
    (root / f"{task}_logs.log").write_text(
        json.dumps({"timestamp": "2026-05-09T10:00:00+08:00", "elapsed_seconds": 1}) + "\n",
        encoding="utf-8",
    )
    code_dir = root / "code"
    if code_dir.exists():
        append_code_trace_log(root / f"{task}_logs.log", code_dir)


def test_validate_submission_accepts_minimal_task_bundle(tmp_path):
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "train.py").write_text("print('train')\n", encoding="utf-8")
    (tmp_path / "methodology.pdf").write_bytes(b"%PDF-1.4\n% placeholder\n")
    (tmp_path / "submission.json").write_text(
        json.dumps(
            {
                "submission_id": "team",
                "problem_id": "PDE_Burgers",
                "code_path": "code",
            }
        ),
        encoding="utf-8",
    )
    _write_task_files(tmp_path, "task1")

    report = validate_submission(tmp_path)

    assert report.valid is True
    assert report.tasks == ["task1"]


def test_validate_submission_rejects_missing_methodology_pdf(tmp_path):
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "train.py").write_text("print('train')\n", encoding="utf-8")
    (tmp_path / "submission.json").write_text(
        json.dumps({"submission_id": "team", "problem_id": "PDE_Burgers", "code_path": "code"}),
        encoding="utf-8",
    )
    _write_task_files(tmp_path, "task1")

    with pytest.raises(SubmissionError, match="methodology.pdf"):
        validate_submission(tmp_path)


def test_validate_submission_rejects_wrong_prediction_shape(tmp_path):
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "train.py").write_text("print('train')\n", encoding="utf-8")
    (tmp_path / "methodology.pdf").write_bytes(b"%PDF-1.4\n% placeholder\n")
    (tmp_path / "submission.json").write_text(
        json.dumps({"submission_id": "team", "problem_id": "PDE_Burgers", "code_path": "code"}),
        encoding="utf-8",
    )
    with h5py.File(tmp_path / "task1_pred.hdf5", "w") as h5:
        h5.create_dataset("tensor", data=np.zeros((2, 199, 256), dtype=np.float32))
    (tmp_path / "task1_time.csv").write_text("train_time,inference_time\n1,1\n", encoding="utf-8")
    (tmp_path / "task1_logs.log").write_text(
        json.dumps({"timestamp": "2026-05-09T10:00:00+08:00", "elapsed_seconds": 1}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SubmissionError, match="shape"):
        validate_submission(tmp_path)


def test_validate_submission_rejects_missing_tensor_dataset(tmp_path):
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "train.py").write_text("print('train')\n", encoding="utf-8")
    (tmp_path / "methodology.pdf").write_bytes(b"%PDF-1.4\n% placeholder\n")
    (tmp_path / "submission.json").write_text(
        json.dumps({"submission_id": "team", "problem_id": "PDE_Burgers", "code_path": "code"}),
        encoding="utf-8",
    )
    with h5py.File(tmp_path / "task1_pred.hdf5", "w") as h5:
        h5.create_dataset("prediction", data=np.zeros((2, 200, 256), dtype=np.float32))
    (tmp_path / "task1_time.csv").write_text("train_time,inference_time\n1,1\n", encoding="utf-8")
    (tmp_path / "task1_logs.log").write_text(
        json.dumps({"timestamp": "2026-05-09T10:00:00+08:00", "elapsed_seconds": 1}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SubmissionError, match="tensor"):
        validate_submission(tmp_path)


def test_pack_submission_writes_zip_after_validation(tmp_path):
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "train.py").write_text("print('train')\n", encoding="utf-8")
    (tmp_path / "methodology.pdf").write_bytes(b"%PDF-1.4\n% placeholder\n")
    (tmp_path / "submission.json").write_text(
        json.dumps({"submission_id": "team", "problem_id": "PDE_Burgers", "code_path": "code"}),
        encoding="utf-8",
    )
    _write_task_files(tmp_path, "task1")

    zip_path = pack_submission(tmp_path, tmp_path / "submission.zip")

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    assert "submission.json" in names
    assert "task1_pred.hdf5" in names
    assert "code/train.py" in names


def test_default_pack_path_uses_pred_zip_inside_run_dir(tmp_path):
    run_dir = tmp_path / "runs" / "task1-experiment"
    assert default_pack_path(run_dir) == run_dir / "pred.zip"


def test_validate_initial_condition_accepts_matching_first_ten_frames(tmp_path):
    init = np.random.default_rng(0).normal(size=(3, 10, 256)).astype(np.float32)
    pred = np.zeros((3, 200, 256), dtype=np.float32)
    pred[:, :10, :] = init
    with h5py.File(tmp_path / "task1_test.hdf5", "w") as h5:
        h5.create_dataset("tensor", data=init)
    with h5py.File(tmp_path / "task1_pred.hdf5", "w") as h5:
        h5.create_dataset("prediction", data=pred)

    validate_initial_condition(tmp_path / "task1_pred.hdf5", tmp_path / "task1_test.hdf5")


def test_validate_initial_condition_rejects_mismatched_first_ten_frames(tmp_path):
    init = np.zeros((2, 10, 256), dtype=np.float32)
    pred = np.zeros((2, 200, 256), dtype=np.float32)
    pred[:, 9, 0] = 0.01
    with h5py.File(tmp_path / "task1_test.hdf5", "w") as h5:
        h5.create_dataset("tensor", data=init)
    with h5py.File(tmp_path / "task1_pred.hdf5", "w") as h5:
        h5.create_dataset("prediction", data=pred)

    with pytest.raises(SubmissionError, match="initial condition"):
        validate_initial_condition(tmp_path / "task1_pred.hdf5", tmp_path / "task1_test.hdf5")
