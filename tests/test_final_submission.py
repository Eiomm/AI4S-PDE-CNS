import json
import csv
from pathlib import Path

import h5py
import numpy as np
import pytest

from agent.final_submission import _prediction_shape, create_final_submission
from agent.pde_journal import CandidatePlan, ExperimentJournal
from agent.submission import validate_initial_condition, validate_submission


def _write_hdf5(path: Path, data: np.ndarray, *, task2: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("tensor", data=data.astype(np.float32))
        if task2:
            h5.create_dataset("x_coordinate", data=np.linspace(0.0, 1.0, data.shape[2], dtype=np.float32))
            h5.create_dataset("t_coordinate", data=np.linspace(0.0, 1.0, data.shape[1], dtype=np.float32))
        else:
            h5.create_dataset("x-coordinate", data=np.linspace(0.0, 1.0, data.shape[2], dtype=np.float32))
            h5.create_dataset("t-coordinate", data=np.linspace(0.0, 1.0, data.shape[1], dtype=np.float32))


def _write_submission_task_run(root: Path, task: str) -> None:
    root.mkdir(parents=True)
    with h5py.File(root / f"{task}_pred.hdf5", "w") as h5:
        h5.create_dataset("tensor", data=np.zeros((2, 200, 256), dtype=np.float32))
    with (root / f"{task}_time.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["train_time", "inference_time"])
        writer.writeheader()
        writer.writerow({"train_time": "1.0", "inference_time": "0.5"})
    (root / f"{task}_logs.log").write_text(
        json.dumps({"timestamp": "2026-05-15T00:00:00+08:00", "elapsed_seconds": 1.0, "response": {}})
        + "\n",
        encoding="utf-8",
    )
    (root / "code").mkdir()
    (root / "code" / f"{task}_infer.py").write_text(f"print('{task}')\n", encoding="utf-8")


def test_create_final_submission_packages_task1_and_task2_scaffold(tmp_path):
    rng = np.random.default_rng(42)
    task1_initial = rng.normal(size=(3, 10, 256)).astype(np.float32)
    task1_val = rng.normal(size=(3, 200, 256)).astype(np.float32)
    task1_val[:, :10, :] = task1_initial
    task2_initial = rng.normal(size=(4, 10, 256)).astype(np.float32)
    task2_val = rng.normal(size=(4, 210, 256)).astype(np.float32)
    task2_val[:, :10, :] = task2_initial

    _write_hdf5(tmp_path / "data" / "Task1" / "task1_test.hdf5", task1_initial)
    _write_hdf5(tmp_path / "data" / "Task1" / "task1_val.hdf5", task1_val)
    _write_hdf5(tmp_path / "data" / "Task2" / "task2_test.h5", task2_initial, task2=True)
    _write_hdf5(tmp_path / "data" / "Task2" / "task2_val.h5", task2_val, task2=True)

    code_dir = tmp_path / "code"
    code_dir.mkdir()
    source = Path(__file__).resolve().parents[1] / "code" / "task2_persistence_baseline.py"
    (code_dir / "task2_persistence_baseline.py").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    methodology_path = tmp_path / "docs" / "methodology.pdf"
    methodology_path.parent.mkdir()
    methodology_path.write_bytes(b"%PDF-1.4\n% methodology\n")

    def provider(input_path, weights, output_steps):
        with h5py.File(input_path, "r") as h5:
            initial = h5["tensor"][:].astype(np.float32)
        prediction = np.zeros((initial.shape[0], output_steps, initial.shape[2]), dtype=np.float32)
        prediction[:, :10, :] = initial
        prediction[:, 10:, :] = initial[:, -1:, :]
        return prediction

    report = create_final_submission(
        project_root=tmp_path,
        run_root=tmp_path / "runs",
        run_name="final-test",
        code_dir=code_dir,
        methodology_path=methodology_path,
        prediction_provider=provider,
    )
    run_dir = tmp_path / "runs" / "final-test"

    assert report["tasks"] == ["task1", "task2"]
    assert (run_dir / "pred.zip").exists()
    assert validate_submission(run_dir).tasks == ["task1", "task2"]
    validate_initial_condition(run_dir / "task1_pred.hdf5", tmp_path / "data" / "Task1" / "task1_test.hdf5")
    validate_initial_condition(run_dir / "task2_pred.hdf5", tmp_path / "data" / "Task2" / "task2_test.h5")
    with h5py.File(run_dir / "task1_pred.hdf5", "r") as h5:
        assert h5["tensor"].shape == (3, 200, 256)
    with h5py.File(run_dir / "task2_pred.hdf5", "r") as h5:
        assert h5["tensor"].shape == (4, 200, 256)


def test_create_final_submission_cleans_stale_task_logs_before_regenerating(tmp_path):
    rng = np.random.default_rng(7)
    task1_initial = rng.normal(size=(2, 10, 256)).astype(np.float32)
    task1_val = rng.normal(size=(2, 200, 256)).astype(np.float32)
    task1_val[:, :10, :] = task1_initial
    task2_initial = rng.normal(size=(2, 10, 256)).astype(np.float32)
    task2_val = rng.normal(size=(2, 210, 256)).astype(np.float32)
    task2_val[:, :10, :] = task2_initial
    _write_hdf5(tmp_path / "data" / "Task1" / "task1_test.hdf5", task1_initial)
    _write_hdf5(tmp_path / "data" / "Task1" / "task1_val.hdf5", task1_val)
    _write_hdf5(tmp_path / "data" / "Task2" / "task2_test.h5", task2_initial, task2=True)
    _write_hdf5(tmp_path / "data" / "Task2" / "task2_val.h5", task2_val, task2=True)
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    source = Path(__file__).resolve().parents[1] / "code" / "task2_persistence_baseline.py"
    (code_dir / "task2_persistence_baseline.py").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    methodology_path = tmp_path / "docs" / "methodology.pdf"
    methodology_path.parent.mkdir()
    methodology_path.write_bytes(b"%PDF-1.4\n% methodology\n")
    stale = tmp_path / "runs" / "final-test"
    stale.mkdir(parents=True)
    _write_hdf5(stale / "task2_pred.hdf5", np.zeros((2, 200, 256), dtype=np.float32))
    (stale / "task2_time.csv").write_text("train_time,inference_time\n0,0\n", encoding="utf-8")
    (stale / "task2_logs.log").write_text(
        json.dumps(
            {
                "timestamp": "2026-05-14T00:00:00+08:00",
                "elapsed_seconds": 0.0,
                "response": {
                    "action": "write_code_file",
                    "path": "code/task2_persistence_baseline.py",
                    "sha256": "stale",
                    "content": "stale",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def provider(input_path, weights, output_steps):
        with h5py.File(input_path, "r") as h5:
            initial = h5["tensor"][:].astype(np.float32)
        prediction = np.zeros((initial.shape[0], output_steps, initial.shape[2]), dtype=np.float32)
        prediction[:, :10, :] = initial
        prediction[:, 10:, :] = initial[:, -1:, :]
        return prediction

    report = create_final_submission(
        project_root=tmp_path,
        run_root=tmp_path / "runs",
        run_name="final-test",
        code_dir=code_dir,
        methodology_path=methodology_path,
        prediction_provider=provider,
    )

    assert report["tasks"] == ["task1", "task2"]
    assert validate_submission(stale).valid is True


def test_create_final_submission_can_merge_existing_task_runs(tmp_path):
    task1_run = tmp_path / "runs" / "task1-node"
    task2_run = tmp_path / "runs" / "task2-node"
    _write_submission_task_run(task1_run, "task1")
    _write_submission_task_run(task2_run, "task2")
    methodology_path = tmp_path / "docs" / "methodology.pdf"
    methodology_path.parent.mkdir()
    methodology_path.write_bytes(b"%PDF-1.4\n% methodology\n")

    report = create_final_submission(
        project_root=tmp_path,
        run_root=tmp_path / "runs",
        run_name="final-merged",
        methodology_path=methodology_path,
        task1_run=task1_run,
        task2_run=task2_run,
    )

    run_dir = tmp_path / "runs" / "final-merged"
    assert report["tasks"] == ["task1", "task2"]
    assert (run_dir / "pred.zip").is_file()
    assert (run_dir / "code" / "task1_infer.py").is_file()
    assert (run_dir / "code" / "task2_infer.py").is_file()
    assert validate_submission(run_dir).tasks == ["task1", "task2"]


def test_create_final_submission_can_require_autonomy_audit(tmp_path):
    task1_run = tmp_path / "runs" / "task1-node"
    _write_submission_task_run(task1_run, "task1")
    methodology_path = tmp_path / "docs" / "methodology.pdf"
    methodology_path.parent.mkdir()
    methodology_path.write_bytes(b"%PDF-1.4\n% methodology\n")
    study = tmp_path / "runs" / "task1" / "autonomous" / "20260515" / "bad-study"
    study.mkdir(parents=True)
    journal = ExperimentJournal(study / "journal.json")
    node = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis="bootstrap should not pass strict audit",
            action_type="finetune_checkpoint",
            params={"temporal_stride": 5},
        )
    )
    journal.update_result(node.id, success=True, metrics={"competition_score_proxy": 1.0}, artifacts={})
    (study / "planner_logs.log").write_text(
        json.dumps(
            {
                "timestamp": "2026-05-15T00:00:00+08:00",
                "elapsed_seconds": 1.0,
                "provider": "bootstrap",
                "model": "task1-bootstrap",
                "messages": [{"role": "user", "content": "preset"}],
                "response": {"content": "{}"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="autonomy audit failed"):
        create_final_submission(
            project_root=tmp_path,
            run_root=tmp_path / "runs",
            run_name="final-audited",
            methodology_path=methodology_path,
            task1_run=task1_run,
            require_autonomy_audit=True,
            task1_study_dir=study,
        )


def test_prediction_shape_accepts_prediction_dataset(tmp_path):
    path = tmp_path / "pred.hdf5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("prediction", data=np.zeros((2, 200, 256), dtype=np.float32))

    assert _prediction_shape(path) == (2, 200, 256)
