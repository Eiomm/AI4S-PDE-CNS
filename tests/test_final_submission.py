from pathlib import Path

import h5py
import numpy as np

from agent.final_submission import create_final_submission
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
