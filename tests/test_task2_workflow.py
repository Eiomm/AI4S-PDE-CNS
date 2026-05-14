import h5py
import numpy as np

from agent.pde_tasks import TaskSpec, task2_spec
from agent.submission import validate_initial_condition, validate_submission
from agent.task2_workflow import Task2PersistenceWorkflow


def _load_task2_baseline_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "code" / "task2_persistence_baseline.py"
    spec = importlib.util.spec_from_file_location("task2_persistence_baseline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_hdf5(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("tensor", data=data.astype(np.float32))
        h5.create_dataset("x_coordinate", data=np.linspace(0.0, 1.0, data.shape[2], endpoint=False, dtype=np.float32))
        h5.create_dataset("t_coordinate", data=np.linspace(0.0, 1.0, data.shape[1], dtype=np.float32))


def _minimal_code_and_methodology(root):
    code_dir = root / "code"
    code_dir.mkdir()
    source = __import__("pathlib").Path(__file__).resolve().parents[1] / "code" / "task2_persistence_baseline.py"
    (code_dir / "task2_persistence_baseline.py").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    methodology_path = root / "methodology.pdf"
    methodology_path.write_bytes(b"%PDF-1.4\n% placeholder\n")
    return code_dir, methodology_path


def _fake_task2_spec(root):
    data_dir = root / "data" / "Task2"
    test = np.random.default_rng(4).normal(size=(3, 10, 256)).astype(np.float32)
    val = np.concatenate([test[:2], np.ones((2, 200, 256), dtype=np.float32)], axis=1)
    _write_hdf5(data_dir / "task2_test.h5", test)
    _write_hdf5(data_dir / "task2_val.h5", val)
    return TaskSpec(
        task_id="task2",
        test_input_path=data_dir / "task2_test.h5",
        validation_target_path=data_dir / "task2_val.h5",
        initial_condition_path=data_dir / "task2_test.h5",
        output_shape=(None, 200, 256),
        prediction_name="task2_pred.hdf5",
        time_budget_seconds=3600,
    )


def test_task2_spec_points_to_local_task2_data():
    spec = task2_spec(project_root="D:/Study/AI4S-PDE-CNS")

    assert spec.task_id == "task2"
    assert spec.test_input_path.as_posix().endswith("data/Task2/task2_test.h5")
    assert spec.validation_target_path.as_posix().endswith("data/Task2/task2_val.h5")
    assert spec.output_shape == (None, 200, 256)


def test_task2_persistence_prediction_copies_initial_and_repeats_last_frame():
    baseline = _load_task2_baseline_module()
    initial = np.random.default_rng(5).normal(size=(2, 10, 256)).astype(np.float32)

    pred = baseline.persistence_prediction(initial, output_steps=200)

    assert pred.shape == (2, 200, 256)
    assert np.allclose(pred[:, :10, :], initial)
    assert np.allclose(pred[:, 10:, :], initial[:, -1:, :])


def test_task2_workflow_validates_and_packages_persistence_submission(tmp_path):
    spec = _fake_task2_spec(tmp_path)
    code_dir, methodology_path = _minimal_code_and_methodology(tmp_path)
    workflow = Task2PersistenceWorkflow(
        spec=spec,
        run_root=tmp_path / "runs",
        code_dir=code_dir,
        methodology_path=methodology_path,
        project_root=tmp_path,
    )

    validation = workflow.run_validation()
    submission = workflow.run_test_submission()

    assert validation.success is True
    assert validation.metrics["num_samples"] == 2
    assert submission.success is True
    assert submission.zip_path is not None
    report = validate_submission(submission.run_dir)
    assert report.tasks == ["task2"]
    validate_initial_condition(submission.run_dir / "task2_pred.hdf5", spec.test_input_path)
