import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from agent.pde_executor import ControlledExperimentExecutor
from agent.pde_journal import CandidatePlan, ExperimentJournal
from agent.run_task2_autonomous_experiment import run_autonomous_task2, task2_bootstrap_train_plan


def test_task2_bootstrap_train_plan_uses_task2_action_and_data():
    plan = task2_bootstrap_train_plan(study_name="smoke")

    assert plan["action_type"] == "task2_train_model"
    assert plan["params"]["model"] == "minifno_nu"
    assert "task2" in plan["params"]["output_dir"]


def test_autonomous_task2_cli_function_runs_with_mock_provider(tmp_path):
    (tmp_path / "code").mkdir()
    config = tmp_path / "mock.yaml"
    config.write_text("provider: mock\nmodel: mock-planner\n", encoding="utf-8")

    summary_path = run_autonomous_task2(
        config_path=config,
        project_root=tmp_path,
        study_name="task2-autonomous-smoke",
        max_iterations=1,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    study_root = tmp_path / "runs" / "task2" / "autonomous"
    matches = list(study_root.glob("*/task2-autonomous-smoke"))
    assert summary["iterations"] == 1
    assert len(matches) == 1
    assert (matches[0] / "journal_report.md").is_file()
    assert (matches[0] / "experiment_results.json").is_file()
    assert (matches[0] / "planner_logs.log").is_file()


def test_autonomous_task2_strict_mode_rejects_bootstrap_train(tmp_path):
    (tmp_path / "code").mkdir()
    config = tmp_path / "mock.yaml"
    config.write_text("provider: mock\nmodel: mock-planner\n", encoding="utf-8")

    with pytest.raises(ValueError, match="strict_autonomy"):
        run_autonomous_task2(
            config_path=config,
            project_root=tmp_path,
            study_name="strict-task2",
            max_iterations=1,
            strict_autonomy=True,
            bootstrap_train=True,
        )


def test_task2_submit_best_packages_trained_checkpoint_candidate(tmp_path, monkeypatch):
    initial = np.random.default_rng(3).normal(size=(2, 10, 256)).astype(np.float32)
    data_dir = tmp_path / "data" / "Task2"
    data_dir.mkdir(parents=True)
    with h5py.File(data_dir / "task2_test.h5", "w") as h5:
        h5.create_dataset("tensor", data=initial)
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "infer_task2_model.py").write_text("print('fake')\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "methodology.pdf").write_bytes(b"%PDF-1.4\n% methodology\n")
    checkpoint = tmp_path / "runs" / "task2" / "models" / "task2_minifno.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake checkpoint")

    def fake_run(command, cwd, capture_output, text, timeout):
        output_path = command[command.index("--output") + 1]
        prediction = np.zeros((initial.shape[0], 200, 256), dtype=np.float32)
        prediction[:, :10, :] = initial
        with h5py.File(output_path, "w") as h5:
            h5.create_dataset("prediction", data=prediction)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"output_path": output_path, "inference_time": 0.25}),
            stderr="",
        )

    monkeypatch.setattr("agent.pde_executor.subprocess.run", fake_run)
    journal = ExperimentJournal(tmp_path / "journal.json")
    trained = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis="trained task2 candidate beats persistence",
            action_type="task2_train_model",
            params={},
        )
    )
    journal.update_result(
        trained.id,
        success=True,
        metrics={"forecast_mse": 0.01},
        artifacts={
            "best_candidate": {
                "task": "task2",
                "checkpoint_path": str(checkpoint),
                "train_time": 12.0,
                "metrics": {"forecast_mse": 0.01},
            }
        },
    )
    submit = journal.append_plan(
        CandidatePlan(
            intent="submit",
            hypothesis="package trained task2 checkpoint",
            action_type="task2_submit_best",
            params={"run_dir": "runs/task2/submissions/test"},
        )
    )
    executor = ControlledExperimentExecutor(
        project_root=tmp_path,
        code_dir=code_dir,
        journal=journal,
        metric="forecast_mse",
        maximize=False,
    )

    execution = executor.execute(submit)

    assert execution.success is True
    run_dir = tmp_path / "runs" / "task2" / "submissions" / "test"
    assert (run_dir / "pred.zip").is_file()
    assert (run_dir / "task2_pred.hdf5").is_file()
    assert execution.artifacts["zip_path"].endswith("pred.zip")
