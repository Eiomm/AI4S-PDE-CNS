import json
import os

import h5py
import numpy as np

from agent.run import execute_action
from agent.run import load_project_env
from agent.run import run_agent
from agent.tools import ToolRunner


def test_mock_agent_runs_one_observe_plan_act_record_cycle(tmp_path):
    config = tmp_path / "task1.yaml"
    config.write_text(
        "\n".join(
            [
                "provider: mock",
                "model: mock-planner",
                "max_iterations: 1",
                "time_budget_seconds: 60",
                "allowed_shell_commands:",
                "  - python",
            ]
        ),
        encoding="utf-8",
    )

    run_dir = run_agent(task="task1", config_path=config, project_root=tmp_path)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["task"] == "task1"
    assert manifest["iterations"][0]["action"]["tool"] == "record_note"
    log_record = json.loads((run_dir / "task1_logs.log").read_text(encoding="utf-8").strip())
    assert log_record["provider"] == "mock"


def test_load_project_env_uses_project_root_relative_env_file(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("DEEPSEEK_API_KEY=root-secret\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    loaded = load_project_env({"env_file": ".env"}, tmp_path)

    assert loaded == ["DEEPSEEK_API_KEY"]
    assert os.getenv("DEEPSEEK_API_KEY") == "root-secret"


def test_execute_action_preserves_quoted_python_command(tmp_path):
    runner = ToolRunner(project_root=tmp_path, allowed_shell_commands=["python"])
    action = {
        "tool": "run_shell",
        "args": {
            "command": "python -c \"print('hello quoted world')\"",
            "timeout": 30,
        },
    }

    result = execute_action(action, runner, tmp_path / "run", tmp_path)

    assert result["ok"] is True
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "hello quoted world"


def test_execute_action_accepts_structured_shell_args(tmp_path):
    runner = ToolRunner(project_root=tmp_path, allowed_shell_commands=["python"])
    action = {
        "tool": "run_shell",
        "args": {
            "args": ["python", "-c", "print('structured ok')"],
            "timeout": 30,
        },
    }

    result = execute_action(action, runner, tmp_path / "run", tmp_path)

    assert result["ok"] is True
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "structured ok"


def test_execute_action_creates_task1_submission_bundle(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "task1-test"
    run_dir.mkdir(parents=True)
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "main.py").write_text("print('submission code')\n", encoding="utf-8")

    initial = np.zeros((2, 10, 256), dtype=np.float32)
    prediction = np.zeros((2, 200, 256), dtype=np.float32)
    initial_path = tmp_path / "data" / "task1_test.hdf5"
    initial_path.parent.mkdir()
    prediction_path = run_dir / "task1_pred.hdf5"
    with h5py.File(initial_path, "w") as h5:
        h5.create_dataset("tensor", data=initial)
    with h5py.File(prediction_path, "w") as h5:
        h5.create_dataset("prediction", data=prediction)
    (run_dir / "task1_logs.log").write_text(
        '{"timestamp":"2026-05-12T00:00:00","elapsed_seconds":0,"response":{"content":"ok"}}\n',
        encoding="utf-8",
    )
    fake_log_path = run_dir / "fake_logs.log"
    fake_log_path.write_text(
        '{"timestamp":"2026-05-12T00:00:01","elapsed_seconds":0,"response":{"content":"fake"}}\n',
        encoding="utf-8",
    )

    runner = ToolRunner(project_root=tmp_path, allowed_shell_commands=[])
    output_dir = tmp_path / "runs" / "submission"
    action = {
        "tool": "create_task1_submission",
        "args": {
            "prediction_path": str(prediction_path),
            "initial_path": str(initial_path),
            "output_dir": str(output_dir),
            "code_dir": str(code_dir),
            "log_path": str(fake_log_path),
            "train_time": "elapsed_without_inference",
            "inference_time": 0.4,
        },
    }
    monkeypatch.setattr("agent.run.time.perf_counter", lambda: 112.3)

    result = execute_action(action, runner, run_dir, tmp_path, agent_started=100.0)

    assert result == {"ok": True, "path": str(output_dir)}
    assert (output_dir / "task1_pred.hdf5").exists()
    assert '"content":"ok"' in (output_dir / "task1_logs.log").read_text(encoding="utf-8")
    assert (output_dir / "task1_time.csv").read_text(encoding="utf-8").splitlines()[1] == "11.900000,0.400000"
