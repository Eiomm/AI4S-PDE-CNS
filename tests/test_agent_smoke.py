import json

from agent.run import run_agent


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
