from pathlib import Path

import pytest

from agent.tools import ToolError, ToolRunner


def test_read_file_refuses_paths_outside_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    runner = ToolRunner(project_root=project)

    with pytest.raises(ToolError, match="outside allowed roots"):
        runner.read_file(outside)


def test_shell_refuses_commands_outside_allowlist(tmp_path):
    runner = ToolRunner(project_root=tmp_path, allowed_shell_commands=["python", "pytest"])

    with pytest.raises(ToolError, match="not allowed"):
        runner.run_shell(["git", "status"])


def test_write_file_records_manifest_entry(tmp_path):
    runner = ToolRunner(project_root=tmp_path)
    target = tmp_path / "code" / "model.py"

    runner.write_file(target, "print('ok')\n", reason="agent generated baseline")

    assert target.read_text(encoding="utf-8") == "print('ok')\n"
    assert runner.manifest["writes"][0]["path"] == str(target)
    assert runner.manifest["writes"][0]["reason"] == "agent generated baseline"


def test_shell_uses_command_alias_for_python(tmp_path):
    runner = ToolRunner(
        project_root=tmp_path,
        allowed_shell_commands=["python"],
        command_aliases={"python": "python"},
    )

    result = runner.run_shell(["python", "-c", "print('alias ok')"])

    assert result["returncode"] == 0
    assert result["args"] == ["python", "-c", "print('alias ok')"]
    assert result["resolved_args"][0] == "python"
    assert result["stdout"].strip() == "alias ok"


def test_shell_records_elapsed_seconds(tmp_path):
    runner = ToolRunner(project_root=tmp_path, allowed_shell_commands=["python"])

    result = runner.run_shell(["python", "-c", "print('timed')"])

    assert result["returncode"] == 0
    assert isinstance(result["elapsed_seconds"], float)
    assert result["elapsed_seconds"] >= 0.0
    assert runner.manifest["shell"][0]["elapsed_seconds"] == result["elapsed_seconds"]
