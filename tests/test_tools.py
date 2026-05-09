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
