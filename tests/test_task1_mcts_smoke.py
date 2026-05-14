import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from agent.pde_executor import ExperimentExecution
from agent.pde_journal import CandidatePlan
from agent.pde_journal import ExperimentJournal
from agent.pde_mcts import MockMCTSRunner
from agent.pde_mcts import PDEMCTSRunner
from agent.pde_results import RunResult
from agent.project_env import resolve_project_python
from agent.pde_reviewer import ExperimentReviewer
import agent.run_task1_mcts_experiment as mcts_cli


def test_mock_mcts_branches_updates_metrics_and_reports_best_minimize(tmp_path):
    journal = ExperimentJournal(tmp_path / "mcts-journal.json")
    runner = MockMCTSRunner(
        journal=journal,
        metric="mse",
        maximize=False,
        max_children=2,
        exploration_constant=0.0,
    )

    report = runner.run(
        actions=[
            {"name": "baseline", "metric": 0.50, "params": {"kind": "seed"}},
            {"name": "physics-residual", "metric": 0.20, "params": {"kind": "regularized"}},
            {"name": "rollout-refiner", "metric": 0.15, "params": {"kind": "refiner"}},
        ]
    )

    nodes = journal.read()
    root = nodes[0]
    baseline, physics, rollout = nodes[1], nodes[2], nodes[3]

    assert root.plan.action_type == "mcts_root"
    assert root.children_ids == [baseline.id, physics.id]
    assert baseline.parent_id == root.id
    assert physics.parent_id == root.id
    assert rollout.parent_id == physics.id
    assert physics.children_ids == [rollout.id]
    assert rollout.metrics["mse"] == 0.15

    assert report["metric"] == {"name": "mse", "maximize": False}
    assert report["best_node"]["id"] == rollout.id
    assert report["best_node"]["metric"] == 0.15
    assert report["trace"][-1]["selected_parent_id"] == physics.id
    assert report["search_stats"][root.id]["visits"] == 3
    assert json.loads(json.dumps(report))["best_node"]["action"] == "rollout-refiner"


def test_mock_mcts_best_node_respects_maximize_direction(tmp_path):
    journal = ExperimentJournal(tmp_path / "maximize-journal.json")
    runner = MockMCTSRunner(journal=journal, metric="score", maximize=True)

    report = runner.run(
        actions=[
            {"name": "small-score", "metric": 0.10},
            {"name": "large-score", "metric": 0.90},
        ]
    )

    assert report["best_node"]["action"] == "large-score"
    assert report["best_node"]["metric"] == 0.90
    assert journal.best(metric="score", maximize=True).id == report["best_node"]["id"]


def test_mock_mcts_creates_dedicated_root_when_journal_is_non_empty(tmp_path):
    journal = ExperimentJournal(tmp_path / "mixed-journal.json")
    unrelated = journal.append_plan(
        CandidatePlan(
            intent="previous",
            hypothesis="pre-existing non-MCTS experiment",
            action_type="baseline_validate",
            params={"study_name": "old-study"},
            expected_effect="already present in journal",
            risk="none",
        )
    )
    journal.update_result(unrelated.id, success=True, metrics={"mse": 0.01}, artifacts={})
    runner = MockMCTSRunner(journal=journal, metric="mse", maximize=False, study_name="task1-mcts-review")

    report = runner.run(actions=[{"name": "isolated-child", "metric": 0.42}])

    nodes = journal.read()
    root = next(node for node in nodes if node.plan.params.get("mcts_root") is True)
    child = next(node for node in nodes if node.plan.params.get("name") == "isolated-child")
    assert unrelated.children_ids == []
    assert root.id != unrelated.id
    assert root.plan.params["study_name"] == "task1-mcts-review"
    assert child.parent_id == root.id
    assert report["trace"][0]["selected_parent_id"] == root.id
    assert report["best_node"]["id"] == child.id
    assert {node["id"] for node in report["nodes"]} == {root.id, child.id}


def test_mock_mcts_reads_metric_from_metrics_json(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"mse": 0.123}), encoding="utf-8")
    journal = ExperimentJournal(tmp_path / "metrics-journal.json")
    runner = MockMCTSRunner(journal=journal, metric="mse", maximize=False)

    report = runner.run(actions=[{"name": "file-metric", "metrics_path": str(metrics_path)}])

    assert report["best_node"]["action"] == "file-metric"
    assert report["best_node"]["metric"] == 0.123
    assert journal.best(metric="mse", maximize=False).metrics["mse"] == 0.123


def test_controlled_mcts_evaluates_action_without_direct_metric_through_executor_and_journal(tmp_path):
    journal = ExperimentJournal(tmp_path / "controlled-journal.json")
    calls = []

    class FakeExecutor:
        def execute(self, node):
            calls.append({"node_id": node.id, "status": node.status, "plan": node.plan})
            return ExperimentExecution(
                success=True,
                metrics={"mse": 0.321},
                artifacts={"source": "fake-controlled-executor"},
            )

    runner = MockMCTSRunner(
        journal=journal,
        metric="mse",
        maximize=False,
        executor=FakeExecutor(),
        reviewer=ExperimentReviewer(journal=journal, metric="mse", maximize=False),
        execution="controlled",
    )

    report = runner.run(
        actions=[
            {
                "name": "executor-scored",
                "action_type": "weight_search",
                "params": {"candidates": [{"name": "current-final", "weights": {"nu0.001": 1.0}}]},
            }
        ]
    )

    child = next(node for node in journal.read() if node.plan.params.get("name") == "executor-scored")
    assert calls == [{"node_id": child.id, "status": "running", "plan": child.plan}]
    assert child.status == "completed"
    assert child.metrics["mse"] == 0.321
    assert child.review["metric_value"] == 0.321
    assert report["best_node"]["id"] == child.id
    assert report["trace"][0]["metric"] == 0.321


def test_controlled_mcts_marks_failed_actions_and_excludes_them_from_best(tmp_path):
    journal = ExperimentJournal(tmp_path / "controlled-failure-journal.json")

    class FakeExecutor:
        def execute(self, node):
            if node.plan.params["name"] == "bad-fast":
                return ExperimentExecution(
                    success=False,
                    metrics={"mse": 0.001},
                    artifacts={"source": "fake-controlled-executor"},
                    error="synthetic validation failure",
                )
            return ExperimentExecution(
                success=True,
                metrics={"mse": 0.5},
                artifacts={"source": "fake-controlled-executor"},
            )

    runner = MockMCTSRunner(
        journal=journal,
        metric="mse",
        maximize=False,
        executor=FakeExecutor(),
        reviewer=ExperimentReviewer(journal=journal, metric="mse", maximize=False),
        execution="controlled",
    )

    report = runner.run(
        actions=[
            {"name": "bad-fast", "action_type": "weight_search", "params": {"candidates": [{"weights": {"a": 1.0}}]}},
            {"name": "good-slower", "action_type": "weight_search", "params": {"candidates": [{"weights": {"a": 1.0}}]}},
        ]
    )

    nodes = {node.plan.params.get("name"): node for node in journal.read()}
    assert nodes["bad-fast"].status == "failed"
    assert nodes["bad-fast"].error == "synthetic validation failure"
    assert nodes["good-slower"].status == "completed"
    assert report["best_node"]["action"] == "good-slower"
    assert report["best_node"]["metric"] == 0.5


def test_mcts_selection_does_not_keep_expanding_failed_or_no_metric_children(tmp_path):
    journal = ExperimentJournal(tmp_path / "non-scoring-selection-journal.json")

    class FakeExecutor:
        def execute(self, node):
            name = node.plan.params["name"]
            if name == "failed-branch":
                return ExperimentExecution(success=False, error="synthetic failure")
            if name == "no-metric-branch":
                return ExperimentExecution(success=True, metrics={}, artifacts={"zip_path": "pred.zip"})
            return ExperimentExecution(success=True, metrics={"mse": 0.25}, artifacts={"source": "metric"})

    runner = PDEMCTSRunner(
        journal=journal,
        metric="mse",
        maximize=False,
        max_children=2,
        exploration_constant=0.0,
        execution="controlled",
        executor=FakeExecutor(),
        reviewer=ExperimentReviewer(journal=journal, metric="mse", maximize=False),
    )

    report = runner.run(
        actions=[
            {"name": "metric-branch", "action_type": "weight_search", "params": {"candidates": [{"weights": {"a": 1.0}}]}},
            {"name": "failed-branch", "action_type": "weight_search", "params": {"candidates": [{"weights": {"a": 1.0}}]}},
            {"name": "metric-child", "action_type": "weight_search", "params": {"candidates": [{"weights": {"a": 1.0}}]}},
            {"name": "no-metric-branch", "action_type": "submit_best"},
        ]
    )

    nodes_by_name = {node.plan.params.get("name"): node for node in journal.read()}
    assert nodes_by_name["metric-child"].parent_id == nodes_by_name["metric-branch"].id
    assert report["trace"][2]["selected_parent_id"] == nodes_by_name["metric-branch"].id
    assert report["search_stats"][nodes_by_name["failed-branch"].id]["visits"] == 1
    assert report["search_stats"][nodes_by_name["no-metric-branch"].id]["visits"] == 1


def test_mcts_resume_rebuilds_stats_for_existing_failed_and_no_metric_nodes(tmp_path):
    journal_path = tmp_path / "resume-non-scoring-journal.json"
    first_journal = ExperimentJournal(journal_path)

    class FirstExecutor:
        def execute(self, node):
            name = node.plan.params["name"]
            if name == "failed-branch":
                return ExperimentExecution(success=False, error="synthetic failure")
            if name == "no-metric-branch":
                return ExperimentExecution(success=True, metrics={}, artifacts={"zip_path": "pred.zip"})
            return ExperimentExecution(success=True, metrics={"mse": 0.25}, artifacts={"source": "metric"})

    first_runner = PDEMCTSRunner(
        journal=first_journal,
        metric="mse",
        maximize=False,
        max_children=2,
        exploration_constant=0.0,
        execution="controlled",
        executor=FirstExecutor(),
        reviewer=ExperimentReviewer(journal=first_journal, metric="mse", maximize=False),
    )
    first_runner.run(
        actions=[
            {"name": "metric-branch", "action_type": "weight_search", "params": {"candidates": [{"weights": {"a": 1.0}}]}},
            {"name": "failed-branch", "action_type": "weight_search", "params": {"candidates": [{"weights": {"a": 1.0}}]}},
            {"name": "no-metric-branch", "action_type": "submit_best"},
        ]
    )

    resumed_journal = ExperimentJournal(journal_path)

    class ResumeExecutor:
        def execute(self, node):
            return ExperimentExecution(success=True, metrics={"mse": 0.2}, artifacts={"source": "resume"})

    resumed_runner = PDEMCTSRunner(
        journal=resumed_journal,
        metric="mse",
        maximize=False,
        max_children=2,
        exploration_constant=0.0,
        execution="controlled",
        executor=ResumeExecutor(),
        reviewer=ExperimentReviewer(journal=resumed_journal, metric="mse", maximize=False),
    )
    report = resumed_runner.run(
        actions=[
            {"name": "resume-metric-child", "action_type": "weight_search", "params": {"candidates": [{"weights": {"a": 1.0}}]}},
        ]
    )

    nodes_by_name = {node.plan.params.get("name"): node for node in resumed_journal.read()}
    assert nodes_by_name["resume-metric-child"].parent_id == nodes_by_name["metric-branch"].id
    assert report["search_stats"][nodes_by_name["failed-branch"].id]["visits"] == 1
    assert report["search_stats"][nodes_by_name["no-metric-branch"].id]["visits"] == 1


@pytest.mark.parametrize(
    ("action", "message"),
    [
        ({"name": "missing"}, "missing metric"),
        ({"name": "non-numeric", "metric": "not-a-number"}, "non-numeric metric"),
        ({"name": "nan", "metric": math.nan}, "finite"),
        ({"name": "inf", "metric": math.inf}, "finite"),
    ],
)
def test_mock_mcts_rejects_missing_non_numeric_or_non_finite_metrics(tmp_path, action, message):
    journal = ExperimentJournal(tmp_path / "invalid-metrics-journal.json")
    runner = MockMCTSRunner(journal=journal, metric="mse", maximize=False)

    with pytest.raises(ValueError, match=message):
        runner.run(actions=[action])


def test_task1_mcts_cli_accepts_config_and_max_steps(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "task1_mcts_mock.yaml"
    config_path.write_text((repo_root / "configs" / "task1_mcts_mock.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent.run_task1_mcts_experiment",
            "--config",
            "configs/task1_mcts_mock.yaml",
            "--max-steps",
            "3",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = tmp_path / "runs" / "task1-mcts-mock" / "mcts_summary.json"
    journal_path = tmp_path / "runs" / "task1-mcts-mock" / "journal.json"
    assert summary_path.exists()
    assert journal_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["study_name"] == "task1-mcts-mock"
    assert len(summary["trace"]) == 3
    assert summary["best_node"]["action"] == "residual-smoothing"


def test_task1_mcts_cli_reset_clears_existing_study_dir(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "task1_mcts_mock.yaml"
    config_path.write_text((repo_root / "configs" / "task1_mcts_mock.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    stale_path = tmp_path / "runs" / "task1-mcts-mock" / "stale.txt"
    stale_path.parent.mkdir(parents=True)
    stale_path.write_text("old run artifact", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent.run_task1_mcts_experiment",
            "--config",
            "configs/task1_mcts_mock.yaml",
            "--max-steps",
            "1",
            "--reset",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not stale_path.exists()
    summary = json.loads((tmp_path / "runs" / "task1-mcts-mock" / "mcts_summary.json").read_text(encoding="utf-8"))
    assert len(summary["trace"]) == 1


def test_task1_mcts_config_execution_controlled_uses_injected_project_components(tmp_path, monkeypatch):
    config_path = tmp_path / "controlled.yaml"
    config_path.write_text(
        """
study_name: controlled-from-config
execution: controlled
metric:
  name: mse
  maximize: false
actions:
  - name: config-controlled-action
    action_type: weight_search
    params:
      candidates:
        - name: fake
          weights:
            nu0.001: 0.0
            unet_pf20_nu0.001: 1.0
""".lstrip(),
        encoding="utf-8",
    )
    constructed = {}

    class FakeWorkflow:
        def __init__(self, **kwargs):
            constructed["workflow"] = kwargs

    class FakeExecutor:
        def __init__(self, **kwargs):
            constructed["executor"] = kwargs

        def execute(self, node):
            return ExperimentExecution(success=True, metrics={"mse": 0.111}, artifacts={"fake": True})

    monkeypatch.setattr(mcts_cli, "Task1FNOWorkflow", FakeWorkflow)
    monkeypatch.setattr(mcts_cli, "ControlledExperimentExecutor", FakeExecutor)

    summary_path = mcts_cli.run_task1_mcts_experiment(config_path=config_path, project_root=tmp_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["execution"] == "controlled"
    assert summary["best_node"]["action"] == "config-controlled-action"
    assert summary["best_node"]["metric"] == 0.111
    assert constructed["workflow"]["project_root"] == tmp_path.resolve()
    assert constructed["workflow"]["code_dir"] == tmp_path.resolve() / "code"
    assert constructed["workflow"]["methodology_path"] == tmp_path.resolve() / "docs" / "methodology.pdf"
    assert constructed["executor"]["metric"] == "mse"


def test_pde_mcts_runner_name_is_primary_alias(tmp_path):
    journal = ExperimentJournal(tmp_path / "alias-journal.json")

    runner = PDEMCTSRunner(journal=journal, metric="mse", maximize=False)
    report = runner.run(actions=[{"name": "alias-action", "metric": 0.25}])

    assert report["best_node"]["action"] == "alias-action"
    assert report["nodes"][0]["hypothesis"] == "PDE MCTS search root"


def test_controlled_mcts_weight_search_then_submit_best_exposes_submission_artifact(tmp_path, monkeypatch):
    config_path = tmp_path / "controlled-submit.yaml"
    config_path.write_text(
        """
study_name: controlled-submit
execution: controlled
metric:
  name: mse
  maximize: false
actions:
  - name: choose-best-weights
    intent: validate
    hypothesis: Validate candidate weights before packaging.
    action_type: weight_search
    params:
      metric: mse
      maximize: false
      make_submission: false
      candidates:
        - name: best
          weights:
            nu0.001: 0.75
            unet_pf20_nu0.001: 0.25
  - name: package-best
    intent: submit
    hypothesis: Package the best validated candidate.
    action_type: submit_best
    params:
      train_time: 3.5
""".lstrip(),
        encoding="utf-8",
    )
    captured = {}

    class FakeWorkflow:
        def __init__(self, **kwargs):
            self.run_root = kwargs["run_root"]

        def run_validation(self, weights, *, run_name=None):
            return RunResult(
                task_id="task1",
                run_dir=self.run_root / str(run_name),
                metrics={"mse": 0.123},
                prediction_path=self.run_root / str(run_name) / "task1_val_pred.hdf5",
                zip_path=None,
                train_time=0.0,
                inference_time=1.0,
                success=True,
                error=None,
                weights=dict(weights),
                command=["fake-validation"],
            )

        def run_test_submission(self, weights, *, run_name=None, train_time=0.0):
            captured["weights"] = dict(weights)
            captured["train_time"] = float(train_time)
            run_dir = self.run_root / str(run_name)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "pred.zip").write_bytes(b"fake zip")
            return RunResult(
                task_id="task1",
                run_dir=run_dir,
                metrics={},
                prediction_path=run_dir / "task1_pred.hdf5",
                zip_path=run_dir / "pred.zip",
                train_time=float(train_time),
                inference_time=2.0,
                success=True,
                error=None,
                weights=dict(weights),
                command=["fake-submit"],
            )

    monkeypatch.setattr(mcts_cli, "Task1FNOWorkflow", FakeWorkflow)

    summary_path = mcts_cli.run_task1_mcts_experiment(config_path=config_path, project_root=tmp_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    submit_node = next(node for node in summary["nodes"] if node["action"] == "package-best")
    assert summary["best_node"]["action"] == "choose-best-weights"
    assert summary["submission_node_id"] == submit_node["id"]
    assert summary["submission_zip_path"].endswith("pred.zip")
    assert Path(summary["journal_report_path"]).is_file()
    assert "pred.zip" in Path(summary["journal_report_path"]).read_text(encoding="utf-8")
    assert captured["weights"] == {"nu0.001": 0.75, "unet_pf20_nu0.001": 0.25}
    assert captured["train_time"] == 3.5
    assert submit_node["artifacts"]["zip_path"].endswith("pred.zip")


def test_resolve_project_python_prefers_hwpytorch_default_when_present(tmp_path):
    python_path = tmp_path / "Hwpytorch" / "python.exe"
    python_path.parent.mkdir()
    python_path.write_text("", encoding="utf-8")

    resolved = resolve_project_python(default_python=python_path, env={})

    assert resolved == python_path.resolve()


def test_resolve_project_python_allows_environment_override(tmp_path):
    default_python = tmp_path / "Hwpytorch" / "python.exe"
    override_python = tmp_path / "custom" / "python.exe"
    default_python.parent.mkdir()
    override_python.parent.mkdir()
    default_python.write_text("", encoding="utf-8")
    override_python.write_text("", encoding="utf-8")

    resolved = resolve_project_python(default_python=default_python, env={"AI4S_PROJECT_PYTHON": str(override_python)})

    assert resolved == override_python.resolve()


def test_find_submission_node_requires_completed_successful_existing_zip(tmp_path):
    missing_zip = tmp_path / "missing.zip"
    failed_zip = tmp_path / "failed.zip"
    good_zip = tmp_path / "good.zip"
    failed_zip.write_bytes(b"failed")
    good_zip.write_bytes(b"good")
    summary = {
        "nodes": [
            {
                "id": "failed",
                "status": "failed",
                "error": "bad",
                "action_type": "submit_best",
                "artifacts": {"success": True, "zip_path": str(failed_zip)},
            },
            {
                "id": "missing",
                "status": "completed",
                "error": None,
                "action_type": "submit_best",
                "artifacts": {"success": True, "zip_path": str(missing_zip)},
            },
            {
                "id": "good",
                "status": "completed",
                "error": None,
                "action_type": "submit_best",
                "artifacts": {"success": True, "zip_path": str(good_zip)},
            },
            {
                "id": "later-failed",
                "status": "failed",
                "error": "bad",
                "action_type": "submit_best",
                "artifacts": {"success": True, "zip_path": str(failed_zip)},
            },
        ]
    }

    assert mcts_cli._find_submission_node(summary)["id"] == "good"


def test_full_mcts_config_runs_validation_then_submission():
    repo_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repo_root / "configs" / "task1_mcts_full.yaml").read_text(encoding="utf-8"))

    assert config["execution"] == "controlled"
    assert [action["action_type"] for action in config["actions"]] == ["weight_search", "submit_best"]
    assert config["actions"][0]["params"]["make_submission"] is False
    assert config["actions"][1]["params"]["train_time"] == 0.0


def test_project_python_version_range_supports_hwpytorch_python310():
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.10,<3.13"' in pyproject


def test_hwpytorch_mcts_script_uses_project_python_default():
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "scripts" / "run-task1-mcts.ps1").read_text(encoding="utf-8")

    assert "AI4S_PROJECT_PYTHON" in script
    assert "Hwpytorch" in script
    assert "-m agent.run_task1_mcts_experiment" in script
    assert "$PSScriptRoot" in script
    assert "Push-Location" in script
