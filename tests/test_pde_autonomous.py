import json
import csv

import h5py
import numpy as np
import pytest

from agent.code_trace import append_code_trace_log
from agent.logging import LLMCallLogger
from agent.pde_autonomous import AutonomousExperimentRunner
from agent.pde_executor import ControlledExperimentExecutor
from agent.pde_journal import CandidatePlan, ExperimentJournal
from agent.pde_planner import ExperimentPlanner
from agent.pde_registry import export_experiment_records, rank_experiment_records
from agent.pde_report import render_journal_report
from agent.pde_reviewer import ExperimentReviewer
from agent.pde_results import RunResult
from agent.pde_search import WeightedEnsembleSearch
from agent.run_task1_autonomous_experiment import (
    BootstrapPlanClient,
    run_autonomous_task1,
    task1_bootstrap_weight_search_plan,
    task1_local_weight_grid_candidates,
)


class StaticExperimentClient:
    provider = "static"
    model = "planner"

    def __init__(self, payload):
        self.payload = payload

    def complete(self, messages):
        return {"content": json.dumps(self.payload, ensure_ascii=False)}


class RecordingExperimentClient(StaticExperimentClient):
    def __init__(self, payload):
        super().__init__(payload)
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return super().complete(messages)


class SequenceExperimentClient:
    provider = "sequence"
    model = "planner"

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.index = 0

    def complete(self, messages):
        payload = self.payloads[self.index]
        self.index += 1
        return {"content": json.dumps(payload, ensure_ascii=False)}


class FakeValidationWorkflow:
    def __init__(self, run_root):
        self.run_root = run_root

    def run_validation(self, weights, *, run_name=None):
        run_dir = self.run_root / str(run_name or "validation")
        run_dir.mkdir(parents=True, exist_ok=True)
        mse = float(weights.get("penalty", weights.get("unet_pf20_nu0.001", 0.1)))
        return RunResult(
            task_id="task1",
            run_dir=run_dir,
            metrics={"mse": mse},
            prediction_path=run_dir / "task1_val_pred.hdf5",
            zip_path=None,
            train_time=0.0,
            inference_time=0.1,
            success=True,
            error=None,
            weights=dict(weights),
            command=["fake-validation"],
        )


def _write_minimal_submission(root):
    code_dir = root / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "train.py").write_text("print('train')\n", encoding="utf-8")
    (root / "methodology.pdf").write_bytes(b"%PDF-1.4\n% placeholder\n")
    (root / "submission.json").write_text(
        json.dumps({"submission_id": "team", "problem_id": "PDE_Burgers", "code_path": "code"}),
        encoding="utf-8",
    )
    with h5py.File(root / "task1_pred.hdf5", "w") as h5:
        h5.create_dataset("tensor", data=np.zeros((2, 200, 256), dtype=np.float32))
    (root / "task1_time.csv").write_text("train_time,inference_time\n1,1\n", encoding="utf-8")
    (root / "task1_logs.log").write_text(
        json.dumps({"timestamp": "2026-05-13T00:00:00+08:00", "elapsed_seconds": 0}) + "\n",
        encoding="utf-8",
    )
    append_code_trace_log(root / "task1_logs.log", code_dir)


def test_experiment_journal_links_nodes_and_selects_best(tmp_path):
    journal = ExperimentJournal(tmp_path / "journal.json")
    root = journal.append_plan(
        CandidatePlan(
            intent="draft",
            hypothesis="baseline ensemble",
            action_type="weight_search",
            params={"candidates": [{"name": "base", "weights": {"nu0.001": 1.0}}]},
            expected_effect="establish validation score",
            risk="low",
        )
    )
    improved = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis="rewrite rollout code to reduce drift",
            action_type="code_patch",
            params={"files": [{"path": "fno_ensemble.py", "content": "print('patched')\n"}]},
            expected_effect="lower mse",
            risk="medium",
        ),
        parent_id=root.id,
    )

    journal.update_result(root.id, success=True, metrics={"mse": 0.2}, artifacts={"run_dir": "runs/base"})
    journal.update_result(improved.id, success=True, metrics={"mse": 0.1}, artifacts={"run_dir": "runs/improved"})
    reloaded = ExperimentJournal(tmp_path / "journal.json")

    assert reloaded.get(root.id).children_ids == [improved.id]
    assert reloaded.best(metric="mse", maximize=False).id == improved.id
    assert reloaded.best(metric="mse", maximize=True).id == root.id


def test_experiment_planner_accepts_llm_code_patch_plan(tmp_path):
    payload = {
        "intent": "improve",
        "hypothesis": "make inference code use a lower-drift postprocess",
        "action_type": "code_patch",
        "params": {
            "files": [
                {
                    "path": "fno_ensemble.py",
                    "content": "def postprocess(x):\n    return x\n",
                }
            ]
        },
        "expected_effect": "improve long horizon mse",
        "risk": "touches submitted code, so run validation immediately",
    }
    journal = ExperimentJournal(tmp_path / "journal.json")
    planner = ExperimentPlanner(
        client=StaticExperimentClient(payload),
        logger=LLMCallLogger(tmp_path / "planner.log"),
        journal=journal,
    )

    node = planner.plan_next({"task": "task1", "best_metric": 0.2})

    assert node.plan.action_type == "code_patch"
    assert node.plan.params["files"][0]["path"] == "fno_ensemble.py"
    assert journal.get(node.id).plan.hypothesis.startswith("make inference")
    assert json.loads((tmp_path / "planner.log").read_text(encoding="utf-8").splitlines()[0])["provider"] == "static"


def test_experiment_planner_prompt_contains_champion_strategy_priorities(tmp_path):
    payload = {
        "intent": "stop",
        "hypothesis": "inspect strategy prompt",
        "action_type": "stop",
        "params": {"reason": "prompt inspected"},
        "expected_effect": "none",
        "risk": "none",
    }
    client = RecordingExperimentClient(payload)
    planner = ExperimentPlanner(
        client=client,
        logger=LLMCallLogger(tmp_path / "planner.log"),
        journal=ExperimentJournal(tmp_path / "journal.json"),
    )

    planner.plan_next({"task": "task1"})

    prompt = "\n".join(message["content"] for message in client.messages)
    assert "multi-step rollout" in prompt
    assert "spectral loss" in prompt
    assert "physics residual" in prompt
    assert "PDE-Refiner" in prompt


def test_code_patch_executor_writes_only_under_code_dir(tmp_path):
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    journal = ExperimentJournal(tmp_path / "journal.json")
    node = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis="rewrite submitted helper",
            action_type="code_patch",
            params={"files": [{"path": "helpers/new_helper.py", "content": "VALUE = 3\n"}]},
            expected_effect="enable next validation run",
            risk="medium",
        )
    )
    executor = ControlledExperimentExecutor(project_root=tmp_path, code_dir=code_dir)

    execution = executor.execute(node)

    assert execution.success is True
    assert (code_dir / "helpers" / "new_helper.py").read_text(encoding="utf-8") == "VALUE = 3\n"
    assert execution.artifacts["patched_files"] == ["code/helpers/new_helper.py"]


def test_code_patch_executor_runs_validation_command_after_patch(tmp_path):
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    journal = ExperimentJournal(tmp_path / "journal.json")
    node = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis="patch then run smoke validation",
            action_type="code_patch",
            params={
                "files": [{"path": "helper.py", "content": "VALUE = 7\n"}],
                "validation_command": ["python", "-c", "print('validation ok')"],
            },
            expected_effect="validated patch",
            risk="validation can fail",
        )
    )
    executor = ControlledExperimentExecutor(project_root=tmp_path, code_dir=code_dir)

    execution = executor.execute(node)

    assert execution.success is True
    assert execution.artifacts["validation"]["returncode"] == 0
    assert "validation ok" in execution.artifacts["validation"]["stdout_tail"]


def test_code_patch_executor_fails_when_validation_command_fails(tmp_path):
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    journal = ExperimentJournal(tmp_path / "journal.json")
    node = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis="patch then fail validation",
            action_type="code_patch",
            params={
                "files": [{"path": "helper.py", "content": "VALUE = 9\n"}],
                "validation_command": ["python", "-c", "raise SystemExit(4)"],
            },
            expected_effect="validated patch",
            risk="validation can fail",
        )
    )
    executor = ControlledExperimentExecutor(project_root=tmp_path, code_dir=code_dir)

    execution = executor.execute(node)

    assert execution.success is False
    assert "validation command failed" in execution.error
    assert execution.artifacts["validation"]["returncode"] == 4


def test_code_patch_executor_runs_submission_validation_gate(tmp_path):
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    submission_dir = tmp_path / "runs" / "candidate-submission"
    _write_minimal_submission(submission_dir)
    journal = ExperimentJournal(tmp_path / "journal.json")
    node = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis="patch then verify submission consistency",
            action_type="code_patch",
            params={
                "files": [{"path": "helper.py", "content": "VALUE = 13\n"}],
                "submission_validation_path": str(submission_dir),
            },
            expected_effect="submission gate passes",
            risk="submission validation can fail",
        )
    )
    executor = ControlledExperimentExecutor(project_root=tmp_path, code_dir=code_dir)

    execution = executor.execute(node)

    assert execution.success is True
    assert execution.artifacts["submission_validation"]["valid"] is True
    assert execution.artifacts["submission_validation"]["tasks"] == ["task1"]


def test_code_patch_executor_fails_when_submission_validation_gate_fails(tmp_path):
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    submission_dir = tmp_path / "runs" / "broken-submission"
    submission_dir.mkdir(parents=True)
    journal = ExperimentJournal(tmp_path / "journal.json")
    node = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis="patch then catch broken submission",
            action_type="code_patch",
            params={
                "files": [{"path": "helper.py", "content": "VALUE = 17\n"}],
                "submission_validation_path": str(submission_dir),
            },
            expected_effect="submission gate fails",
            risk="submission missing files",
        )
    )
    executor = ControlledExperimentExecutor(project_root=tmp_path, code_dir=code_dir)

    execution = executor.execute(node)

    assert execution.success is False
    assert "submission validation failed" in execution.error
    assert "submission.json" in execution.artifacts["submission_validation"]["error"]


def test_code_patch_executor_can_require_a_validation_gate(tmp_path):
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    journal = ExperimentJournal(tmp_path / "journal.json")
    node = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis="patch without validation should be rejected in strict mode",
            action_type="code_patch",
            params={"files": [{"path": "helper.py", "content": "VALUE = 19\n"}]},
            expected_effect="strict mode catches missing validation",
            risk="missing validation",
        )
    )
    executor = ControlledExperimentExecutor(
        project_root=tmp_path,
        code_dir=code_dir,
        require_code_patch_validation=True,
    )

    execution = executor.execute(node)

    assert execution.success is False
    assert "code_patch requires validation_command or submission_validation_path" in execution.error


def test_code_patch_executor_rejects_paths_outside_code_dir(tmp_path):
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    journal = ExperimentJournal(tmp_path / "journal.json")
    node = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis="attempt unsafe rewrite",
            action_type="code_patch",
            params={"files": [{"path": "../agent/run.py", "content": "bad\n"}]},
            expected_effect="should be rejected",
            risk="high",
        )
    )
    executor = ControlledExperimentExecutor(project_root=tmp_path, code_dir=code_dir)

    execution = executor.execute(node)

    assert execution.success is False
    assert "outside code_dir" in execution.error
    assert not (tmp_path / "agent" / "run.py").exists()


def test_experiment_reviewer_recommends_debug_for_failed_execution(tmp_path):
    journal = ExperimentJournal(tmp_path / "journal.json")
    node = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis="broken run",
            action_type="finetune",
            params={"command": ["python", "-c", "raise SystemExit(1)"]},
            expected_effect="none",
            risk="command can fail",
        )
    )
    reviewer = ExperimentReviewer(journal=journal, metric="mse", maximize=False)

    reviewed = reviewer.review_execution(
        node,
        success=False,
        metrics={},
        artifacts={"returncode": 1},
        error="command failed",
    )

    assert reviewed.status == "failed"
    assert reviewed.review["next_intent"] == "debug"
    assert reviewed.review["analysis"].startswith("Execution failed")


def test_autonomous_runner_executes_planned_nodes_and_stops(tmp_path):
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    journal = ExperimentJournal(tmp_path / "journal.json")
    client = SequenceExperimentClient(
        [
            {
                "intent": "improve",
                "hypothesis": "patch a submitted helper before validation",
                "action_type": "code_patch",
                "params": {"files": [{"path": "helper.py", "content": "VALUE = 5\n"}]},
                "expected_effect": "prepare code improvement",
                "risk": "needs validation after patch",
            },
            {
                "intent": "stop",
                "hypothesis": "stop after smoke patch",
                "action_type": "stop",
                "params": {"reason": "smoke complete"},
                "expected_effect": "no more work",
                "risk": "none",
            },
        ]
    )
    planner = ExperimentPlanner(
        client=client,
        logger=LLMCallLogger(tmp_path / "planner.log"),
        journal=journal,
    )
    executor = ControlledExperimentExecutor(project_root=tmp_path, code_dir=code_dir, journal=journal)
    reviewer = ExperimentReviewer(journal=journal)
    runner = AutonomousExperimentRunner(planner=planner, executor=executor, reviewer=reviewer)

    summary = runner.run(context={"task": "task1"}, max_iterations=3)

    assert summary["iterations"] == 2
    assert summary["stopped"] is True
    assert (code_dir / "helper.py").read_text(encoding="utf-8") == "VALUE = 5\n"
    nodes = journal.read()
    assert [node.plan.action_type for node in nodes] == ["code_patch", "stop"]
    assert nodes[0].status == "completed"


def test_autonomous_runner_writes_markdown_report(tmp_path):
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    journal = ExperimentJournal(tmp_path / "journal.json")
    client = SequenceExperimentClient(
        [
            {
                "intent": "improve",
                "hypothesis": "patch helper and validate report",
                "action_type": "code_patch",
                "params": {"files": [{"path": "helper.py", "content": "VALUE = 11\n"}]},
                "expected_effect": "report contains the node",
                "risk": "none",
            }
        ]
    )
    planner = ExperimentPlanner(
        client=client,
        logger=LLMCallLogger(tmp_path / "planner.log"),
        journal=journal,
    )
    runner = AutonomousExperimentRunner(
        planner=planner,
        executor=ControlledExperimentExecutor(project_root=tmp_path, code_dir=code_dir, journal=journal),
        reviewer=ExperimentReviewer(journal=journal),
    )

    summary = runner.run(context={"task": "task1"}, max_iterations=1, report_path=tmp_path / "journal_report.md")

    report = (tmp_path / "journal_report.md").read_text(encoding="utf-8")
    assert summary["report_path"].endswith("journal_report.md")
    assert "patch helper and validate report" in report
    assert "| 0 |" in report
    assert "code_patch" in report


def test_render_journal_report_lists_best_node(tmp_path):
    journal = ExperimentJournal(tmp_path / "journal.json")
    node = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis="lower validation mse",
            action_type="weight_search",
            params={},
            expected_effect="lower mse",
            risk="low",
        )
    )
    journal.update_result(node.id, success=True, metrics={"mse": 0.123}, artifacts={"run_dir": "runs/demo"})

    report = render_journal_report(journal, title="Demo Report", metric="mse")

    assert "# Demo Report" in report
    assert "Best node" in report
    assert "0.123" in report


def test_bootstrap_weight_search_plan_contains_real_task1_candidates():
    plan = task1_bootstrap_weight_search_plan(metric="mse", grid_step=0.01, grid_radius=2)

    assert plan["action_type"] == "weight_search"
    assert plan["params"]["metric"] == "mse"
    assert plan["params"]["make_submission"] is False
    assert plan["params"]["grid_step"] == 0.01
    assert plan["params"]["grid_radius"] == 2
    names = [candidate["name"] for candidate in plan["params"]["candidates"]]
    assert "current-final-proxy" in names
    assert len(names) == 5
    assert all("weights" in candidate for candidate in plan["params"]["candidates"])


def test_task1_local_weight_grid_candidates_sweeps_around_current_best():
    candidates = task1_local_weight_grid_candidates(grid_step=0.01, grid_radius=2)

    assert [candidate["name"] for candidate in candidates] == [
        "grid-delta-neg0.020",
        "grid-delta-neg0.010",
        "current-final-proxy",
        "grid-delta-pos0.010",
        "grid-delta-pos0.020",
    ]
    assert [candidate["weights"]["unet_pf20_nu0.001"] for candidate in candidates] == [0.86, 0.87, 0.88, 0.89, 0.9]
    assert [candidate["weights"]["nu0.001"] for candidate in candidates] == [0.14, 0.13, 0.12, 0.11, 0.1]
    for candidate in candidates:
        weights = candidate["weights"]
        assert set(weights) == {"nu0.001", "unet_pf20_nu0.001"}
        assert round(sum(weights.values()), 6) == 1.0
        assert all(value >= 0.0 for value in weights.values())


def test_task1_local_weight_grid_candidates_skips_invalid_edges():
    candidates = task1_local_weight_grid_candidates(
        base_weights={"nu0.001": 0.01, "unet_pf20_nu0.001": 0.99},
        grid_step=0.02,
        grid_radius=2,
    )

    assert [candidate["name"] for candidate in candidates] == [
        "grid-delta-neg0.040",
        "grid-delta-neg0.020",
        "current-final-proxy",
    ]
    assert all(round(sum(candidate["weights"].values()), 6) == 1.0 for candidate in candidates)


def test_task1_local_weight_grid_candidate_names_are_unique_after_path_sanitization():
    candidates = task1_local_weight_grid_candidates(grid_step=0.01, grid_radius=2)

    safe_names = [WeightedEnsembleSearch._safe_name(candidate["name"]) for candidate in candidates]

    assert len(safe_names) == len(set(safe_names))


def test_bootstrap_plan_client_runs_weight_search_through_runner(tmp_path):
    journal = ExperimentJournal(tmp_path / "journal.json")
    bootstrap_plan = {
        "intent": "draft",
        "hypothesis": "run fake lightweight weight search",
        "action_type": "weight_search",
        "params": {
            "metric": "mse",
            "make_submission": False,
            "candidates": [
                {"name": "worse", "weights": {"penalty": 0.2}},
                {"name": "better", "weights": {"penalty": 0.05}},
            ],
        },
        "expected_effect": "select lower mse",
        "risk": "fake workflow only",
    }
    planner = ExperimentPlanner(
        client=BootstrapPlanClient(StaticExperimentClient({"intent": "stop", "action_type": "stop"}), [bootstrap_plan]),
        logger=LLMCallLogger(tmp_path / "planner.log"),
        journal=journal,
    )
    runner = AutonomousExperimentRunner(
        planner=planner,
        executor=ControlledExperimentExecutor(
            project_root=tmp_path,
            code_dir=tmp_path / "code",
            journal=journal,
            workflow=FakeValidationWorkflow(tmp_path / "runs"),
        ),
        reviewer=ExperimentReviewer(journal=journal),
    )

    summary = runner.run(context={"task": "task1"}, max_iterations=1, report_path=tmp_path / "report.md")

    nodes = journal.read()
    assert summary["iterations"] == 1
    assert nodes[0].plan.action_type == "weight_search"
    assert nodes[0].status == "completed"
    assert nodes[0].metrics["mse"] == 0.05
    assert nodes[0].artifacts["best_candidate"]["name"] == "better"
    assert [candidate["name"] for candidate in nodes[0].artifacts["candidate_results"]] == ["worse", "better"]
    assert nodes[0].artifacts["candidate_results"][1]["metrics"] == {"mse": 0.05}
    assert "weight_search" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_export_experiment_records_writes_json_csv_and_global_registry(tmp_path):
    journal = ExperimentJournal(tmp_path / "runs" / "study-a" / "journal.json")
    node = journal.append_plan(
        CandidatePlan(
            intent="draft",
            hypothesis="compare two ensemble candidates",
            action_type="weight_search",
            params={"metric": "mse"},
            expected_effect="lower mse",
            risk="low",
        )
    )
    journal.update_result(
        node.id,
        success=True,
        metrics={"mse": 0.05, "competition_score_proxy": 60.0},
        artifacts={
            "best_candidate": {"name": "better", "weights": {"nu0.001": 0.9, "unet_pf20_nu0.001": 0.1}},
            "candidate_results": [
                {
                    "name": "worse",
                    "weights": {"nu0.001": 0.8, "unet_pf20_nu0.001": 0.2},
                    "metrics": {"mse": 0.1, "competition_score_proxy": 55.0},
                    "run_dir": "runs/study-a/worse",
                    "prediction_path": "runs/study-a/worse/task1_val_pred.hdf5",
                    "success": True,
                },
                {
                    "name": "better",
                    "weights": {"nu0.001": 0.9, "unet_pf20_nu0.001": 0.1},
                    "metrics": {"mse": 0.05, "competition_score_proxy": 60.0},
                    "run_dir": "runs/study-a/better",
                    "prediction_path": "runs/study-a/better/task1_val_pred.hdf5",
                    "success": True,
                },
            ],
            "run_dir": "runs/study-a/nodes/node/better",
            "prediction_path": "runs/study-a/nodes/node/better/task1_val_pred.hdf5",
        },
        review={"next_intent": "improve", "analysis": "better candidate"},
    )

    outputs = export_experiment_records(
        journal,
        study_dir=tmp_path / "runs" / "study-a",
        study_name="study-a",
        runs_root=tmp_path / "runs",
        metric="mse",
        maximize=False,
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    rows = list(csv.DictReader(outputs["csv"].open("r", encoding="utf-8")))
    candidate_rows = list(csv.DictReader(outputs["candidate_csv"].open("r", encoding="utf-8")))
    registry_lines = outputs["registry"].read_text(encoding="utf-8").splitlines()
    assert payload["best"]["node_id"] == node.id
    assert payload["records"][0]["best_candidate_name"] == "better"
    assert payload["records"][0]["weights"] == {"nu0.001": 0.9, "unet_pf20_nu0.001": 0.1}
    assert rows[0]["study_name"] == "study-a"
    assert rows[0]["metric_value"] == "0.05"
    assert [row["candidate_name"] for row in candidate_rows] == ["worse", "better"]
    assert candidate_rows[1]["mse"] == "0.05"
    assert candidate_rows[1]["competition_score_proxy"] == "60.0"
    assert json.loads(candidate_rows[1]["weights"]) == {"nu0.001": 0.9, "unet_pf20_nu0.001": 0.1}
    assert json.loads(registry_lines[0])["node_id"] == node.id

    export_experiment_records(
        journal,
        study_dir=tmp_path / "runs" / "study-a",
        study_name="study-a",
        runs_root=tmp_path / "runs",
        metric="mse",
        maximize=False,
    )
    assert len(outputs["registry"].read_text(encoding="utf-8").splitlines()) == 1


def test_export_experiment_records_tolerates_bom_in_existing_registry(tmp_path):
    registry = tmp_path / "runs" / "experiment_registry.jsonl"
    registry.parent.mkdir()
    registry.write_bytes(b"\xef\xbb\xbf" + json.dumps({"study_name": "old", "node_id": "1"}).encode("utf-8") + b"\n")
    journal = ExperimentJournal(tmp_path / "runs" / "study-b" / "journal.json")
    node = journal.append_plan(
        CandidatePlan(
            intent="draft",
            hypothesis="write after a BOM-prefixed registry",
            action_type="weight_search",
            params={"metric": "mse"},
            expected_effect="registry remains readable",
            risk="low",
        )
    )
    journal.update_result(node.id, success=True, metrics={"mse": 0.04}, artifacts={})

    outputs = export_experiment_records(
        journal,
        study_dir=tmp_path / "runs" / "study-b",
        study_name="study-b",
        runs_root=tmp_path / "runs",
        metric="mse",
        maximize=False,
    )

    lines = outputs["registry"].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["study_name"] == "old"
    assert json.loads(lines[1])["study_name"] == "study-b"


def test_rank_experiment_records_sorts_by_metric_direction():
    records = [
        {"study_name": "a", "node_id": "1", "metric_name": "mse", "metric_value": 0.2},
        {"study_name": "b", "node_id": "2", "metric_name": "mse", "metric_value": 0.1},
        {"study_name": "c", "node_id": "3", "metric_name": "score", "metric_value": 10.0},
    ]

    assert [record["study_name"] for record in rank_experiment_records(records, metric="mse", maximize=False)] == ["b", "a"]
    assert [record["study_name"] for record in rank_experiment_records(records, metric="mse", maximize=True)] == ["a", "b"]


def test_autonomous_task1_cli_function_runs_with_mock_provider(tmp_path):
    (tmp_path / "code").mkdir()
    config = tmp_path / "mock.yaml"
    config.write_text("provider: mock\nmodel: mock-planner\n", encoding="utf-8")

    summary_path = run_autonomous_task1(
        config_path=config,
        project_root=tmp_path,
        study_name="autonomous-smoke",
        max_iterations=1,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["iterations"] == 1
    assert (tmp_path / "runs" / "autonomous-smoke" / "journal_report.md").is_file()
    assert (tmp_path / "runs" / "autonomous-smoke" / "experiment_results.json").is_file()
    assert (tmp_path / "runs" / "autonomous-smoke" / "experiment_comparison.csv").is_file()
    assert (tmp_path / "runs" / "autonomous-smoke" / "candidate_comparison.csv").is_file()
    assert (tmp_path / "runs" / "experiment_registry.jsonl").is_file()
    assert summary["experiment_results_path"].endswith("experiment_results.json")
    assert summary["candidate_comparison_path"].endswith("candidate_comparison.csv")
    assert (tmp_path / "code" / "mock_autonomous_smoke.py").read_text(encoding="utf-8") == "VALUE = 1\n"
