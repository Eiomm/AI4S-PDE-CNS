from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.logging import LLMCallLogger
from agent.pde_journal import CandidatePlan, ExperimentJournal


class _RecordingStopClient:
    provider = "recording"
    model = "recording"

    def __init__(self):
        self.messages = []

    def complete(self, messages):
        self.messages = messages
        return {
            "content": json.dumps(
                {
                    "intent": "stop",
                    "hypothesis": "inspect method candidates",
                    "action_type": "stop",
                    "params": {"reason": "done"},
                    "expected_effect": "none",
                    "risk": "none",
                }
            )
        }


def test_method_library_recommends_long_horizon_stability_methods():
    from agent.pde_method_library import select_method_candidates

    candidates = select_method_candidates(
        task="task1",
        metrics={
            "forecast_mse": 0.00062,
            "long_horizon_mse": 0.00075,
            "segment1_rel_mse": 0.0078,
            "segment2_rel_mse": 0.0306,
            "segment3_rmse": 0.0274,
        },
    )

    names = [candidate["name"] for candidate in candidates]
    assert "DPOT-style autoregressive rollout stability" in names
    assert "Flow-Marching-inspired long-horizon stabilization" in names
    assert candidates[0]["implementation_knobs"]["horizon_loss_gamma"] == [1.02, 1.05, 1.1]
    assert all("source" in candidate and "reason" in candidate for candidate in candidates)


def test_observer_exposes_method_candidates(tmp_path):
    from agent.pde_observer import observe_research_context

    context = observe_research_context(tmp_path)

    assert "method_candidates" in context
    assert any(candidate["name"].startswith("PINO") for candidate in context["method_candidates"])


def test_task_research_log_export_contains_trace_and_failed_experiments(tmp_path):
    from agent.task_log_export import export_task_research_log

    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    study_dir = tmp_path / "runs" / "study"
    study_dir.mkdir(parents=True)
    journal = ExperimentJournal(study_dir / "journal.json")
    baseline = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis="stride5 fine-tune establishes the current baseline",
            action_type="finetune_checkpoint",
            params={"gradient_loss_weight": 0.0},
            expected_effect="improve rollout score",
            risk="overfit",
        )
    )
    journal.update_result(
        baseline.id,
        success=True,
        metrics={"competition_score_proxy": 79.0168964, "long_horizon_mse": 0.0007546},
        artifacts={"best_checkpoint": "runs/best.pt"},
    )
    failed = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis="strong spectral regularization may reduce long-horizon drift",
            action_type="finetune_checkpoint",
            params={"gradient_loss_weight": 0.1, "spectral_loss_weight": 0.05},
            expected_effect="reduce long_horizon_mse",
            risk="over-smoothing",
        ),
        parent_id=baseline.id,
    )
    journal.update_result(
        failed.id,
        success=True,
        metrics={"competition_score_proxy": 76.5, "long_horizon_mse": 0.0010},
        artifacts={"best_checkpoint": "runs/failed.pt"},
        review={"analysis": "No improvement; regularization was too strong."},
    )
    logger = LLMCallLogger(study_dir / "planner_logs.log")
    logger.write_call(
        provider="hkustgz_gpt",
        model="gpt-5.3-chat",
        messages=[{"role": "user", "content": "diagnose long-horizon drift"}],
        response={"content": "try mild rollout regularization"},
        elapsed_seconds=1.2,
    )

    output_path = export_task_research_log(
        study_dir=study_dir,
        output_path=study_dir / "task1_logs.log",
        task="task1",
        code_dir=code_dir,
    )
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    sections = [record.get("response", {}).get("section") for record in records]
    assert "problem_understanding" in sections
    assert "literature_method_trace" in sections
    assert "bottleneck_diagnosis" in sections
    assert "code_evolution" in sections
    assert "experiment_tracking" in sections
    assert all("timestamp" in record and "elapsed_seconds" in record for record in records)
    assert any("No improvement" in record.get("response", {}).get("content", "") for record in records)
    assert any(record.get("response", {}).get("action") == "write_code_file" for record in records)


def test_task_research_log_export_requires_existing_study(tmp_path):
    from agent.task_log_export import export_task_research_log

    with pytest.raises(FileNotFoundError):
        export_task_research_log(study_dir=tmp_path / "missing", output_path=tmp_path / "task1_logs.log")


def test_planner_receives_ranked_method_candidates_from_best_metrics(tmp_path):
    from agent.pde_planner import ExperimentPlanner

    client = _RecordingStopClient()
    journal = ExperimentJournal(tmp_path / "journal.json")
    node = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis="baseline has rollout drift",
            action_type="finetune_checkpoint",
            params={},
        )
    )
    journal.update_result(
        node.id,
        success=True,
        metrics={
            "competition_score_proxy": 79.0,
            "forecast_mse": 0.00062,
            "long_horizon_mse": 0.00075,
            "segment1_rel_mse": 0.0078,
            "segment2_rel_mse": 0.0306,
            "segment3_rmse": 0.0274,
        },
    )
    planner = ExperimentPlanner(
        client=client,
        logger=LLMCallLogger(tmp_path / "planner.log"),
        journal=journal,
        metric="competition_score_proxy",
        maximize=True,
    )

    planner.plan_next({"task": "task1"})

    prompt = "\n".join(message["content"] for message in client.messages)
    assert "method_candidates" in prompt
    assert "DPOT-style autoregressive rollout stability" in prompt
    assert "horizon_loss_gamma" in prompt
