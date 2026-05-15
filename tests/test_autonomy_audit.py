import json

import pytest

from agent.autonomy_audit import AutonomyAuditError, audit_autonomous_study
from agent.pde_journal import CandidatePlan, ExperimentJournal


def _write_llm_record(path, *, provider="hkustgz_gpt", content="{}"):
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "timestamp": "2026-05-15T00:00:00+08:00",
                    "elapsed_seconds": 1.0,
                    "provider": provider,
                    "model": "gpt-5.3-chat",
                    "messages": [{"role": "user", "content": "next experiment"}],
                    "response": {"content": content},
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _add_completed_node(journal, *, action_type, metric=None, params=None):
    node = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis=f"{action_type} hypothesis",
            action_type=action_type,
            params=params or {},
            expected_effect="improve or diagnose",
            risk="controlled",
        )
    )
    journal.update_result(node.id, success=True, metrics=metric or {}, artifacts={})
    return node


def test_audit_autonomous_study_rejects_bootstrap_planner_records(tmp_path):
    study = tmp_path / "study"
    study.mkdir()
    journal = ExperimentJournal(study / "journal.json")
    _add_completed_node(journal, action_type="inspect_data")
    _add_completed_node(journal, action_type="code_patch")
    _add_completed_node(journal, action_type="finetune_checkpoint", metric={"competition_score_proxy": 70.0})
    _add_completed_node(journal, action_type="evaluate_candidate", metric={"competition_score_proxy": 71.0})
    failed = journal.append_plan(
        CandidatePlan(
            intent="debug",
            hypothesis="negative control should fail",
            action_type="finetune_checkpoint",
            params={"temporal_stride": 1},
        )
    )
    journal.update_result(failed.id, success=False, metrics={}, artifacts={}, error="negative control failed")
    _write_llm_record(study / "planner_logs.log", provider="bootstrap")

    with pytest.raises(AutonomyAuditError, match="synthetic planner provider"):
        audit_autonomous_study(study, task="task1", metric="competition_score_proxy")


def test_audit_autonomous_study_accepts_real_llm_research_chain(tmp_path):
    study = tmp_path / "study"
    study.mkdir()
    journal = ExperimentJournal(study / "journal.json")
    _add_completed_node(journal, action_type="inspect_data")
    _add_completed_node(
        journal,
        action_type="code_patch",
        params={"source_files": ["third_party/baseline/PDEBench/pdebench/models/fno/fno.py"]},
    )
    _add_completed_node(
        journal,
        action_type="finetune_checkpoint",
        metric={"competition_score_proxy": 70.0},
        params={
            "source_files": ["third_party/baseline/PDEBench/pdebench/models/fno/fno.py"],
            "base_checkpoint": "checkpoints/extracted/1D_Burgers_Sols_Nu0.001_FNO.pt",
            "temporal_stride": 5,
        },
    )
    _add_completed_node(
        journal,
        action_type="evaluate_candidate",
        metric={"competition_score_proxy": 71.0},
    )
    failed = journal.append_plan(
        CandidatePlan(
            intent="debug",
            hypothesis="stride1 negative control exposes time-scale mismatch",
            action_type="finetune_checkpoint",
            params={"temporal_stride": 1},
        )
    )
    journal.update_result(failed.id, success=False, metrics={}, artifacts={}, error="worse validation score")
    for _ in range(5):
        _write_llm_record(study / "planner_logs.log")

    report = audit_autonomous_study(study, task="task1", metric="competition_score_proxy")

    assert report["ok"] is True
    assert report["llm_call_count"] == 5
    assert report["metric_experiment_count"] == 2
