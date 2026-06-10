from pathlib import Path

from chem_evolve_agent.workflow.runner import run_target_agent_pipeline


def test_run_target_agent_pipeline_returns_ranked_candidates(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "0")
    candidates, logs = run_target_agent_pipeline(
        target_path=Path("examples/target.pdb"),
        out_dir=tmp_path,
        rounds=2,
        per_round=4,
        mode="proxy",
    )

    assert candidates
    assert candidates[0].score.total >= candidates[-1].score.total
    assert any('"event": "agent_plan"' in line or '"event":"agent_plan"' in line for line in logs)
    assert any('"event": "agent_observe"' in line or '"event":"agent_observe"' in line for line in logs)
    assert any('"event": "agent_memory"' in line or '"event":"agent_memory"' in line for line in logs)
    assert any('"runner": "agent"' in line or '"runner":"agent"' in line for line in logs)
