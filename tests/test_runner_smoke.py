from chem_evolve_agent.workflow.runner import run_target_smoke


def test_run_target_smoke_returns_ranked_candidates():
    candidates, logs = run_target_smoke(target_id="target", rounds=2, per_round=4)
    assert candidates
    assert candidates[0].score.total >= candidates[-1].score.total
    assert any('"round": 1' in line or '"round":1' in line for line in logs)
    assert any('"event": "reflect"' in line or '"event":"reflect"' in line for line in logs)
