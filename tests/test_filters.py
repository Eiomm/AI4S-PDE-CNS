from chem_evolve_agent.evaluators.local_proxy import score_smiles_proxy


def test_proxy_score_is_deterministic():
    first = score_smiles_proxy("CCO")
    second = score_smiles_proxy("CCO")
    assert first == second


def test_proxy_penalizes_invalid_smiles():
    score = score_smiles_proxy("not-a-smiles")
    assert score.molecule_score == 0
    assert "invalid_smiles" in score.penalties
