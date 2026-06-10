from chem_evolve_agent.models import Candidate, Route, Score


def test_score_total_prefers_better_composite():
    weak = Score(molecule_score=0.4, route_score=0.2)
    strong = Score(molecule_score=0.7, route_score=0.5)
    assert strong.total > weak.total


def test_score_total_uses_final_round_six_four_weighting():
    score = Score(molecule_score=1.0, route_score=0.0)
    assert score.total == 0.6
    score = Score(molecule_score=0.0, route_score=1.0)
    assert score.total == 0.4


def test_candidate_requires_smiles_and_route():
    candidate = Candidate(
        mol_smiles="CCO",
        route=Route(steps=["CCBr.O>>CCO"]),
        score=Score(molecule_score=0.5, route_score=0.5),
        metadata={"source": "test"},
    )
    assert candidate.mol_smiles == "CCO"
    assert candidate.route.steps[-1].endswith(">>CCO")
