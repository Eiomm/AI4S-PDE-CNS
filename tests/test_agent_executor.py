from chem_evolve_agent.agent.actions import AgentAction
from chem_evolve_agent.agent.executor import ToolExecutor
from chem_evolve_agent.agent.state import AgentState
from chem_evolve_agent.chemistry.smiles import is_valid_smiles
from chem_evolve_agent.models import Candidate, Route, Score


def test_guided_mutation_ignores_unknown_parent_smiles():
    state = AgentState(
        target_id="target",
        pocket_summary="pocket",
        round_index=2,
        candidates=[
            Candidate(
                mol_smiles="CCO",
                route=Route(steps=["CCBr.O>>CCO"]),
                score=Score(molecule_score=0.8, route_score=0.7),
            )
        ],
        seen_smiles={"CCO"},
    )
    action = AgentAction(
        action_id="a",
        action_type="generate_guided_mutation",
        round_index=2,
        limit=3,
        params={"parent_smiles": "not-a-known-candidate"},
    )

    execution = ToolExecutor().execute(action, state)

    assert execution.generated_smiles
    assert "ignored_unknown_parent=not-a-known-candidate" in execution.notes
    assert "parent_smiles=CCO" in execution.notes


def test_guided_mutation_outputs_valid_unique_smiles():
    state = AgentState(
        target_id="target",
        pocket_summary="pocket",
        round_index=2,
        candidates=[
            Candidate(
                mol_smiles="FCCOc1ccccc1",
                route=Route(steps=["START.O>>FCCOc1ccccc1"]),
                score=Score(molecule_score=0.58, route_score=0.75),
            )
        ],
        seen_smiles={"FCCOc1ccccc1"},
    )
    action = AgentAction(
        action_id="guided",
        action_type="generate_guided_mutation",
        round_index=2,
        limit=8,
    )

    execution = ToolExecutor().execute(action, state)

    assert len(execution.generated_smiles) >= 4
    assert len(set(execution.generated_smiles)) == len(execution.generated_smiles)
    assert all(is_valid_smiles(smiles) for smiles in execution.generated_smiles)
    assert "FCCOc1ccccc1" not in execution.generated_smiles


def test_run_seed_changes_generated_seed_candidates():
    state = AgentState(target_id="target", pocket_summary="pocket", round_index=0)
    action = AgentAction(
        action_id="seed",
        action_type="generate_seed",
        round_index=0,
        limit=8,
    )

    first = ToolExecutor(run_seed=1).execute(action, state).generated_smiles
    second = ToolExecutor(run_seed=2).execute(action, state).generated_smiles

    assert first != second
