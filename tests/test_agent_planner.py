from chem_evolve_agent.agent.planner import HeuristicPlanner, LlmPlanner
from chem_evolve_agent.agent.state import AgentBudget, AgentState
from chem_evolve_agent.models import Candidate, Route, Score


class FakePlannerClient:
    available = True

    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, messages):
        self.messages = messages
        return self.payload


class BrokenPlannerClient:
    available = True

    def complete_json(self, messages):
        raise RuntimeError("boom")


def test_llm_planner_accepts_enabled_structured_action():
    state = AgentState(target_id="target", pocket_summary="pocket")
    budget = AgentBudget(rounds=3, per_round=8)
    client = FakePlannerClient(
        {
            "action_type": "generate_fragment",
            "limit": 4,
            "params": {},
            "rationale": "need broad fragment exploration",
        }
    )

    decision = LlmPlanner(client=client).plan_next(state, budget, memory=["none"])

    assert decision.planner_name == "llm_planner"
    assert decision.action.action_type == "generate_fragment"
    assert decision.action.limit == 4
    assert decision.fallback_reason is None


def test_llm_planner_rejects_disabled_action_and_falls_back():
    state = AgentState(target_id="target", pocket_summary="pocket")
    budget = AgentBudget(rounds=3, per_round=8)
    client = FakePlannerClient({"action_type": "generate_guided_mutation", "limit": 4})

    decision = LlmPlanner(client=client).plan_next(state, budget, memory=[])

    assert decision.action.action_type != "generate_guided_mutation"
    assert decision.fallback_reason
    assert "unavailable action" in decision.fallback_reason


def test_llm_planner_falls_back_when_client_fails():
    state = AgentState(target_id="target", pocket_summary="pocket")
    budget = AgentBudget(rounds=3, per_round=8)

    decision = LlmPlanner(client=BrokenPlannerClient()).plan_next(state, budget, memory=[])

    assert decision.planner_name.startswith("llm_planner->")
    assert decision.fallback_reason
    assert decision.action.action_type == "generate_seed"


def test_heuristic_planner_exploits_after_initial_candidates():
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
    )
    budget = AgentBudget(rounds=4, per_round=8)

    decision = HeuristicPlanner().plan_next(state, budget, memory=[])

    assert decision.action.action_type == "generate_guided_mutation"
