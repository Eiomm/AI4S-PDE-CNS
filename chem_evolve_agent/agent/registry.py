from __future__ import annotations

from chem_evolve_agent.agent.actions import AvailableAction
from chem_evolve_agent.agent.state import AgentState


def build_available_actions(state: AgentState, llm_available: bool = False) -> list[AvailableAction]:
    has_candidates = bool(state.candidates)
    return [
        AvailableAction(
            action_type="generate_seed",
            description="Start or refresh a stable deterministic baseline set of small molecules.",
        ),
        AvailableAction(
            action_type="generate_mutation",
            description="Sample a simple mutation/template generator for aromatic CNS-like molecules.",
        ),
        AvailableAction(
            action_type="generate_fragment",
            description="Pull diverse molecules from the hand-built fragment catalog.",
        ),
        AvailableAction(
            action_type="generate_llm",
            description="Ask the molecule-generation LLM to propose diverse valid SMILES.",
            enabled=llm_available,
            reason="" if llm_available else "LLM disabled or unavailable; executor will fall back to seed generation.",
        ),
        AvailableAction(
            action_type="generate_guided_mutation",
            description="Mutate the current top-scoring candidate/scaffold to exploit a promising branch.",
            enabled=has_candidates,
            reason="" if has_candidates else "No accepted candidates yet.",
        ),
    ]


def enabled_action_types(actions: list[AvailableAction]) -> set[str]:
    return {action.action_type for action in actions if action.enabled}
