from __future__ import annotations

from pydantic import BaseModel, Field

from chem_evolve_agent.agent.actions import AgentObservation, PlannerDecision
from chem_evolve_agent.agent.state import AgentState


class AgentMemoryEntry(BaseModel):
    kind: str
    text: str
    score: float = 0.0
    round_index: int = 0


class AgentMemory(BaseModel):
    entries: list[AgentMemoryEntry] = Field(default_factory=list)

    def retrieve(self, state: AgentState, limit: int = 6) -> list[str]:
        entries = sorted(self.entries, key=lambda item: item.score, reverse=True)
        memory = [entry.text for entry in entries[:limit]]
        ranked = state.ranked_candidates()
        if ranked:
            top = ranked[0]
            memory.insert(
                0,
                (
                    f"Current best molecule is {top.mol_smiles} with total={top.score.total:.3f}, "
                    f"molecule={top.score.molecule_score:.3f}, route={top.score.route_score:.3f}."
                ),
            )
        return memory[:limit]

    def observe(self, state: AgentState, decision: PlannerDecision, observation: AgentObservation) -> None:
        ranked = state.ranked_candidates()
        if ranked:
            top = ranked[0]
            self.entries.append(
                AgentMemoryEntry(
                    kind="top_candidate",
                    text=(
                        f"After {decision.action.action_type}, top candidate {top.mol_smiles} scored "
                        f"{top.score.total:.3f}; penalties={','.join(top.score.penalties) or 'none'}."
                    ),
                    score=top.score.total,
                    round_index=state.round_index,
                )
            )
        if observation.accepted_count == 0:
            self.entries.append(
                AgentMemoryEntry(
                    kind="failed_action",
                    text=(
                        f"{decision.action.action_type} accepted no molecules "
                        f"(invalid={observation.skipped_invalid}, duplicate={observation.skipped_duplicate})."
                    ),
                    score=0.0,
                    round_index=state.round_index,
                )
            )
