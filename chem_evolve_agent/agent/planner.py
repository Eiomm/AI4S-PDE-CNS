from __future__ import annotations

from typing import Any, Protocol

from chem_evolve_agent.agent.actions import ActionType, AgentAction, AvailableAction, PlannerDecision
from chem_evolve_agent.agent.registry import build_available_actions, enabled_action_types
from chem_evolve_agent.agent.state import AgentBudget, AgentState
from chem_evolve_agent.llm import LiteLlmClient


GENERATION_SEQUENCE: tuple[ActionType, ...] = (
    "generate_seed",
    "generate_mutation",
    "generate_fragment",
    "generate_llm",
    "generate_guided_mutation",
)


class AgentPlanner(Protocol):
    name: str

    def plan_next(self, state: AgentState, budget: AgentBudget, memory: list[str]) -> PlannerDecision:
        ...


class HeuristicPlanner:
    """Safe fallback planner with simple exploration/exploitation behavior."""

    name = "heuristic_planner"

    def plan_next(self, state: AgentState, budget: AgentBudget, memory: list[str]) -> PlannerDecision:
        # LLM 不可用或输出非法时走这里：前期探索，后期围绕当前 top 分子做 guided mutation。
        available_actions = build_available_actions(state, llm_available=False)
        if state.round_index >= 2 and state.candidates:
            action_type: ActionType = "generate_guided_mutation"
            rationale = "exploit current top candidate after initial exploration"
        else:
            action_type = GENERATION_SEQUENCE[state.round_index % 4]
            if action_type == "generate_llm":
                action_type = "generate_fragment"
            rationale = "safe deterministic exploration while LLM planning is unavailable"

        action = AgentAction(
            action_id=f"round_{state.round_index:03d}_{action_type}",
            action_type=action_type,
            round_index=state.round_index,
            limit=budget.per_round,
            rationale=rationale,
        )
        return PlannerDecision(
            action=action,
            planner_name=self.name,
            available_actions=available_actions,
            memory_used=memory,
        )


class LlmPlanner:
    name = "llm_planner"

    def __init__(self, client: LiteLlmClient | None = None, fallback: AgentPlanner | None = None):
        self.client = client or LiteLlmClient()
        self.fallback = fallback or HeuristicPlanner()

    def plan_next(self, state: AgentState, budget: AgentBudget, memory: list[str]) -> PlannerDecision:
        available_actions = build_available_actions(state, llm_available=self.client.available)
        enabled_actions = enabled_action_types(available_actions)
        if not self.client.available:
            return self._fallback(state, budget, memory, "llm_unavailable", available_actions)

        try:
            payload = self.client.complete_json(self._messages(state, budget, memory, available_actions))
            action = self._action_from_payload(payload, state, budget, enabled_actions)
            return PlannerDecision(
                action=action,
                planner_name=self.name,
                available_actions=available_actions,
                memory_used=memory,
                raw_payload=payload if isinstance(payload, dict) else {"payload": payload},
            )
        except Exception as exc:
            return self._fallback(state, budget, memory, f"{type(exc).__name__}: {exc}", available_actions)

    def _fallback(
        self,
        state: AgentState,
        budget: AgentBudget,
        memory: list[str],
        reason: str,
        available_actions: list[AvailableAction],
    ) -> PlannerDecision:
        decision = self.fallback.plan_next(state, budget, memory)
        decision.planner_name = f"{self.name}->{decision.planner_name}"
        decision.available_actions = available_actions
        decision.fallback_reason = reason
        return decision

    def _messages(
        self,
        state: AgentState,
        budget: AgentBudget,
        memory: list[str],
        available_actions: list[AvailableAction],
    ) -> list[dict[str, str]]:
        # 只让 LLM planner 选择“下一步动作”，不让它直接改状态或绕过工具。
        # 真正的化学合法性、去重、评分都在 executor/judge 层统一处理。
        actions_text = "\n".join(
            f"- {action.action_type}: enabled={action.enabled}; {action.description}; {action.reason}"
            for action in available_actions
        )
        ranked = state.ranked_candidates()[:5]
        candidates_text = "\n".join(
            f"- {candidate.mol_smiles}: total={candidate.score.total:.3f}, "
            f"mol={candidate.score.molecule_score:.3f}, route={candidate.score.route_score:.3f}"
            for candidate in ranked
        ) or "- none yet"
        memory_text = "\n".join(f"- {item}" for item in memory) or "- none"
        return [
            {
                "role": "system",
                "content": (
                    "You are the planner for an autonomous small-molecule design agent. "
                    "Choose exactly one enabled action for the next round. "
                    "Return only a JSON object with keys: action_type, limit, params, rationale. "
                    "Do not invent action names."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Target: {state.target_id}\n"
                    f"Pocket summary: {state.pocket_summary}\n"
                    f"Round: {state.round_index + 1}/{budget.rounds}\n"
                    f"Per-round molecule budget: {budget.per_round}\n"
                    f"Mode: {budget.mode}\n\n"
                    f"Available actions:\n{actions_text}\n\n"
                    f"Top candidates:\n{candidates_text}\n\n"
                    f"Memory:\n{memory_text}\n"
                ),
            },
        ]

    def _action_from_payload(
        self,
        payload: Any,
        state: AgentState,
        budget: AgentBudget,
        enabled_actions: set[str],
    ) -> AgentAction:
        if not isinstance(payload, dict):
            raise ValueError("planner payload must be a JSON object")
        action_type = str(payload.get("action_type", ""))
        if action_type not in enabled_actions:
            raise ValueError(f"planner selected unavailable action: {action_type}")

        # LLM 的输出只能落在 registry 声明的 enabled action 里；limit 也被夹在本轮预算内。
        # 这层防护能避免 prompt 漂移时调用不存在的工具或一次请求过量候选。
        limit = int(payload.get("limit") or budget.per_round)
        limit = max(1, min(limit, budget.per_round))
        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        rationale = str(payload.get("rationale", ""))[:500]
        return AgentAction(
            action_id=f"round_{state.round_index:03d}_{action_type}",
            action_type=action_type,  # type: ignore[arg-type]
            round_index=state.round_index,
            limit=limit,
            params=params,
            rationale=rationale,
        )


RoundRobinPlanner = HeuristicPlanner
