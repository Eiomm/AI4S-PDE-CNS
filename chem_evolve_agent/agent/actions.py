from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ActionType = Literal[
    "generate_seed",
    "generate_mutation",
    "generate_fragment",
    "generate_llm",
    "generate_guided_mutation",
]


class AgentAction(BaseModel):
    action_id: str
    action_type: ActionType
    round_index: int
    limit: int
    params: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class AvailableAction(BaseModel):
    action_type: ActionType
    description: str
    enabled: bool = True
    reason: str = ""
    required_tools: List[str] = Field(default_factory=list)


class PlannerDecision(BaseModel):
    action: AgentAction
    planner_name: str
    available_actions: List[AvailableAction] = Field(default_factory=list)
    memory_used: List[str] = Field(default_factory=list)
    fallback_reason: Optional[str] = None
    raw_payload: Dict[str, Any] = Field(default_factory=dict)


class ActionExecution(BaseModel):
    action: AgentAction
    tool_name: str
    generated_smiles: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class AgentObservation(BaseModel):
    action_id: str
    action_type: ActionType
    tool_name: str
    generated_count: int = 0
    accepted_count: int = 0
    skipped_invalid: int = 0
    skipped_duplicate: int = 0
    notes: List[str] = Field(default_factory=list)
