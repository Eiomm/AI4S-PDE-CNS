"""State container for the DS-Agent workflow."""

from __future__ import annotations

import pickle
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from dslighting.state.base import State


class DSAgentState(BaseModel, State[Any]):
    """
    Holds the state for a DS-Agent workflow execution, primarily the running log
    which accumulates summaries of each experimental step.
    """

    running_log: str = ""
    final_code: str = ""
    last_plan: str = ""  # 保存最新的 plan，用于后续节点访问
    extra_state: dict[str, Any] = Field(default_factory=dict)
    created_timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def created_at(self) -> datetime:
        """When this state object was created."""
        return self.created_timestamp

    def get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Get a workflow state value by key."""
        if key in type(self).model_fields:
            return getattr(self, key)
        return self.extra_state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a workflow state value by key."""
        if key in type(self).model_fields and key not in {"created_timestamp"}:
            setattr(self, key, value)
        else:
            self.extra_state[key] = value

    def delete(self, key: str) -> bool:
        """Delete a workflow state key."""
        if key in self.extra_state:
            del self.extra_state[key]
            return True
        if key in {"running_log", "final_code", "last_plan"}:
            setattr(self, key, "")
            return True
        return False

    def clear(self) -> None:
        """Clear all mutable workflow state."""
        self.running_log = ""
        self.final_code = ""
        self.last_plan = ""
        self.extra_state.clear()

    def snapshot(self) -> bytes:
        """Create a checkpoint snapshot."""
        return pickle.dumps(self.model_dump(mode="python"))

    def restore(self, data: bytes) -> bool:
        """Restore this state from a checkpoint snapshot."""
        try:
            restored = self.model_validate(pickle.loads(data))
        except (pickle.PickleError, TypeError, ValueError):
            return False

        self.running_log = restored.running_log
        self.final_code = restored.final_code
        self.last_plan = restored.last_plan
        self.extra_state = restored.extra_state
        self.created_timestamp = restored.created_timestamp
        return True
