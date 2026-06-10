from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    mode: str = "proxy"
    rounds: int = 3
    per_round: int = 16
    top_k: int = 20
    smoke_fallback_routes: bool = True
    extra: Dict[str, Any] = Field(default_factory=dict)


def load_config(path: Optional[Path] = None) -> AgentConfig:
    if path is None:
        return AgentConfig()
    data = yaml.safe_load(path.read_text()) or {}
    known = {key: value for key, value in data.items() if key in AgentConfig.model_fields}
    extra = {key: value for key, value in data.items() if key not in AgentConfig.model_fields}
    return AgentConfig(**known, extra=extra)
