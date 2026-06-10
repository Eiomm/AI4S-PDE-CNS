from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    lesson_id: str
    context: Dict[str, Any] = Field(default_factory=dict)
    rule: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    confidence: float
    applies_when: List[str] = Field(default_factory=list)
    fails_when: List[str] = Field(default_factory=list)


class MemoryBank(BaseModel):
    records: List[MemoryRecord] = Field(default_factory=list)

    def promote_observations(self, observations: List[Dict[str, Any]]) -> List[MemoryRecord]:
        rules = Counter(str(item.get("rule", "")) for item in observations if item.get("rule"))
        promoted: List[MemoryRecord] = []
        for obs in observations:
            rule = str(obs.get("rule", ""))
            improvement = float(obs.get("score_improvement", 0.0))
            branches = rules[rule]
            if not rule or (branches < 2 and improvement < 0.10):
                continue
            record = MemoryRecord(
                lesson_id=f"lesson_{len(self.records) + len(promoted) + 1}",
                context=dict(obs.get("context", {})),
                rule=rule,
                evidence={"branches": branches, "score_improvement": improvement},
                confidence=min(1.0, 0.5 + 0.2 * branches + improvement),
                applies_when=list(obs.get("applies_when", [])),
                fails_when=list(obs.get("fails_when", [])),
            )
            promoted.append(record)
        self.records.extend(promoted)
        return promoted
