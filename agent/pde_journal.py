from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .logging import utc_now_iso


NodeStatus = Literal["draft", "running", "completed", "failed", "skipped"]


@dataclass
class CandidatePlan:
    intent: str
    hypothesis: str
    action_type: str
    params: dict[str, Any] = field(default_factory=dict)
    expected_effect: str = ""
    risk: str = ""


@dataclass
class CandidateNode:
    plan: CandidatePlan
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    step: int = -1
    status: NodeStatus = "draft"
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    review: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    @property
    def is_buggy(self) -> bool:
        return self.status == "failed" or self.error is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateNode":
        data = dict(payload)
        data["plan"] = CandidatePlan(**data["plan"])
        return cls(**data)


class ExperimentJournal:
    """Persistent AIDE-style experiment tree for PDE runs."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def read(self) -> list[CandidateNode]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        raw_nodes = payload.get("nodes", payload if isinstance(payload, list) else [])
        return [CandidateNode.from_dict(item) for item in raw_nodes]

    def write(self, nodes: list[CandidateNode]) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": utc_now_iso(),
            "nodes": [node.to_dict() for node in nodes],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return self.path

    def get(self, node_id: str) -> CandidateNode:
        for node in self.read():
            if node.id == node_id:
                return node
        raise KeyError(f"unknown journal node: {node_id}")

    def append_plan(self, plan: CandidatePlan, *, parent_id: str | None = None) -> CandidateNode:
        nodes = self.read()
        node = CandidateNode(plan=plan, parent_id=parent_id, step=len(nodes))
        if parent_id is not None:
            for existing in nodes:
                if existing.id == parent_id:
                    if node.id not in existing.children_ids:
                        existing.children_ids.append(node.id)
                    existing.updated_at = utc_now_iso()
                    break
            else:
                raise KeyError(f"unknown parent journal node: {parent_id}")
        nodes.append(node)
        self.write(nodes)
        return node

    def update_result(
        self,
        node_id: str,
        *,
        success: bool,
        metrics: dict[str, float] | None = None,
        artifacts: dict[str, Any] | None = None,
        error: str | None = None,
        review: dict[str, Any] | None = None,
    ) -> CandidateNode:
        nodes = self.read()
        for node in nodes:
            if node.id != node_id:
                continue
            node.status = "completed" if success else "failed"
            node.metrics = dict(metrics or {})
            node.artifacts = dict(artifacts or {})
            node.error = error
            node.review = dict(review or {})
            node.updated_at = utc_now_iso()
            self.write(nodes)
            return node
        raise KeyError(f"unknown journal node: {node_id}")

    def mark_running(self, node_id: str) -> CandidateNode:
        nodes = self.read()
        for node in nodes:
            if node.id == node_id:
                node.status = "running"
                node.updated_at = utc_now_iso()
                self.write(nodes)
                return node
        raise KeyError(f"unknown journal node: {node_id}")

    def best(self, *, metric: str = "mse", maximize: bool = False) -> CandidateNode | None:
        candidates = [
            node
            for node in self.read()
            if node.status == "completed" and metric in node.metrics and node.error is None
        ]
        if not candidates:
            return None
        key = lambda node: float(node.metrics[metric])
        return max(candidates, key=key) if maximize else min(candidates, key=key)

    def summary(self, *, metric: str = "mse", limit: int = 8) -> str:
        rows = []
        for node in self.read()[-limit:]:
            metric_value = node.metrics.get(metric)
            rows.append(
                {
                    "id": node.id,
                    "step": node.step,
                    "parent_id": node.parent_id,
                    "status": node.status,
                    "action_type": node.plan.action_type,
                    "hypothesis": node.plan.hypothesis,
                    "metric": metric_value,
                    "error": node.error,
                }
            )
        return json.dumps(rows, ensure_ascii=False, indent=2)
