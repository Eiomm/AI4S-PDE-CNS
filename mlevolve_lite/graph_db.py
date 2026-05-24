from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .node_schema import Node, utc_now


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


class GraphDB:
    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace)
        self.nodes_dir = self.workspace / "nodes"
        self.graph_path = self.workspace / "graph.json"
        self.leaderboard_path = self.workspace / "leaderboard.json"
        self.promoted_path = self.workspace / "promoted.json"
        self.events_path = self.workspace / "events.jsonl"

    def load_graph(self) -> dict[str, Any]:
        if not self.graph_path.exists():
            return {"nodes": {}, "edges": [], "root_ids": [], "last_updated": utc_now()}
        return json.loads(self.graph_path.read_text(encoding="utf-8"))

    def save_graph(self, graph: dict[str, Any]) -> None:
        graph["last_updated"] = utc_now()
        atomic_write_json(self.graph_path, graph)

    def upsert_node(self, graph: dict[str, Any], node: Node) -> None:
        graph.setdefault("nodes", {})[node.node_id] = node.to_dict()
        if not node.parent_ids and node.node_id not in graph.setdefault("root_ids", []):
            graph["root_ids"].append(node.node_id)
        for parent_id in node.parent_ids:
            edge = [parent_id, node.node_id]
            if edge not in graph.setdefault("edges", []):
                graph["edges"].append(edge)

    def nodes(self, graph: dict[str, Any]) -> list[Node]:
        return [Node.from_dict(item) for item in graph.get("nodes", {}).values()]

    def get_node(self, graph: dict[str, Any], node_id: str) -> Node:
        return Node.from_dict(graph["nodes"][node_id])

    def append_event(self, event: dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        event.setdefault("timestamp", utc_now())
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(event), sort_keys=True) + "\n")

    def update_leaderboard(self, graph: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
        rows = []
        for node in self.nodes(graph):
            metrics = node.metrics
            if metrics is None:
                continue
            rows.append(
                {
                    "node_id": node.node_id,
                    "status": node.status,
                    "overall_mse": metrics.overall_mse,
                    "worst_nu_mse": metrics.worst_nu_mse,
                    "reward": metrics.reward,
                    "compliance_pass": metrics.compliance_pass,
                    "updated_at": node.updated_at,
                }
            )
        rows.sort(key=lambda row: row["reward"], reverse=True)
        atomic_write_json(self.leaderboard_path, rows[:limit])
        return rows[:limit]

    def load_leaderboard(self) -> list[dict[str, Any]]:
        if not self.leaderboard_path.exists():
            return []
        return json.loads(self.leaderboard_path.read_text(encoding="utf-8"))

    def update_promoted(self, graph: dict[str, Any], min_reward: float = 0.0, limit: int = 5) -> list[dict[str, Any]]:
        promoted = []
        for node in self.nodes(graph):
            metrics = node.metrics
            if metrics is None or not metrics.compliance_pass or metrics.reward < min_reward:
                continue
            promoted.append(
                {
                    "node_id": node.node_id,
                    "overall_mse": metrics.overall_mse,
                    "worst_nu_mse": metrics.worst_nu_mse,
                    "reward": metrics.reward,
                    "code_dir": node.code_dir,
                    "checkpoint": node.checkpoint,
                    "prediction_path": str(Path(node.artifact_dir) / "task2_pred.hdf5"),
                    "promoted_at": utc_now(),
                    "reason": "top compliant cheap-probe node",
                }
            )
        promoted.sort(key=lambda row: row["reward"], reverse=True)
        atomic_write_json(self.promoted_path, promoted[:limit])
        return promoted[:limit]
