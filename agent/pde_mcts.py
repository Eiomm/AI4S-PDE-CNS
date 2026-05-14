from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pde_journal import CandidateNode, CandidatePlan, ExperimentJournal


@dataclass
class _MCTSStats:
    visits: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits


NON_SCORING_REWARD = -1_000_000.0


class PDEMCTSRunner:
    """Windows-friendly MCTS adapter over the PDE experiment journal."""

    def __init__(
        self,
        *,
        journal: ExperimentJournal,
        metric: str = "mse",
        maximize: bool = False,
        study_name: str = "task1-mcts-mock",
        max_children: int = 2,
        exploration_constant: float = 1.414,
        execution: str = "mock",
        executor: Any | None = None,
        reviewer: Any | None = None,
    ):
        if max_children < 1:
            raise ValueError("max_children must be at least 1")
        if execution not in {"mock", "controlled"}:
            raise ValueError("execution must be 'mock' or 'controlled'")
        if execution == "controlled" and (executor is None or reviewer is None):
            raise ValueError("controlled execution requires executor and reviewer")
        self.journal = journal
        self.metric = metric
        self.maximize = maximize
        self.study_name = study_name
        self.max_children = max_children
        self.exploration_constant = exploration_constant
        self.execution = execution
        self.executor = executor
        self.reviewer = reviewer
        self._stats: dict[str, _MCTSStats] = {}

    def run(self, *, actions: list[dict[str, Any]]) -> dict[str, Any]:
        root = self._ensure_root()
        self._rebuild_stats(root.id)
        trace = []

        for index, action in enumerate(actions):
            selected_parent = self._select_expandable(root.id)
            child = self.journal.append_plan(self._plan_from_action(action), parent_id=selected_parent.id)
            updated_child = self._evaluate_child(child, action=action, index=index, selected_parent=selected_parent)
            metric_value = updated_child.metrics.get(self.metric)
            path = self._path_to_root(updated_child.id)
            reward = self._reward_for_node(updated_child, metric_value, index=index)
            if updated_child.status == "completed" and updated_child.error is None and metric_value is not None:
                metric_value = self._validate_metric(
                    metric_value,
                    action_name=updated_child.plan.params.get("name", f"action-{index}"),
                    source=f"reviewed metrics[{self.metric!r}]",
                )
            for node_id in path:
                stats = self._ensure_stats(node_id)
                stats.visits += 1
                stats.total_reward += reward
            trace.append(
                {
                    "step": index,
                    "action": updated_child.plan.params["name"],
                    "node_id": updated_child.id,
                    "selected_parent_id": selected_parent.id,
                    "metric": metric_value,
                    "reward": reward,
                    "path": path,
                }
            )

        return self._report(root.id, trace)

    def _evaluate_child(
        self,
        child: CandidateNode,
        *,
        action: dict[str, Any],
        index: int,
        selected_parent: CandidateNode,
    ) -> CandidateNode:
        if self.execution == "controlled":
            running = self.journal.mark_running(child.id)
            execution = self.executor.execute(running)
            return self.reviewer.review_execution(
                running,
                success=execution.success,
                metrics=execution.metrics,
                artifacts=execution.artifacts,
                error=execution.error,
            )

        metric_value = self._metric_from_action(action)
        return self.journal.update_result(
            child.id,
            success=True,
            metrics={self.metric: metric_value},
            artifacts={
                "mock_action_index": index,
                "mock_action": action.get("name", f"action-{index}"),
                "selected_parent_id": selected_parent.id,
            },
            review={
                "analysis": f"Mock MCTS evaluated {action.get('name', f'action-{index}')}.",
                "metric": self.metric,
                "metric_value": metric_value,
            },
        )

    def _ensure_root(self) -> CandidateNode:
        nodes = self.journal.read()
        for node in nodes:
            if (
                node.plan.action_type == "mcts_root"
                and node.plan.params.get("mcts_root") is True
                and node.plan.params.get("study_name") == self.study_name
            ):
                return node
        return self.journal.append_plan(
            CandidatePlan(
                intent="root",
                hypothesis="PDE MCTS search root",
                action_type="mcts_root",
                params={
                    "mcts_root": True,
                    "study_name": self.study_name,
                    "metric": self.metric,
                    "maximize": self.maximize,
                },
                expected_effect="anchor PDE MCTS search for this study",
                risk="none; root node only records search metadata",
            )
        )

    def _metric_from_action(self, action: dict[str, Any]) -> float:
        name = str(action.get("name", "mock-action"))
        if "metric" in action:
            return self._validate_metric(action["metric"], action_name=name, source="metric")
        if isinstance(action.get("metrics"), dict):
            metrics = action["metrics"]
            if self.metric in metrics:
                return self._validate_metric(metrics[self.metric], action_name=name, source=f"metrics[{self.metric!r}]")
        metrics_path = action.get("metrics_path") or action.get("metrics_json")
        if metrics_path is not None:
            path = Path(metrics_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"metrics file for action {name!r} must contain a JSON object")
            if self.metric in payload:
                return self._validate_metric(payload[self.metric], action_name=name, source=str(path))
        raise ValueError(f"missing metric {self.metric!r} for action {name!r}")

    def _validate_metric(self, value: Any, *, action_name: str, source: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"non-numeric metric {self.metric!r} for action {action_name!r} from {source}")
        try:
            metric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric metric {self.metric!r} for action {action_name!r} from {source}") from exc
        if not math.isfinite(metric_value):
            raise ValueError(f"metric {self.metric!r} for action {action_name!r} must be finite")
        return metric_value

    def _plan_from_action(self, action: dict[str, Any]) -> CandidatePlan:
        name = str(action.get("name", "mock-action"))
        params = dict(action.get("params", {}))
        params["name"] = name
        return CandidatePlan(
            intent=str(action.get("intent", "improve")),
            hypothesis=str(action.get("hypothesis", name)),
            action_type=str(action.get("action_type", "mock_mcts_action")),
            params=params,
            expected_effect=str(action.get("expected_effect", "update mock search metric")),
            risk=str(action.get("risk", "none; deterministic mock action")),
        )

    def _select_expandable(self, node_id: str) -> CandidateNode:
        node = self.journal.get(node_id)
        if len(node.children_ids) < self.max_children:
            return node
        children = [self.journal.get(child_id) for child_id in node.children_ids]
        best_child = max(children, key=lambda child: (self._uct(child, node), -child.step))
        return self._select_expandable(best_child.id)

    def _uct(self, node: CandidateNode, parent: CandidateNode) -> float:
        stats = self._ensure_stats(node.id)
        if stats.visits == 0:
            return float("inf")
        parent_visits = max(self._ensure_stats(parent.id).visits, 1)
        exploration = self.exploration_constant * math.sqrt(math.log(parent_visits) / stats.visits)
        return stats.mean_reward + exploration

    def _path_to_root(self, node_id: str) -> list[str]:
        nodes = {node.id: node for node in self.journal.read()}
        path = []
        current_id: str | None = node_id
        while current_id is not None:
            path.append(current_id)
            current_id = nodes[current_id].parent_id
        return path

    def _reward(self, metric_value: float) -> float:
        return metric_value if self.maximize else -metric_value

    def _reward_for_node(self, node: CandidateNode, metric_value: Any, *, index: int) -> float:
        if node.status == "completed" and node.error is None and metric_value is not None:
            metric_value = self._validate_metric(
                metric_value,
                action_name=node.plan.params.get("name", f"action-{index}"),
                source=f"reviewed metrics[{self.metric!r}]",
            )
            return self._reward(metric_value)
        return NON_SCORING_REWARD

    def _ensure_stats(self, node_id: str) -> _MCTSStats:
        if node_id not in self._stats:
            self._stats[node_id] = _MCTSStats()
        return self._stats[node_id]

    def _rebuild_stats(self, root_id: str) -> None:
        if self._stats:
            return
        self._ensure_stats(root_id)
        for node in self._descendants_from(root_id):
            if node.id == root_id or node.status not in {"completed", "failed"}:
                continue
            reward = self._reward_for_node(node, node.metrics.get(self.metric), index=node.step)
            for node_id in self._path_to_root(node.id):
                stats = self._ensure_stats(node_id)
                stats.visits += 1
                stats.total_reward += reward

    def _report(self, root_id: str, trace: list[dict[str, Any]]) -> dict[str, Any]:
        nodes = self._descendants_from(root_id)
        best = self._best_from(nodes)
        return {
            "metric": {"name": self.metric, "maximize": self.maximize},
            "best_node": self._node_summary(best) if best is not None else None,
            "nodes": [self._node_summary(node) for node in nodes],
            "trace": trace,
            "search_stats": {
                node_id: {
                    "visits": stats.visits,
                    "total_reward": stats.total_reward,
                    "mean_reward": stats.mean_reward,
                }
                for node_id, stats in self._stats.items()
            },
        }

    def _descendants_from(self, root_id: str) -> list[CandidateNode]:
        nodes_by_id = {node.id: node for node in self.journal.read()}
        if root_id not in nodes_by_id:
            raise KeyError(f"unknown journal node: {root_id}")
        scoped = []
        pending = [root_id]
        while pending:
            node_id = pending.pop(0)
            node = nodes_by_id[node_id]
            scoped.append(node)
            pending.extend(child_id for child_id in node.children_ids if child_id in nodes_by_id)
        return scoped

    def _best_from(self, nodes: list[CandidateNode]) -> CandidateNode | None:
        candidates = [
            node
            for node in nodes
            if node.status == "completed" and self.metric in node.metrics and node.error is None
        ]
        if not candidates:
            return None
        key = lambda node: float(node.metrics[self.metric])
        return max(candidates, key=key) if self.maximize else min(candidates, key=key)

    def _node_summary(self, node: CandidateNode) -> dict[str, Any]:
        metric_value = node.metrics.get(self.metric)
        return {
            "id": node.id,
            "parent_id": node.parent_id,
            "children_ids": list(node.children_ids),
            "step": node.step,
            "status": node.status,
            "intent": node.plan.intent,
            "hypothesis": node.plan.hypothesis,
            "action_type": node.plan.action_type,
            "action": node.plan.params.get("name", node.plan.action_type),
            "metric": metric_value,
            "artifacts": dict(node.artifacts),
            "error": node.error,
        }


MockMCTSRunner = PDEMCTSRunner
