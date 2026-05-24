from __future__ import annotations

import math

from .node_schema import Node, TERMINAL_STATUSES


def selection_score(node: Node, total_visits: int, c: float = 1.4) -> float:
    if node.status in TERMINAL_STATUSES:
        return float("-inf")
    if node.metrics is not None and not node.metrics.compliance_pass and node.status in {"shape_failed", "rejected"}:
        return float("-inf")
    novelty_bonus = 0.2 if node.visits == 0 else 0.0
    risk_penalty = 0.5 if node.status in {"created", "code_generated"} else 0.0
    exploitation = node.mean_score
    exploration = c * math.sqrt(math.log(total_visits + 1) / (node.visits + 1))
    return exploitation + exploration + novelty_bonus - risk_penalty


def select_parent(nodes: list[Node]) -> Node:
    candidates = [node for node in nodes if node.status not in TERMINAL_STATUSES]
    if not candidates:
        raise ValueError("no selectable nodes")
    total_visits = sum(max(node.visits, 0) for node in candidates) + 1
    return max(candidates, key=lambda node: selection_score(node, total_visits))
