from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Operator(StrEnum):
    DRAFT = "draft"
    IMPROVE = "improve"
    DEBUG = "debug"
    ABLATE = "ablate"
    FUSION = "fusion"


TERMINAL_STATUSES = {"static_failed", "shape_failed", "failed", "rejected"}


@dataclass
class Metrics:
    overall_mse: float | None = None
    short_mse: float | None = None
    long_stat_error: float | None = None
    per_nu_mse: dict[str, float] = field(default_factory=dict)
    worst_nu_mse: float | None = None
    heldout_nu_mse: float | None = None
    nu_estimation_mae: float | None = None
    runtime_sec: float | None = None
    shape_pass: bool = False
    first_10_pass: bool = False
    uses_true_nu_at_test: bool = False
    compliance_pass: bool = False
    reward: float = -0.5
    official_score_estimate: float | None = None
    rel_mse_seg1: float | None = None
    rel_mse_seg2: float | None = None
    rel_mse_seg3: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Metrics | None":
        return None if data is None else cls(**data)


@dataclass
class Node:
    node_id: str
    signature: str
    parent_ids: list[str]
    operator: Operator | str
    hypothesis: str
    code_dir: str
    artifact_dir: str
    checkpoint: str | None = None
    status: str = "created"
    visits: int = 0
    mean_score: float = 0.0
    best_score: float = float("-inf")
    metrics: Metrics | None = None
    lineage: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    failure_reason: str | None = None
    log_path: str | None = None
    response_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["operator"] = str(self.operator)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Node":
        copy = dict(data)
        copy["metrics"] = Metrics.from_dict(copy.get("metrics"))
        return cls(**copy)

    def update_status(self, status: str, failure_reason: str | None = None) -> None:
        self.status = status
        self.failure_reason = failure_reason
        self.updated_at = utc_now()

    def apply_reward(self, reward: float) -> None:
        self.visits += 1
        self.mean_score += (reward - self.mean_score) / self.visits
        self.best_score = max(self.best_score, reward)
        self.updated_at = utc_now()
