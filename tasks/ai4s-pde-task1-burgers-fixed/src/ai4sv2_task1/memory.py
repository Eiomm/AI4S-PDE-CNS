from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .hdf5_io import read_named_or_single
from .paths import resolve_path, task_root


def now_iso() -> str:
    """返回 UTC ISO 时间戳，用于 memory 记录的可追溯创建时间。"""

    return datetime.now(timezone.utc).isoformat()


def memory_root() -> Path:
    """Task1 memory OS 的根目录。

    目录结构采用用户调研文档里的分层：
    - contract：硬规则；
    - working：当前实验上下文；
    - episodic：每次实验 episode；
    - findings：验证过的正/负发现和 leaderboard；
    - failures：错误模式；
    - procedures：可复用流程；
    - wisdom：长期策略摘要。

    注意：这里仍然不是官方提交 log，也不是 LLM 原始对话。
    """

    return task_root() / "memory"


def episodic_runs_path() -> Path:
    """所有实验 episode 的 append-only JSONL 文件。"""

    return memory_root() / "episodic" / "runs.jsonl"


def metric_leaderboard_path() -> Path:
    """候选板 / leaderboard 的 CSV 路径。"""

    return memory_root() / "findings" / "metric_leaderboard.csv"


def task_rules_path() -> Path:
    """Agent 每轮必须读取的 Task1 硬规则入口。"""

    return memory_root() / "contract" / "task1_rules.yaml"


def strategy_summary_path() -> Path:
    """Agent 每轮读取的长期策略摘要入口。"""

    return memory_root() / "wisdom" / "strategy_summary.md"


def load_json(path: str | Path) -> Any:
    """读取 JSON 文件；调用方负责确认文件存在。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_yaml(path: str | Path) -> Any:
    """读取 YAML 文件；不存在时返回空 dict。"""

    source = Path(path)
    if not source.is_file():
        return {}
    return yaml.safe_load(source.read_text(encoding="utf-8")) or {}


def load_yaml_records(path: str | Path) -> list[dict[str, Any]]:
    """读取 YAML 记录列表；不存在或为空时返回空列表。

    findings 和 failures 需要人工可读、可审计，所以采用 YAML。
    这里约定顶层结构为 `records: [...]`。
    """

    source = Path(path)
    if not source.is_file():
        return []
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [item for item in payload["records"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """向 JSONL 文件追加一条记录。

    JSONL 适合作为 append-only registry：每条实验记录独立成行，后续即使
    某条记录损坏，也不会影响其他记录读取。
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件；不存在时返回空列表。"""

    source = Path(path)
    if not source.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number} is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{source}:{line_number} must be a JSON object")
        records.append(payload)
    return records


def _relative_or_string(path_value: str | None) -> str | None:
    """尽量把绝对路径压缩成相对 Task1 根目录的路径，减少 memory 冗余。"""

    if not path_value:
        return None
    path = Path(path_value)
    try:
        return path.resolve().relative_to(task_root()).as_posix()
    except Exception:
        return str(path_value)


def build_experiment_record(run_dir: str | Path, *, hypothesis: str, decision: str, tags: list[str] | None = None) -> dict[str, Any]:
    """从一次 run 的产物中提取长期 memory 记录。

    这个函数只抽取最小必要字段：路线、核心指标、校验状态、artifact 指针、
    checkpoint hash 和决策。完整 metadata / log 仍保留在 run 目录里。
    """

    run_path = resolve_path(run_dir)
    metadata = load_json(run_path / "metadata.json")
    metrics = metadata.get("metrics")
    validation = metadata.get("validation") or {}
    checkpoints = metadata.get("checkpoints") or []
    prediction_path = metadata.get("prediction_path")
    prediction_exists = bool(prediction_path and Path(prediction_path).is_file())
    checkpoint_exists = any(bool(item.get("path") and Path(item["path"]).is_file()) for item in checkpoints if isinstance(item, dict))
    artifact_exists = prediction_exists or checkpoint_exists

    # 若 prediction 存在，快速确认 dataset 可读，避免把坏 artifact 写成长记忆。
    if prediction_exists:
        _ = read_named_or_single(prediction_path, "tensor").shape

    route = str(metadata.get("route") or "unknown")
    run_name = str(metadata.get("run_name") or run_path.name)
    record = {
        "schema": "task1_experiment_record_v1",
        "record_id": f"task1:{run_name}",
        "task": "task1",
        "created_at": now_iso(),
        "route": route,
        "hypothesis": hypothesis,
        "changes": {
            "models": [
                {
                    "kind": item.get("kind"),
                    "checkpoint": _relative_or_string(item.get("path")),
                    "checkpoint_sha256": item.get("sha256"),
                    "weight": item.get("weight"),
                }
                for item in checkpoints
            ],
            "config": _relative_or_string(metadata.get("config_path")),
        },
        "metrics": metrics or {},
        "validation": {
            "shape": validation.get("shape"),
            "first_ten_match": validation.get("first_ten_match"),
            "finite": validation.get("finite"),
            "max_initial_error": validation.get("max_initial_error"),
        },
        "artifacts": {
            "run_dir": _relative_or_string(str(run_path)),
            "prediction": _relative_or_string(prediction_path),
            "metrics": _relative_or_string(metadata.get("metrics_path")),
            "metadata": _relative_or_string(str(run_path / "metadata.json")),
            "log": _relative_or_string(metadata.get("log_path")),
        },
        "decision": decision,
        "promotable": bool(decision in {"promote_candidate", "submit_ready"}),
        "quality": {
            "artifact_exists": artifact_exists,
            "metrics_verified": bool(metrics),
            "log_trace_available": bool(metadata.get("log_path") and Path(metadata["log_path"]).is_file()),
            "memory_review": "keep",
        },
        "tags": tags or [route],
    }
    return record


def export_run_memory(run_dir: str | Path, *, hypothesis: str, decision: str, tags: list[str] | None = None) -> Path:
    """在 run 目录下写出 `memory_export.json`，供长期 registry 追加。"""

    run_path = resolve_path(run_dir)
    record = build_experiment_record(run_path, hypothesis=hypothesis, decision=decision, tags=tags)
    output = run_path / "memory_export.json"
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def add_record_to_registry(record_path: str | Path) -> Path:
    """把 `memory_export.json` 追加到长期 episodic memory。

    旧版名字叫 registry；新版按照 v1 调研结果叫 `episodic/runs.jsonl`。
    函数名保持不变，是为了让脚本接口稳定。
    """

    record = load_json(record_path)
    registry = episodic_runs_path()
    append_jsonl(registry, record)
    return registry


def query_memory(*, route: str | None = None, tags: list[str] | None = None, limit: int = 8, max_chars: int = 6000) -> dict[str, Any]:
    """按确定性规则检索 memory，返回可直接放进 LLM prompt 的小包。

    当前按用户要求只读取 4 个入口：
    1. `memory/contract/task1_rules.yaml`
    2. `memory/episodic/runs.jsonl`
    3. `memory/findings/metric_leaderboard.csv`
    4. `memory/wisdom/strategy_summary.md`

    其它 findings/failures/procedures 先保留为人工参考，不进入默认 LLM
    prompt，避免 memory 系统一开始就过重。
    """

    records = read_jsonl(episodic_runs_path())
    selected: list[dict[str, Any]] = []
    requested_tags = set(tags or [])
    for record in reversed(records):
        if route and record.get("route") != route:
            continue
        record_tags = set(record.get("tags") or [])
        if requested_tags and not (requested_tags & record_tags):
            continue
        selected.append(record)
        if len(selected) >= limit:
            break

    candidates = read_leaderboard()[:5]
    strategy_summary = strategy_summary_path().read_text(encoding="utf-8") if strategy_summary_path().is_file() else ""
    packet = {
        "task_rules": load_yaml(task_rules_path()),
        "strategy_summary": strategy_summary,
        "best_candidates": candidates,
        "relevant_experiments": [
            {
                "record_id": item.get("record_id"),
                "route": item.get("route"),
                "hypothesis": item.get("hypothesis"),
                "metrics": item.get("metrics"),
                "decision": item.get("decision"),
                "artifacts": item.get("artifacts"),
                "tags": item.get("tags"),
            }
            for item in selected
        ],
        "retrieval_budget": {"max_records": limit, "max_chars": max_chars},
        "memory_sources": [
            "memory/contract/task1_rules.yaml",
            "memory/episodic/runs.jsonl",
            "memory/findings/metric_leaderboard.csv",
            "memory/wisdom/strategy_summary.md",
        ],
    }

    # 简单字符预算控制：超过预算时逐条减少 experiment examples。
    while len(json.dumps(packet, ensure_ascii=False)) > max_chars and packet["relevant_experiments"]:
        packet["relevant_experiments"].pop()
    return packet


def promote_candidate(record_id: str, *, slot: str, metric: str, value: float, blockers: list[str] | None = None) -> Path:
    """更新当前候选板 / metric leaderboard。

    promote 只是更新候选板，不代表可以直接提交；submit_ready 仍需要 replay、
    shape 校验、官方 log 追溯和提交目录校验。
    """

    path = metric_leaderboard_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [item for item in read_leaderboard() if item.get("slot") != slot]
    rows.append(
        {
            "slot": slot,
            "record_id": record_id,
            "metric": metric,
            "value": f"{float(value):.12g}",
            "submit_ready": "false",
            "blockers": " | ".join(blockers or []),
            "updated_at": now_iso(),
        }
    )
    write_leaderboard(rows)
    return path


def read_leaderboard() -> list[dict[str, Any]]:
    """读取 `findings/metric_leaderboard.csv`。

    CSV 用来给人快速扫描当前候选；脚本读取后会把数值字段转成 float，并按
    当前 Task1 分数从高到低排序。这样 `memory_query` 给 Agent 的
    `best_candidates` 不会被 slot 名字误导。
    """

    path = metric_leaderboard_path()
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if "value" in row:
            try:
                row["value"] = float(row["value"])
            except (TypeError, ValueError):
                pass
    return sorted(rows, key=lambda row: float(row.get("value", float("-inf"))) if isinstance(row.get("value"), (int, float)) else float("-inf"), reverse=True)


def write_leaderboard(rows: list[dict[str, Any]]) -> None:
    """写回候选板 CSV，字段顺序固定，方便 diff 和人工审查。"""

    path = metric_leaderboard_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["slot", "record_id", "metric", "value", "submit_ready", "blockers", "updated_at"]
    def sort_key(item: dict[str, Any]) -> tuple[float, str]:
        try:
            value = float(item.get("value", float("-inf")))
        except (TypeError, ValueError):
            value = float("-inf")
        return (-value, str(item.get("slot", "")))

    normalized = sorted(rows, key=sort_key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in normalized:
            writer.writerow({field: row.get(field, "") for field in fields})
