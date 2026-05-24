from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .evaluator import Evaluator
from .gpu_queue import GPUQueue
from .graph_db import GraphDB, atomic_write_json
from .llm_backend import LLMBackend
from .node_schema import Metrics, Node, Operator
from .operators import apply_llm_to_child, create_child_workspace, create_seed_nodes
from .research_log import append_research_event_many
from .selector import select_parent


RESEARCH_BRANCH_NAMES = [
    "long_rollout_stability",
    "nu_conditioned_dynamics",
    "residual_delta_rollout",
    "spectral_error_control",
    "temporal_consistency_search",
    "multi_regime_expert_search",
    "uncertainty_weighted_rollout",
    "auto_ml_architecture_sweep",
]


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: Path


def _persist_node(node: Node) -> None:
    atomic_write_json(Path(node.artifact_dir).parent / "node.json", node.to_dict())


def _locked(lock: threading.Lock | None):
    return lock if lock is not None else nullcontext()


def _research_log_path(db: GraphDB) -> Path:
    return db.workspace / "task2_logs.jsonl"


def create_run_context(db: GraphDB, run_id: str | None = None) -> RunContext:
    if run_id is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{stamp}_{uuid4().hex[:6]}"
    run_dir = db.workspace / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(run_id=run_id, run_dir=run_dir)


def _node_research_log_path(db: GraphDB, node_id: str) -> Path:
    return db.workspace / "nodes" / node_id / "logs" / "research_loop.jsonl"


def _run_node_research_log_path(run_context: RunContext, node_id: str) -> Path:
    return run_context.run_dir / "nodes" / node_id / "research_loop.jsonl"


def _event_log_paths(db: GraphDB, event: dict, run_context: RunContext | None = None) -> list[Path]:
    paths = [_research_log_path(db)]
    node_id = event.get("node_id")
    if node_id:
        paths.append(_node_research_log_path(db, node_id))
    if run_context is not None:
        paths.append(run_context.run_dir / "task2_logs.jsonl")
        if node_id:
            paths.append(_run_node_research_log_path(run_context, node_id))
    return paths


def _append_task_log(
    db: GraphDB,
    event: dict,
    log_lock: threading.Lock | None = None,
    run_context: RunContext | None = None,
) -> None:
    payload = dict(event)
    if run_context is not None:
        payload.setdefault("run_id", run_context.run_id)
    append_research_event_many(_event_log_paths(db, payload, run_context), payload, lock=log_lock)


def _snapshot_node_for_run(run_context: RunContext | None, node: Node) -> None:
    if run_context is None:
        return

    node_root = Path(node.code_dir).parent
    dst_root = run_context.run_dir / "nodes" / node.node_id
    (dst_root / "code").mkdir(parents=True, exist_ok=True)
    (dst_root / "logs").mkdir(parents=True, exist_ok=True)
    (dst_root / "artifacts").mkdir(parents=True, exist_ok=True)

    atomic_write_json(dst_root / "node.json", node.to_dict())
    for src, dst in [
        (Path(node.code_dir) / "train.py", dst_root / "code" / "train.py"),
        (node_root / "metrics.json", dst_root / "metrics.json"),
        (node_root / "metrics_full.json", dst_root / "metrics_full.json"),
        (node_root / "logs" / "llm_response.md", dst_root / "logs" / "llm_response.md"),
        (node_root / "logs" / "train.log", dst_root / "logs" / "train.log"),
    ]:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    artifact_dir = Path(node.artifact_dir)
    for src_name in ["task2_logs.log", "task2_time.csv"]:
        src = artifact_dir / src_name
        if src.exists():
            shutil.copy2(src, dst_root / "artifacts" / src_name)


def _append_db_event(db: GraphDB, event: dict, graph_lock: threading.Lock | None = None) -> None:
    with _locked(graph_lock):
        db.append_event(event)


def _upsert_persist_save(db: GraphDB, graph: dict, node: Node, graph_lock: threading.Lock | None = None) -> None:
    with _locked(graph_lock):
        db.upsert_node(graph, node)
        _persist_node(node)
        db.save_graph(graph)


def _make_llm_backend(args: argparse.Namespace) -> LLMBackend | None:
    if not args.use_llm:
        return None
    return LLMBackend(
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        model=args.llm_model,
        reasoning_effort=args.llm_reasoning_effort,
    )


def _branch_slug(branch_index: int) -> str:
    return RESEARCH_BRANCH_NAMES[(branch_index - 1) % len(RESEARCH_BRANCH_NAMES)]


def initialize_graph(db: GraphDB) -> dict:
    graph = db.load_graph()
    if graph.get("nodes"):
        return graph
    db.workspace.mkdir(parents=True, exist_ok=True)
    for node in create_seed_nodes(db.workspace):
        node.update_status("code_generated")
        db.upsert_node(graph, node)
        _persist_node(node)
    db.save_graph(graph)
    db.append_event({"event": "initialized_seed_nodes", "count": len(graph["nodes"])})
    return graph


def backup_reward(graph: dict, db: GraphDB, node: Node, reward: float) -> None:
    seen = set()
    stack = [node.node_id, *node.parent_ids]
    while stack:
        node_id = stack.pop()
        if node_id in seen or node_id not in graph["nodes"]:
            continue
        seen.add(node_id)
        current = Node.from_dict(graph["nodes"][node_id])
        current.apply_reward(reward)
        graph["nodes"][node_id] = current.to_dict()
        stack.extend(current.parent_ids)


def _select_parents(nodes: list[Node], k: int) -> list[Node]:
    """Select up to k distinct parents, avoiding terminal nodes."""
    candidates = [node for node in nodes if node.status not in {"static_failed", "shape_failed", "failed", "rejected"}]
    if not candidates:
        raise ValueError("no selectable nodes")

    def priority(node: Node) -> tuple[float, float, float, str]:
        reward = float("-inf") if node.metrics is None else node.metrics.reward
        score = max(reward, node.mean_score, node.best_score)
        novelty = 0.05 if node.visits == 0 else 0.0
        full_train_bonus = 0.1 if node.status == "full_train_passed" else 0.0
        return (score + novelty + full_train_bonus, reward, node.mean_score, node.updated_at)

    # Keep diversity, but represent each signature by its best observed node.
    best_by_signature: dict[str, Node] = {}
    for node in candidates:
        current = best_by_signature.get(node.signature)
        if current is None or priority(node) > priority(current):
            best_by_signature[node.signature] = node

    result = sorted(best_by_signature.values(), key=priority, reverse=True)[:k]

    # Fill with the best remaining scored nodes if we do not have enough signatures.
    if len(result) < k:
        for node in sorted(candidates, key=priority, reverse=True):
            if node not in result:
                result.append(node)
            if len(result) >= k:
                break
    return result[:k]


def _run_single_child(
    args: argparse.Namespace,
    db: GraphDB,
    graph: dict,
    parent: Node,
    *,
    round_index: int,
    branch_index: int,
    run_context: RunContext | None = None,
    graph_lock: threading.Lock | None = None,
    log_lock: threading.Lock | None = None,
) -> Node:
    """Create one child, optionally apply LLM, run static check and cheap probe."""
    start = time.time()
    branch_slug = _branch_slug(branch_index)
    child = create_child_workspace(
        db.workspace,
        parent,
        Operator.IMPROVE,
        round_index=round_index,
        branch_index=branch_index,
        node_slug=branch_slug,
    )
    child.update_status("code_generated")
    _upsert_persist_save(db, graph, child, graph_lock)
    _append_db_event(db, {"event": "child_created", "node_id": child.node_id, "parent_id": parent.node_id}, graph_lock)
    _append_task_log(
        db,
        {
            "event": "child_created",
            "elapsed_seconds": time.time() - start,
            "node_id": child.node_id,
            "parent_id": parent.node_id,
            "operator": child.operator.value,
            "tool_calls": [
                {
                    "name": "create_child_workspace",
                    "arguments": {
                        "round_index": round_index,
                        "branch_index": branch_index,
                        "node_slug": branch_slug,
                        "parent_id": parent.node_id,
                    },
                }
            ],
        },
        log_lock,
        run_context,
    )

    llm = _make_llm_backend(args)
    with _locked(graph_lock):
        prompt_nodes = db.nodes(graph)
    extra_log_paths = []
    event_defaults = {}
    if run_context is not None:
        extra_log_paths = [
            run_context.run_dir / "task2_logs.jsonl",
            _run_node_research_log_path(run_context, child.node_id),
        ]
        event_defaults = {"run_id": run_context.run_id}
    llm_ok, llm_err = apply_llm_to_child(
        child=child,
        parent=parent,
        all_nodes=prompt_nodes,
        llm=llm,
        prompts_dir=Path(__file__).parent / "prompts",
        research_log_path=_research_log_path(db),
        extra_log_paths=extra_log_paths,
        event_defaults=event_defaults,
        log_lock=log_lock,
    )
    if not llm_ok:
        child.metrics = Metrics(reward=-1.0)
        child.update_status("failed", f"llm_failed: {llm_err}")
        _upsert_persist_save(db, graph, child, graph_lock)
        _snapshot_node_for_run(run_context, child)
        _append_db_event(db, {"event": "llm_failed", "node_id": child.node_id, "error": llm_err}, graph_lock)
        _append_task_log(
            db,
            {
                "event": "llm_failed",
                "elapsed_seconds": time.time() - start,
                "node_id": child.node_id,
                "parent_id": parent.node_id,
                "response_id": child.response_id,
                "tool_calls": [{"name": "llm_code_generation", "error": llm_err}],
            },
            log_lock,
            run_context,
        )
        return child
    _upsert_persist_save(db, graph, child, graph_lock)
    _snapshot_node_for_run(run_context, child)

    evaluator = Evaluator(args.data_dir)
    static_ok, reasons = evaluator.static_check(child.code_dir)
    if not static_ok:
        child.metrics = Metrics(reward=-1.0)
        child.update_status("static_failed", "; ".join(reasons))
        with _locked(graph_lock):
            db.upsert_node(graph, child)
            _persist_node(child)
            backup_reward(graph, db, child, child.metrics.reward)
            db.save_graph(graph)
        _snapshot_node_for_run(run_context, child)
        _append_task_log(
            db,
            {
                "event": "metrics_recorded",
                "elapsed_seconds": time.time() - start,
                "node_id": child.node_id,
                "parent_id": parent.node_id,
                "response_id": child.response_id,
                "metrics": child.metrics.__dict__,
                "tool_calls": [{"name": "static_check", "result": "failed", "reasons": reasons}],
            },
            log_lock,
            run_context,
        )
        _append_db_event(
            db,
            {"event": "round_finished", "node_id": child.node_id, "status": child.status, "metrics": None},
            graph_lock,
        )
        return child

    child.update_status("preflight_passed")
    _upsert_persist_save(db, graph, child, graph_lock)
    _snapshot_node_for_run(run_context, child)

    cmd = [
        sys.executable,
        str((Path(child.code_dir) / "train.py").resolve()),
        "--data-dir",
        str(Path(args.data_dir).resolve()),
        "--out-dir",
        str(Path(child.artifact_dir).resolve()),
        "--epochs",
        str(args.cheap_epochs),
        "--cheap-probe",
    ]
    child.update_status("running")
    _upsert_persist_save(db, graph, child, graph_lock)
    _append_task_log(
        db,
        {
            "event": "trial_started",
            "elapsed_seconds": time.time() - start,
            "node_id": child.node_id,
            "parent_id": parent.node_id,
            "response_id": child.response_id,
            "tool_calls": [
                {
                    "name": "run_experiment",
                    "arguments": {
                        "command": cmd,
                        "artifact_dir": child.artifact_dir,
                        "log_path": child.log_path,
                        "timeout_sec": args.timeout_sec,
                    },
                }
            ],
        },
        log_lock,
        run_context,
    )
    job = GPUQueue(db.workspace).run(child.node_id, cmd, Path(child.log_path), args.timeout_sec)
    _append_task_log(
        db,
        {
            "event": "trial_finished",
            "elapsed_seconds": job.get("runtime_sec") or (time.time() - start),
            "node_id": child.node_id,
            "parent_id": parent.node_id,
            "response_id": child.response_id,
            "status": job["status"],
            "returncode": job["returncode"],
            "runtime_sec": job["runtime_sec"],
            "tool_calls": [{"name": "run_experiment", "result": job}],
        },
        log_lock,
        run_context,
    )

    if job["status"] != "completed":
        child.metrics = Metrics(runtime_sec=job["runtime_sec"], reward=-1.0)
        child.update_status("failed", f"cheap probe failed with returncode {job['returncode']}")
    else:
        metrics = evaluator.validation_metrics(Path(child.artifact_dir) / "val_pred.hdf5", runtime_sec=job["runtime_sec"])
        shape_metrics = evaluator.shape_check(Path(child.artifact_dir) / "task2_pred.hdf5")
        metrics.shape_pass = shape_metrics.shape_pass
        metrics.first_10_pass = shape_metrics.first_10_pass
        metrics.uses_true_nu_at_test = shape_metrics.uses_true_nu_at_test
        metrics.compliance_pass = metrics.shape_pass and metrics.first_10_pass and not metrics.uses_true_nu_at_test
        metrics.reward = evaluator.compute_reward(metrics)
        child.metrics = metrics
        child.update_status("cheap_probe_passed" if metrics.compliance_pass else "shape_failed")
        atomic_write_json(Path(child.artifact_dir).parent / "metrics.json", metrics.__dict__)
    with _locked(graph_lock):
        db.upsert_node(graph, child)
        _persist_node(child)
        backup_reward(graph, db, child, child.metrics.reward if child.metrics else -1.0)
        db.save_graph(graph)
    _snapshot_node_for_run(run_context, child)
    _append_task_log(
        db,
        {
            "event": "metrics_recorded",
            "elapsed_seconds": time.time() - start,
            "node_id": child.node_id,
            "parent_id": parent.node_id,
            "response_id": child.response_id,
            "status": child.status,
            "metrics": None if child.metrics is None else child.metrics.__dict__,
            "tool_calls": [{"name": "record_metrics", "artifact": str(Path(child.artifact_dir).parent / "metrics.json")}],
        },
        log_lock,
        run_context,
    )
    _append_db_event(
        db,
        {
            "event": "round_finished",
            "node_id": child.node_id,
            "status": child.status,
            "metrics": None if child.metrics is None else child.metrics.__dict__,
        },
        graph_lock,
    )
    return child


def _next_round_index(graph: dict) -> int:
    return int(graph.get("round_index", 0)) + 1


def run_round(
    args: argparse.Namespace,
    db: GraphDB,
    graph: dict,
    round_index: int | None = None,
    run_context: RunContext | None = None,
) -> list[Node]:
    nodes = db.nodes(graph)
    # Select up to children_per_round distinct parents
    k = max(1, args.children_per_round)
    parents = _select_parents(nodes, k)
    round_index = round_index or _next_round_index(graph)
    run_context = run_context or create_run_context(db)
    graph_lock = threading.Lock()
    log_lock = threading.Lock()

    children: list[Node] = []
    worker_count = 1
    if len(parents) > 1:
        if args.parallel_gpu:
            worker_count = max(1, min(len(parents), args.parallel_gpu))
        elif args.parallel_llm:
            worker_count = min(len(parents), 4)

    _append_task_log(
        db,
        {
            "event": "round_started",
            "round_index": round_index,
            "tool_calls": [
                {
                    "name": "select_research_parents",
                    "arguments": {
                        "parents": [parent.node_id for parent in parents],
                        "children_per_round": args.children_per_round,
                        "worker_count": worker_count,
                    },
                }
            ],
        },
        log_lock,
        run_context,
    )

    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_parent = {
                executor.submit(
                    _run_single_child,
                    args,
                    db,
                    graph,
                    parent,
                    round_index=round_index,
                    branch_index=idx,
                    run_context=run_context,
                    graph_lock=graph_lock,
                    log_lock=log_lock,
                ): parent
                for idx, parent in enumerate(parents, start=1)
            }
            for future in as_completed(future_to_parent):
                try:
                    child = future.result()
                    children.append(child)
                except Exception as exc:
                    _append_db_event(db, {"event": "scheduler_error", "error": repr(exc)}, graph_lock)
                    _append_task_log(
                        db,
                        {
                            "event": "scheduler_error",
                            "round_index": round_index,
                            "tool_calls": [{"name": "run_research_branch", "error": repr(exc)}],
                        },
                        log_lock,
                        run_context,
                    )
    else:
        for idx, parent in enumerate(parents, start=1):
            try:
                child = _run_single_child(
                    args,
                    db,
                    graph,
                    parent,
                    round_index=round_index,
                    branch_index=idx,
                    run_context=run_context,
                    graph_lock=graph_lock,
                    log_lock=log_lock,
                )
                children.append(child)
            except Exception as exc:
                _append_db_event(db, {"event": "scheduler_error", "error": repr(exc)}, graph_lock)
                _append_task_log(
                    db,
                    {
                        "event": "scheduler_error",
                        "round_index": round_index,
                        "tool_calls": [{"name": "run_research_branch", "error": repr(exc)}],
                    },
                    log_lock,
                    run_context,
                )

    with graph_lock:
        graph["round_index"] = max(int(graph.get("round_index", 0)), round_index)
        db.save_graph(graph)

    return children


def _run_full_train(
    args: argparse.Namespace,
    db: GraphDB,
    graph: dict,
    node_id: str,
    run_context: RunContext | None = None,
) -> dict:
    """Run full training on a promoted node."""
    start = time.time()
    node = Node.from_dict(graph["nodes"][node_id])
    cmd = [
        sys.executable,
        str((Path(node.code_dir) / "train.py").resolve()),
        "--data-dir", str(Path(args.data_dir).resolve()),
        "--out-dir", str(Path(node.artifact_dir).resolve()),
        "--epochs", str(args.full_epochs),
    ]
    node.update_status("full_training")
    db.upsert_node(graph, node)
    _persist_node(node)
    _append_task_log(
        db,
        {
            "event": "trial_started",
            "stage": "full_train",
            "elapsed_seconds": time.time() - start,
            "node_id": node.node_id,
            "response_id": node.response_id,
            "tool_calls": [
                {
                    "name": "run_full_train",
                    "arguments": {
                        "command": cmd,
                        "artifact_dir": node.artifact_dir,
                        "timeout_sec": args.timeout_sec,
                    },
                }
            ],
        },
        run_context=run_context,
    )
    job = GPUQueue(db.workspace).run(node.node_id, cmd, Path(node.log_path).with_suffix(".full_train.log"), args.timeout_sec)
    _append_task_log(
        db,
        {
            "event": "trial_finished",
            "stage": "full_train",
            "elapsed_seconds": job.get("runtime_sec") or (time.time() - start),
            "node_id": node.node_id,
            "response_id": node.response_id,
            "status": job["status"],
            "returncode": job["returncode"],
            "runtime_sec": job["runtime_sec"],
            "tool_calls": [{"name": "run_full_train", "result": job}],
        },
        run_context=run_context,
    )
    
    evaluator = Evaluator(args.data_dir)
    if job["status"] == "completed":
        metrics = evaluator.validation_metrics(Path(node.artifact_dir) / "val_pred.hdf5", runtime_sec=job["runtime_sec"])
        shape_metrics = evaluator.shape_check(Path(node.artifact_dir) / "task2_pred.hdf5")
        metrics.shape_pass = shape_metrics.shape_pass
        metrics.first_10_pass = shape_metrics.first_10_pass
        metrics.uses_true_nu_at_test = shape_metrics.uses_true_nu_at_test
        metrics.compliance_pass = metrics.shape_pass and metrics.first_10_pass and not metrics.uses_true_nu_at_test
        metrics.reward = evaluator.compute_reward(metrics)
        node.metrics = metrics
        node.update_status("full_train_passed" if metrics.compliance_pass else "full_train_failed")
        atomic_write_json(Path(node.artifact_dir).parent / "metrics_full.json", metrics.__dict__)
    else:
        node.metrics = Metrics(runtime_sec=job["runtime_sec"], reward=-1.0)
        node.update_status("full_train_failed", f"returncode {job['returncode']}")
    db.upsert_node(graph, node)
    _persist_node(node)
    _snapshot_node_for_run(run_context, node)
    _append_task_log(
        db,
        {
            "event": "metrics_recorded",
            "stage": "full_train",
            "elapsed_seconds": time.time() - start,
            "node_id": node.node_id,
            "response_id": node.response_id,
            "status": node.status,
            "metrics": None if node.metrics is None else node.metrics.__dict__,
            "tool_calls": [{"name": "record_metrics", "artifact": str(Path(node.artifact_dir).parent / "metrics_full.json")}],
        },
        run_context=run_context,
    )
    db.append_event({"event": "full_train_finished", "node_id": node.node_id, "status": node.status})
    return node.to_dict()


def _best_compliant_reward(graph: dict) -> float | None:
    rewards = [
        node.metrics.reward
        for node in (Node.from_dict(item) for item in graph.get("nodes", {}).values())
        if node.metrics is not None and node.metrics.compliance_pass
    ]
    return max(rewards) if rewards else None


def _format_metric_summary(node: Node) -> str:
    if node.metrics is None:
        return f"{node.node_id}: status={node.status}, metrics=None"
    metrics = node.metrics
    return (
        f"{node.node_id}: status={node.status}, reward={metrics.reward}, "
        f"overall_mse={metrics.overall_mse}, worst_nu_mse={metrics.worst_nu_mse}, "
        f"score_estimate={metrics.official_score_estimate}"
    )


def _select_full_train_candidates(
    db: GraphDB,
    graph: dict,
    promoted: list[dict],
    *,
    limit: int,
    already_trained: set[str],
) -> list[Node]:
    candidates = []
    for row in promoted:
        node_id = row["node_id"]
        if node_id in already_trained:
            continue
        node = db.get_node(graph, node_id)
        if node.status in {"full_training", "full_train_passed"}:
            continue
        candidates.append(node)
        if len(candidates) >= limit:
            break
    return candidates


def _log_feedback_reflection(
    args: argparse.Namespace,
    db: GraphDB,
    graph: dict,
    *,
    round_index: int,
    trained_nodes: list[Node],
    best_before: float | None,
    best_after: float | None,
    stale_rounds: int,
    run_context: RunContext | None,
) -> None:
    trained_summary = "\n".join(_format_metric_summary(node) for node in trained_nodes) or "No node was promoted to full train this round."
    leaderboard = db.update_leaderboard(graph)
    top_summary = "\n".join(
        f"{idx + 1}. {row['node_id']}: reward={row['reward']}, overall_mse={row['overall_mse']}, status={row['status']}"
        for idx, row in enumerate(leaderboard[:5])
    )
    response = (
        f"Round {round_index} autonomous feedback for next LLM round.\n\n"
        f"Cheap probes have been evaluated, promoted candidates were full-trained, and full metrics are now the feedback signal.\n"
        f"Best compliant reward before full train: {best_before}\n"
        f"Best compliant reward after full train: {best_after}\n"
        f"Stale rounds without >= {args.min_reward_delta} improvement: {stale_rounds}\n\n"
        f"Full-train results:\n{trained_summary}\n\n"
        f"Current leaderboard:\n{top_summary}\n\n"
        f"Instruction for next LLM round: use these full-train metrics, failure reasons, and long-horizon errors "
        f"to decide whether to exploit the current best line or explore a structurally different hypothesis."
    )
    _append_task_log(
        db,
        {
            "event": "llm_feedback_reflection",
            "round_index": round_index,
            "response": response,
            "tool_calls": [
                {
                    "name": "autonomous_feedback_loop",
                    "result": {
                        "trained_nodes": [node.node_id for node in trained_nodes],
                        "best_before": best_before,
                        "best_after": best_after,
                        "stale_rounds": stale_rounds,
                    },
                }
            ],
        },
        run_context=run_context,
    )


def _auto_full_train_after_round(
    args: argparse.Namespace,
    db: GraphDB,
    graph: dict,
    *,
    round_index: int,
    run_context: RunContext | None,
    already_trained: set[str],
    previous_best: float | None,
    stale_rounds: int,
) -> tuple[float | None, int]:
    best_before = _best_compliant_reward(graph)
    db.update_leaderboard(graph)
    promoted = db.update_promoted(graph, min_reward=0.0)
    candidates = _select_full_train_candidates(
        db,
        graph,
        promoted,
        limit=max(1, args.full_train_top_k),
        already_trained=already_trained,
    )
    trained_nodes = []
    for node in candidates:
        _run_full_train(args, db, graph, node.node_id, run_context=run_context)
        already_trained.add(node.node_id)
        trained_nodes.append(db.get_node(graph, node.node_id))

    db.save_graph(graph)
    db.update_leaderboard(graph)
    db.update_promoted(graph, min_reward=0.0)
    best_after = _best_compliant_reward(graph)

    if previous_best is None:
        new_best = best_after
        stale = 0
    elif best_after is not None and best_after > previous_best + args.min_reward_delta:
        new_best = best_after
        stale = 0
    else:
        new_best = previous_best
        stale = stale_rounds + 1

    _log_feedback_reflection(
        args,
        db,
        graph,
        round_index=round_index,
        trained_nodes=trained_nodes,
        best_before=best_before,
        best_after=best_after,
        stale_rounds=stale,
        run_context=run_context,
    )
    return new_best, stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/Task2")
    parser.add_argument("--workspace", default="workspace")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--cheap-epochs", type=int, default=1)
    parser.add_argument("--full-epochs", type=int, default=20)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM code generation/improvement")
    parser.add_argument("--llm-model", default="deepseek-chat", help="LLM model name")
    parser.add_argument("--llm-base-url", default=None, help="LLM API base URL (defaults to DeepSeek)")
    parser.add_argument("--llm-api-key", default=None, help="LLM API key (defaults to env vars)")
    parser.add_argument("--llm-reasoning-effort", default=None, help="OpenAI-style reasoning effort (low/medium/high)")
    parser.add_argument("--auto-full-train", action="store_true", help="Auto full-train promoted nodes with reward > 0")
    parser.add_argument("--children-per-round", type=int, default=1, help="Number of children to generate per round")
    parser.add_argument("--parallel-llm", action="store_true", help="Parallelize LLM code generation across children")
    parser.add_argument("--parallel-gpu", type=int, default=0, help="Number of parallel GPU jobs (0=serial)")
    parser.add_argument("--run-id", default=None, help="Optional experiment run id for workspace/runs/<run_id>")
    parser.add_argument("--full-train-top-k", type=int, default=1, help="Number of promoted cheap-probe nodes to full-train after each round")
    parser.add_argument("--convergence-patience", type=int, default=0, help="Stop autonomous loop after this many rounds without reward improvement; 0 disables")
    parser.add_argument("--min-reward-delta", type=float, default=1e-4, help="Minimum reward improvement considered progress for convergence")
    args = parser.parse_args(argv)

    db = GraphDB(args.workspace)
    graph = initialize_graph(db)
    run_context = create_run_context(db, args.run_id)
    _append_task_log(
        db,
        {
            "event": "experiment_started",
            "tool_calls": [
                {
                    "name": "scheduler_main",
                    "arguments": {
                        "rounds": args.rounds,
                        "children_per_round": args.children_per_round,
                        "parallel_llm": args.parallel_llm,
                        "parallel_gpu": args.parallel_gpu,
                        "use_llm": args.use_llm,
                    },
                }
            ],
        },
        run_context=run_context,
    )
    already_full_trained: set[str] = set()
    best_full_reward = _best_compliant_reward(graph)
    stale_rounds = 0

    for _ in range(args.rounds):
        round_index = _next_round_index(graph)
        try:
            run_round(args, db, graph, round_index=round_index, run_context=run_context)
        except Exception as exc:
            db.append_event({"event": "scheduler_error", "error": repr(exc)})
            _append_task_log(
                db,
                {"event": "scheduler_error", "tool_calls": [{"name": "run_round", "error": repr(exc)}]},
                run_context=run_context,
            )
        graph["round_index"] = max(int(graph.get("round_index", 0)), round_index)
        db.save_graph(graph)
        if args.auto_full_train:
            best_full_reward, stale_rounds = _auto_full_train_after_round(
                args,
                db,
                graph,
                round_index=round_index,
                run_context=run_context,
                already_trained=already_full_trained,
                previous_best=best_full_reward,
                stale_rounds=stale_rounds,
            )
            if args.convergence_patience > 0 and stale_rounds >= args.convergence_patience:
                _append_task_log(
                    db,
                    {
                        "event": "convergence_stopped",
                        "round_index": round_index,
                        "response": (
                            f"Autonomous loop stopped after {stale_rounds} stale round(s). "
                            f"No full-train reward improved by at least {args.min_reward_delta}."
                        ),
                        "tool_calls": [
                            {
                                "name": "check_convergence",
                                "result": {
                                    "stale_rounds": stale_rounds,
                                    "patience": args.convergence_patience,
                                    "best_reward": best_full_reward,
                                },
                            }
                        ],
                    },
                    run_context=run_context,
                )
                break
    db.save_graph(graph)
    leaderboard = db.update_leaderboard(graph)
    promoted = db.update_promoted(graph, min_reward=0.0)
    _append_task_log(
        db,
        {
            "event": "experiment_finished",
            "tool_calls": [
                {
                    "name": "scheduler_main",
                    "result": {
                        "nodes": len(graph["nodes"]),
                        "leaderboard": leaderboard[:3],
                        "promoted": promoted[:3],
                    },
                }
            ],
        },
        run_context=run_context,
    )
    
    print(json.dumps({"nodes": len(graph["nodes"]), "leaderboard": leaderboard[:3], "promoted": promoted[:3]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
