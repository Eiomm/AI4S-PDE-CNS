from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

from chem_evolve_agent.chem_ops import (
    build_score,
    canonicalize_smiles,
    generate_evolved_smiles,
    generate_internal_smiles,
    molecule_score,
    property_metrics,
    proxy_binding_score,
    sa_score,
)
from chem_evolve_agent.llm import LiteLlmClient
from chem_evolve_agent.models import Candidate, Route
from chem_evolve_agent.runtime_tools import (
    TargetContext,
    ToolError,
    json_event,
    load_target,
    run_aizynthfinder_route,
    run_sota_sbdd_generator,
    run_vina_binding_score,
    safe_name,
)


class AgentRunError(RuntimeError):
    pass


def run_agent_for_target(
    target_path: Path,
    out_dir: Path,
    rounds: int,
    per_round: int,
    mode: str,
    docking_limit: int,
    run_seed: int = 0,
    experience_file: Path | None = None,
) -> tuple[list[Candidate], list[str]]:
    if mode not in {"proxy", "docking", "competition"}:
        raise ValueError(f"unsupported mode: {mode}")
    target = load_target(target_path)
    target_signature = _target_signature(target)
    long_term_memory = _load_long_term_memory(experience_file, target)
    long_term_memory_smiles = {canonicalize_smiles(str(item["smiles"])) for item in long_term_memory}
    logs = [
        json_event(
            "target_loaded",
            target_id=target.target_id,
            path=str(target.path),
            atoms=target.atom_count,
            residues=len(target.residues),
        ),
        json_event(
            "pocket_summary",
            target_id=target.target_id,
            center=target.pocket_center,
            box_size=target.pocket_box_size,
            summary=target.pocket_summary,
        ),
        json_event(
            "agent_run_start",
            target_id=target.target_id,
            rounds=rounds,
            per_round=per_round,
            mode=mode,
            docking_limit=docking_limit,
            route_engine=_route_engine(),
            vina_feedback_per_round=_vina_feedback_per_round(mode),
            competition_context_paths=_agent_context_paths(),
        ),
        json_event(
            "agent_experience_loaded",
            target_id=target.target_id,
            target_signature=target_signature,
            memory_file=str(experience_file) if experience_file else None,
            loaded_count=len(long_term_memory),
        ),
    ]
    _progress(
        "start",
        target=target.target_id,
        rounds=rounds,
        per_round=per_round,
        mode=mode,
        route_engine=_route_engine(),
    )
    seen: set[str] = set()
    candidates: list[Candidate] = []
    rejection_memory: list[dict[str, Any]] = []
    target_seed_offset = _target_seed_offset(target)
    for round_index in range(rounds):
        _progress("round_start", target=target.target_id, round=round_index + 1, rounds=rounds)
        rejection_count_before_round = len(rejection_memory)
        memory = _merge_memory(long_term_memory, _elite_memory(candidates))
        round_brief = _round_brief(target, round_index, memory, rejection_memory, per_round)
        strategy = str(round_brief["strategy"])
        _progress(
            "strategy",
            target=target.target_id,
            round=round_index + 1,
            source=round_brief["source"],
            llm_candidates=len(round_brief["candidates"]),
        )
        logs.append(
            json_event(
                "agent_strategy",
                target_id=target.target_id,
                round=round_index,
                source=round_brief["source"],
                strategy=strategy,
                focus=round_brief["focus"],
                avoid=round_brief["avoid"],
                llm_candidate_count=len(round_brief["candidates"]),
                candidate_rationales=_candidate_rationales(round_brief["candidates"]),
            )
        )
        if memory:
            logs.append(json_event("agent_memory", target_id=target.target_id, round=round_index, elites=memory))
        recent_rejections = _recent_rejections(rejection_memory)
        if recent_rejections:
            logs.append(json_event("agent_rejection_memory", target_id=target.target_id, round=round_index, rejections=recent_rejections))
        plan = _plan(memory, round_brief)
        logs.append(json_event("agent_plan", target_id=target.target_id, round=round_index, actions=plan))
        round_seed = run_seed + round_index + target_seed_offset
        generated = _generate_candidates(
            plan,
            target,
            out_dir / "generation" / f"round_{round_index:03d}",
            per_round,
            round_seed,
            memory,
            round_brief,
        )
        _progress("generated", target=target.target_id, round=round_index + 1, count=len(generated), tools="+".join(plan))
        logs.append(
            json_event(
                "generate",
                target_id=target.target_id,
                round=round_index,
                count=len(generated),
                tools=plan,
                seed=round_seed,
                target_seed_offset=target_seed_offset,
            )
        )
        screened = _screen_generated_smiles(
            generated,
            seen,
            rejection_memory,
            logs,
            target.target_id,
            round_index,
            forbidden_smiles=long_term_memory_smiles,
        )
        selected = _select_for_route_planning(screened, mode, docking_limit, logs, target.target_id, round_index)
        _progress(
            "route_prefilter",
            target=target.target_id,
            round=round_index + 1,
            valid=len(screened),
            selected=len(selected),
            route_engine=_route_engine(),
        )
        round_candidates: list[Candidate] = []
        for route_index, smiles in enumerate(selected, start=1):
            seen.add(smiles)
            _progress(
                "route_start",
                target=target.target_id,
                round=round_index + 1,
                index=route_index,
                total=len(selected),
                route_engine=_route_engine(),
                smiles=smiles,
            )
            try:
                candidate = _evaluate_candidate(smiles, target, out_dir, mode, len(candidates), docking_limit)
            except Exception as exc:
                reason = str(exc)
                _remember_rejection(rejection_memory, smiles, reason, round_index)
                logs.append(json_event("reject", target_id=target.target_id, round=round_index, smiles=smiles, reason=reason))
                _progress(
                    "route_reject",
                    target=target.target_id,
                    round=round_index + 1,
                    index=route_index,
                    total=len(selected),
                    reason=reason,
                    smiles=smiles,
                )
                continue
            candidate.metadata.update({"round": round_index, "plan": plan, "memory_size": len(memory)})
            candidates.append(candidate)
            round_candidates.append(candidate)
            _progress(
                "route_accept",
                target=target.target_id,
                round=round_index + 1,
                index=route_index,
                total=len(selected),
                total_score=candidate.score.total,
                route_score=candidate.score.route_score,
                smiles=smiles,
            )
            logs.append(
                json_event(
                    "evaluate",
                    target_id=target.target_id,
                    round=round_index,
                    smiles=smiles,
                    molecule_score=candidate.score.molecule_score,
                    route_score=candidate.score.route_score,
                    route_validity_score=candidate.score.route_validity_score,
                    starting_material_availability_score=candidate.score.starting_material_availability_score,
                    step_penalty_score=candidate.score.step_penalty_score,
                    convergence_score=candidate.score.convergence_score,
                    balance_score=candidate.score.balance_score,
                    total_score=candidate.score.total,
                    binding_source=candidate.score.binding_source,
                    route_source=candidate.route.source,
                    property_prior_score=candidate.score.property_prior_score,
                    penalties=candidate.score.penalties,
                )
        )
        failed_feedback = _apply_round_vina_feedback(round_candidates, target, out_dir, mode, round_index, logs, rejection_memory)
        if failed_feedback:
            candidates = [candidate for candidate in candidates if candidate.mol_smiles not in failed_feedback]
            round_candidates = [candidate for candidate in round_candidates if candidate.mol_smiles not in failed_feedback]
        best_round = max(round_candidates, key=lambda item: item.score.total) if round_candidates else None
        best_overall = max(candidates, key=lambda item: item.score.total) if candidates else None
        logs.append(
            json_event(
                "agent_round_summary",
                target_id=target.target_id,
                round=round_index,
                generated_count=len(generated),
                screened_valid_unique_count=len(screened),
                selected_for_route_count=len(selected),
                accepted_count=len(round_candidates),
                rejected_count=len(rejection_memory) - rejection_count_before_round,
                failed_vina_feedback_count=len(failed_feedback),
                best_round_smiles=best_round.mol_smiles if best_round else None,
                best_round_total=best_round.score.total if best_round else None,
                best_overall_smiles=best_overall.mol_smiles if best_overall else None,
                best_overall_total=best_overall.score.total if best_overall else None,
            )
        )
        _progress(
            "round_done",
            target=target.target_id,
            round=round_index + 1,
            accepted=len(round_candidates),
            rejected=len(rejection_memory) - rejection_count_before_round,
            best=best_overall.score.total if best_overall else None,
        )
    if not candidates:
        raise AgentRunError("agent produced no candidates with valid score and route")
    if mode == "competition":
        candidates = _rerank_with_vina(candidates, target, out_dir, docking_limit, logs)
    candidates.sort(key=lambda item: item.score.total, reverse=True)
    logs.append(json_event("agent_rank", target_id=target.target_id, candidate_count=len(candidates), best_score=candidates[0].score.total))
    _progress("done", target=target.target_id, candidates=len(candidates), best=candidates[0].score.total)
    return candidates, logs


def append_agent_experience(memory_file: Path, target_path: Path, candidates: list[Candidate], top_k: int, run_dir: Path) -> int:
    if top_k <= 0:
        raise AgentRunError("top_k must be positive when writing agent experience")
    target = load_target(target_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "schema_version": 1,
            "event": "agent_experience",
            "created_at": timestamp,
            "target_id": target.target_id,
            "target_signature": _target_signature(target),
            "run_dir": str(run_dir),
            "rank": rank,
            "candidate": _experience_candidate_payload(candidate),
        }
        for rank, candidate in enumerate(candidates[:top_k], start=1)
    ]
    if not entries:
        return 0
    with memory_file.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return len(entries)


def _plan(memory: list[dict[str, Any]], round_brief: dict[str, Any]) -> list[str]:
    tools: list[str] = []
    if os.getenv("AI4S_SBDD_GENERATOR_CMD"):
        tools.append("sota_sbdd_generator")
    if round_brief["candidates"]:
        tools.append("llm_generator")
    if memory:
        tools.append("evolution_generator")
    tools.append("internal_rdkit_generator")
    return tools


def _target_seed_offset(target: TargetContext) -> int:
    fingerprint = "|".join(
        [
            target.target_id,
            str(target.atom_count),
            repr(target.pocket_center),
            repr(target.pocket_box_size),
            ",".join(target.residues[:24]),
        ]
    )
    return int(hashlib.blake2s(fingerprint.encode("utf-8"), digest_size=4).hexdigest(), 16)


def _target_signature(target: TargetContext) -> str:
    return f"{target.target_id}:{_target_seed_offset(target):08x}"


_PROGRESS_EVENT_LABELS = {
    "start": "开始运行",
    "round_start": "开始新一轮",
    "strategy": "生成本轮策略",
    "generated": "候选分子已生成",
    "route_prefilter": "路线规划前筛选",
    "route_start": "开始路线规划",
    "route_reject": "路线规划拒绝",
    "route_accept": "路线规划通过",
    "round_done": "本轮完成",
    "done": "运行完成",
    "vina_feedback_start": "开始 Vina 反馈",
    "vina_feedback_reject": "Vina 反馈拒绝",
    "vina_feedback_accept": "Vina 反馈通过",
    "llm_start": "开始调用 LLM",
    "llm_wait": "等待 LLM 返回",
    "llm_done": "LLM 返回完成",
    "llm_reject": "LLM 输出解析失败",
    "competition_dock_start": "开始竞赛终局 docking",
    "competition_dock_reject": "终局 docking 拒绝",
    "competition_dock_accept": "终局 docking 通过",
}


_PROGRESS_FIELD_LABELS = {
    "target": "靶点",
    "round": "轮次",
    "rounds": "总轮数",
    "per_round": "每轮候选数",
    "mode": "模式",
    "route_engine": "路线引擎",
    "source": "策略来源",
    "llm_candidates": "LLM候选数",
    "count": "数量",
    "tools": "工具",
    "valid": "有效数",
    "selected": "入选数",
    "index": "序号",
    "total": "总数",
    "reason": "原因",
    "smiles": "SMILES",
    "total_score": "总分",
    "route_score": "路线分",
    "accepted": "通过数",
    "rejected": "拒绝数",
    "best": "最佳分",
    "candidates": "候选数",
    "limit": "上限",
    "binding": "结合分",
    "energy": "能量",
    "reused": "复用Vina结果",
    "log_dir": "日志目录",
    "elapsed_seconds": "已等待秒数",
}


_PROGRESS_EVENT_GROUPS = {
    "start": "运行",
    "round_start": "轮次",
    "strategy": "策略",
    "generated": "生成",
    "route_prefilter": "路线",
    "route_start": "路线",
    "route_reject": "路线",
    "route_accept": "路线",
    "round_done": "轮次",
    "done": "运行",
    "vina_feedback_start": "Vina",
    "vina_feedback_reject": "Vina",
    "vina_feedback_accept": "Vina",
    "llm_start": "LLM",
    "llm_wait": "LLM",
    "llm_done": "LLM",
    "llm_reject": "LLM",
    "competition_dock_start": "复排",
    "competition_dock_reject": "复排",
    "competition_dock_accept": "复排",
}


_PROGRESS_GROUP_COLORS = {
    "运行": "36",
    "轮次": "34",
    "策略": "35",
    "生成": "32",
    "路线": "33",
    "Vina": "31",
    "LLM": "35",
    "复排": "31",
}


def _progress(event: str, **payload: Any) -> None:
    if os.getenv("AI4S_PROGRESS_STDERR", "1").strip().lower() in {"0", "false", "no", "off"}:
        return
    event_label = _PROGRESS_EVENT_LABELS.get(event, event)
    group = _PROGRESS_EVENT_GROUPS.get(event, "状态")
    parts = [f"[agent][{_progress_color(group, _PROGRESS_GROUP_COLORS.get(group, '36'))}] {event_label}", f"event={event}"]
    if "round" in payload and "rounds" in payload and payload.get("round") is not None and payload.get("rounds") is not None:
        parts.append(f"轮次={_format_progress_value(payload['round'])}/{_format_progress_value(payload['rounds'])}")
    if "index" in payload and "total" in payload and payload.get("index") is not None and payload.get("total") is not None:
        parts.append(f"序号={_format_progress_value(payload['index'])}/{_format_progress_value(payload['total'])}")
    for key, value in payload.items():
        if key == "rounds" and "round" in payload:
            continue
        if key == "round" and "rounds" in payload:
            continue
        if key == "total" and "index" in payload:
            continue
        if key == "index" and "total" in payload:
            continue
        if value is None:
            continue
        text = _format_progress_value(value)
        if key == "smiles" and len(text) > 72:
            text = text[:69] + "..."
        label = _PROGRESS_FIELD_LABELS.get(key, key)
        parts.append(f"{label}={text}")
    print(" | ".join(parts), file=sys.stderr, flush=True)


def _format_progress_value(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _progress_color(text: str, color_code: str) -> str:
    raw = os.getenv("AI4S_PROGRESS_COLOR", "auto").strip().lower()
    enabled = raw in {"1", "true", "yes", "on"} or (
        raw == "auto" and sys.stderr.isatty() and "NO_COLOR" not in os.environ
    )
    if not enabled:
        return text
    return f"\033[{color_code}m{text}\033[0m"


def _generate_candidates(
    plan: list[str],
    target: TargetContext,
    out_dir: Path,
    limit: int,
    seed: int,
    memory: list[dict[str, Any]],
    round_brief: dict[str, Any],
) -> list[str]:
    generated: list[str] = []
    for tool in plan:
        if tool == "sota_sbdd_generator":
            generated.extend(run_sota_sbdd_generator(target, out_dir / tool, limit))
        elif tool == "llm_generator":
            generated.extend(_brief_candidate_smiles(round_brief["candidates"]))
        elif tool == "evolution_generator":
            parents = [str(item["smiles"]) for item in memory]
            generated.extend(generate_evolved_smiles(parents, limit, seed))
        elif tool == "internal_rdkit_generator":
            generated.extend(generate_internal_smiles(limit=limit, seed=seed))
        else:
            raise AgentRunError(f"unknown generation tool: {tool}")
    return _unique_preserve_order(generated)[:limit]


def _screen_generated_smiles(
    generated: list[str],
    seen: set[str],
    rejection_memory: list[dict[str, Any]],
    logs: list[str],
    target_id: str,
    round_index: int,
    forbidden_smiles: set[str] | None = None,
) -> list[str]:
    screened: list[str] = []
    local_seen: set[str] = set()
    forbidden = forbidden_smiles or set()
    for raw in generated:
        try:
            smiles = canonicalize_smiles(raw)
        except Exception as exc:
            reason = f"invalid_smiles:{type(exc).__name__}"
            _remember_rejection(rejection_memory, raw, reason, round_index)
            logs.append(json_event("reject", target_id=target_id, round=round_index, smiles=raw, reason=reason))
            continue
        if smiles in seen or smiles in local_seen:
            reason = "duplicate_smiles"
            _remember_rejection(rejection_memory, smiles, reason, round_index)
            logs.append(json_event("reject", target_id=target_id, round=round_index, smiles=smiles, reason=reason))
            continue
        if smiles in forbidden:
            reason = "long_term_memory_exact_reuse"
            _remember_rejection(rejection_memory, smiles, reason, round_index)
            logs.append(json_event("reject", target_id=target_id, round=round_index, smiles=smiles, reason=reason))
            continue
        local_seen.add(smiles)
        screened.append(smiles)
    return screened


def _select_for_route_planning(
    smiles_list: list[str],
    mode: str,
    docking_limit: int,
    logs: list[str],
    target_id: str,
    round_index: int,
) -> list[str]:
    if not smiles_list:
        return []
    scored = sorted(
        ((_preliminary_molecule_score(smiles), smiles) for smiles in smiles_list),
        key=lambda item: item[0],
        reverse=True,
    )
    route_limit = _route_planning_limit(mode, len(scored), docking_limit)
    selected = [smiles for _, smiles in scored[:route_limit]]
    skipped = max(0, len(scored) - len(selected))
    logs.append(
        json_event(
            "route_prefilter",
            target_id=target_id,
            round=round_index,
            generated_valid_count=len(scored),
            selected_count=len(selected),
            skipped_count=skipped,
            route_limit=route_limit,
            selected=selected,
        )
    )
    return selected


def _route_planning_limit(mode: str, valid_count: int, docking_limit: int) -> int:
    configured = os.getenv("AI4S_ROUTE_LIMIT_PER_ROUND") or os.getenv("AGENT_ROUTE_LIMIT_PER_ROUND")
    if configured:
        try:
            limit = int(configured)
        except ValueError as exc:
            raise AgentRunError(f"AI4S_ROUTE_LIMIT_PER_ROUND must be an integer: {configured}") from exc
        if limit <= 0:
            raise AgentRunError("AI4S_ROUTE_LIMIT_PER_ROUND must be positive")
        return min(valid_count, limit)
    if mode == "competition" and _route_engine() == "aizynthfinder":
        return min(valid_count, max(4, max(1, docking_limit) // 2))
    return valid_count


def _preliminary_molecule_score(smiles: str) -> float:
    metrics, _ = property_metrics(smiles)
    binding = proxy_binding_score(metrics, smiles)
    return molecule_score(binding, 1.0, sa_score(metrics))


def _vina_feedback_per_round(mode: str) -> int:
    if mode != "competition":
        return 0
    raw = os.getenv("AI4S_VINA_FEEDBACK_PER_ROUND") or os.getenv("AGENT_VINA_FEEDBACK_PER_ROUND", "1")
    try:
        value = int(raw)
    except ValueError as exc:
        raise AgentRunError(f"AI4S_VINA_FEEDBACK_PER_ROUND must be an integer: {raw}") from exc
    if value < 0:
        raise AgentRunError("AI4S_VINA_FEEDBACK_PER_ROUND must be non-negative")
    return value


def _apply_round_vina_feedback(
    round_candidates: list[Candidate],
    target: TargetContext,
    out_dir: Path,
    mode: str,
    round_index: int,
    logs: list[str],
    rejection_memory: list[dict[str, Any]],
) -> set[str]:
    limit = _vina_feedback_per_round(mode)
    if limit <= 0 or not round_candidates:
        return set()
    failed: set[str] = set()
    ranked = sorted(round_candidates, key=lambda item: item.score.total, reverse=True)
    feedback_candidates = ranked[:limit]
    for feedback_index, candidate in enumerate(feedback_candidates, start=1):
        _progress(
            "vina_feedback_start",
            target=target.target_id,
            round=round_index + 1,
            index=feedback_index,
            total=len(feedback_candidates),
            smiles=candidate.mol_smiles,
        )
        try:
            binding, docking_energy = run_vina_binding_score(
                candidate.mol_smiles,
                target,
                out_dir / "docking_feedback" / f"round_{round_index:03d}" / safe_name(candidate.mol_smiles),
            )
            score = build_score(
                smiles=candidate.mol_smiles,
                route=candidate.route,
                binding_score_value=binding,
                binding_source="vina",
                docking_energy=docking_energy,
            )
        except Exception as exc:
            failed.add(candidate.mol_smiles)
            reason = f"round_vina_feedback_failed:{type(exc).__name__}:{exc}"
            _remember_rejection(rejection_memory, candidate.mol_smiles, reason, round_index)
            _progress(
                "vina_feedback_reject",
                target=target.target_id,
                round=round_index + 1,
                index=feedback_index,
                total=len(feedback_candidates),
                reason=reason,
                smiles=candidate.mol_smiles,
            )
            logs.append(
                json_event(
                    "reject",
                    target_id=target.target_id,
                    round=round_index,
                    smiles=candidate.mol_smiles,
                    reason=reason,
                )
            )
            continue
        if score.route_score <= 0 or score.validity_score == 0 or score.molecule_score <= 0:
            failed.add(candidate.mol_smiles)
            reason = "round_vina_feedback_score_zero"
            _remember_rejection(rejection_memory, candidate.mol_smiles, reason, round_index)
            _progress(
                "vina_feedback_reject",
                target=target.target_id,
                round=round_index + 1,
                index=feedback_index,
                total=len(feedback_candidates),
                reason=reason,
                smiles=candidate.mol_smiles,
            )
            logs.append(
                json_event(
                    "reject",
                    target_id=target.target_id,
                    round=round_index,
                    smiles=candidate.mol_smiles,
                    reason=reason,
                )
            )
            continue
        candidate.score = score
        candidate.metadata["round_vina_feedback"] = True
        _progress(
            "vina_feedback_accept",
            target=target.target_id,
            round=round_index + 1,
            index=feedback_index,
            total=len(feedback_candidates),
            binding=score.binding_score,
            energy=score.docking_energy,
            total_score=score.total,
            smiles=candidate.mol_smiles,
        )
        logs.append(
            json_event(
                "competition_feedback_dock",
                target_id=target.target_id,
                round=round_index,
                smiles=candidate.mol_smiles,
                vina_binding=score.binding_score,
                docking_energy=score.docking_energy,
                total_score=score.total,
                route_score=score.route_score,
                route_validity_score=score.route_validity_score,
                starting_material_availability_score=score.starting_material_availability_score,
                step_penalty_score=score.step_penalty_score,
                convergence_score=score.convergence_score,
                balance_score=score.balance_score,
                property_prior_score=score.property_prior_score,
            )
        )
    return failed


def _round_brief(
    target: TargetContext,
    round_index: int,
    memory: list[dict[str, Any]],
    rejection_memory: list[dict[str, Any]],
    limit: int,
) -> dict[str, Any]:
    heuristic = _strategy_from_memory(memory, rejection_memory)
    if not _llm_enabled():
        return {
            "source": "heuristic",
            "strategy": heuristic,
            "focus": ["RDKit-valid CNS-like molecules", "route-valid simple analogs"],
            "avoid": ["route product mismatch", "element imbalance", "self-reaction"],
            "candidates": [],
        }
    return _llm_round_brief(target, round_index, memory, rejection_memory, limit, heuristic)


def _llm_round_brief(
    target: TargetContext,
    round_index: int,
    memory: list[dict[str, Any]],
    rejection_memory: list[dict[str, Any]],
    limit: int,
    heuristic: str,
) -> dict[str, Any]:
    client = LiteLlmClient()
    agent_context = _agent_context()
    memory_text = _format_memory_for_prompt(memory)
    rejection_text = _format_rejections_for_prompt(rejection_memory)
    _progress("llm_start", target=target.target_id, round=round_index + 1, limit=limit)
    try:
        payload = _complete_json_with_heartbeat(
            client,
            [
                {
                    "role": "system",
                    "content": (
                        "You are the round planner and molecule generator inside an AI4S CNS small-molecule "
                        "design agent. Read the competition requirements, scoring context, target pocket, and "
                        "previous memory. Decide the next search strategy, then propose candidates. The downstream "
                        "code will verify RDKit validity, route final-product match, route element balance, SA, "
                        "and binding score. Return only a JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Competition and scoring context:\n{agent_context}\n\n"
                        f"Target: {target.target_id}\n"
                        f"Pocket: {target.pocket_summary}\n"
                        f"Round: {round_index}\n"
                        f"Heuristic suggestion: {heuristic}\n"
                        f"Previous evaluated candidates:\n{memory_text}\n\n"
                        f"Recent rejected candidates and reasons:\n{rejection_text}\n\n"
                        "Use the context above and freely explore chemically reasonable candidates. "
                        "If previous candidates are provided, improve them by making small, synthesizable changes "
                        "that may raise binding, CNS-likeness, SA, or route score. Prefer molecules that are valid, "
                        "CNS-like, dockable, synthetically reachable, and not fixed-library copies.\n\n"
                        "Return exactly this JSON shape and no markdown:\n"
                        "{"
                        "\"strategy\":\"short strategy for this round\","
                        "\"focus\":[\"1-3 optimization focuses\"],"
                        "\"avoid\":[\"1-3 failure modes to avoid\"],"
                        "\"candidates\":[{\"smiles\":\"...\",\"rationale\":\"brief reason tied to score\"}]"
                        "}\n"
                        f"Return at most {limit} diverse candidates."
                    ),
                },
            ],
            target.target_id,
            round_index + 1,
        )
    except ValueError as exc:
        log_dir = getattr(getattr(client, "settings", None), "log_dir", None)
        _progress("llm_reject", target=target.target_id, round=round_index + 1, reason=type(exc).__name__, log_dir=log_dir)
        detail = f"LLM 返回的 JSON 无法解析：{exc}"
        if log_dir:
            detail += f"；原始响应见 {log_dir}"
        raise AgentRunError(detail) from exc
    except Exception as exc:
        log_dir = getattr(getattr(client, "settings", None), "log_dir", None)
        _progress("llm_reject", target=target.target_id, round=round_index + 1, reason=type(exc).__name__, log_dir=log_dir)
        detail = f"LLM 调用失败：{type(exc).__name__}: {exc}"
        if log_dir:
            detail += f"；审计日志见 {log_dir}"
        raise AgentRunError(detail) from exc
    brief = _normalize_round_brief(payload, limit)
    _progress("llm_done", target=target.target_id, round=round_index + 1, candidates=len(brief["candidates"]))
    return brief


def _complete_json_with_heartbeat(client: LiteLlmClient, messages: list[dict[str, str]], target_id: str, round_number: int) -> Any:
    interval = _llm_heartbeat_seconds()
    if interval <= 0:
        return client.complete_json(messages)

    stop = threading.Event()

    def heartbeat() -> None:
        started = time.monotonic()
        while not stop.wait(interval):
            _progress(
                "llm_wait",
                target=target_id,
                round=round_number,
                elapsed_seconds=round(time.monotonic() - started, 1),
            )

    worker = threading.Thread(target=heartbeat, name="ai4s-llm-heartbeat", daemon=True)
    worker.start()
    try:
        return client.complete_json(messages)
    finally:
        stop.set()
        worker.join(timeout=0.2)


def _llm_heartbeat_seconds() -> float:
    raw = os.getenv("AI4S_LLM_HEARTBEAT_SECONDS", "30").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise AgentRunError(f"AI4S_LLM_HEARTBEAT_SECONDS 必须是数字：{raw}") from exc
    return max(0.0, value)


def _normalize_round_brief(payload: Any, limit: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AgentRunError("LLM round planner must return a JSON object")
    strategy = payload.get("strategy")
    if not isinstance(strategy, str) or not strategy.strip():
        raise AgentRunError("LLM round planner response missing strategy")
    candidates = _extract_llm_candidates(payload, limit)
    if not candidates:
        raise AgentRunError("LLM round planner returned no candidate SMILES")
    return {
        "source": "llm",
        "strategy": strategy.strip(),
        "focus": _string_list(payload.get("focus")),
        "avoid": _string_list(payload.get("avoid")),
        "candidates": candidates,
    }


def _strategy_from_memory(memory: list[dict[str, Any]], rejection_memory: list[dict[str, Any]]) -> str:
    recent_reasons = {str(item.get("reason", "")) for item in _recent_rejections(rejection_memory)}
    if any("route" in reason for reason in recent_reasons):
        return "route_repair: previous candidates failed route planning, so simplify chemistry and prefer obvious purchasable precursors"
    if any("invalid_smiles" in reason for reason in recent_reasons):
        return "validity_repair: previous candidates included invalid SMILES, so use conservative RDKit-valid drug-like scaffolds"
    if not memory:
        return "initial_exploration: generate diverse CNS-like, synthetically reachable molecules before exploiting any scaffold"
    best = memory[0]
    route = float(best["route"])
    molecule = float(best["molecule"])
    binding = best.get("binding")
    penalties = {str(value) for item in memory for value in item.get("penalties", [])}
    if any("route" in penalty for penalty in penalties) or route < 0.75:
        return "route_repair: keep high-scoring motifs but simplify synthesis, avoid route mismatch, imbalance, and hard-to-buy starting materials"
    if binding is not None and float(binding) < 0.35:
        return "binding_improvement: preserve route-valid scaffolds while adding modest aromatic, halogen, heteroaryl, or H-bond features"
    if molecule < 0.7:
        return "druglike_tuning: preserve route-valid scaffolds while improving QED, CNS-like logP, size, SA, and ligand efficiency"
    return "local_exploitation: make small synthesizable analog changes around the best route-valid molecules and keep diversity"


def _memory_limit(default: int = 10) -> int:
    raw = os.getenv("AI4S_AGENT_MEMORY_LIMIT", str(default))
    try:
        limit = int(raw)
    except ValueError as exc:
        raise AgentRunError(f"AI4S_AGENT_MEMORY_LIMIT must be an integer: {raw}") from exc
    if limit <= 0:
        raise AgentRunError("AI4S_AGENT_MEMORY_LIMIT must be positive")
    return limit


def _merge_memory(long_term_memory: list[dict[str, Any]], run_memory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _unique_best_memory([*run_memory, *long_term_memory], _memory_limit())


def _unique_best_memory(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    best_by_smiles: dict[str, dict[str, Any]] = {}
    for item in items:
        smiles = str(item["smiles"])
        current = best_by_smiles.get(smiles)
        if current is None or float(item["total"]) > float(current["total"]):
            best_by_smiles[smiles] = item
    return sorted(best_by_smiles.values(), key=lambda item: float(item["total"]), reverse=True)[:limit]


def _load_long_term_memory(memory_file: Path | None, target: TargetContext) -> list[dict[str, Any]]:
    if memory_file is None or not memory_file.exists():
        return []
    target_signature = _target_signature(target)
    items: list[dict[str, Any]] = []
    for line_number, line in enumerate(memory_file.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AgentRunError(f"invalid agent memory JSON at {memory_file}:{line_number}") from exc
        if record.get("schema_version") != 1 or record.get("event") != "agent_experience":
            raise AgentRunError(f"invalid agent memory record at {memory_file}:{line_number}")
        if record.get("target_signature") != target_signature and record.get("scope") != "global":
            continue
        items.append(_memory_item_from_experience(record, memory_file, line_number))
    return _unique_best_memory(items, _memory_limit())


def _memory_item_from_experience(record: dict[str, Any], memory_file: Path, line_number: int) -> dict[str, Any]:
    candidate = record.get("candidate")
    if not isinstance(candidate, dict):
        raise AgentRunError(f"agent memory record missing candidate at {memory_file}:{line_number}")
    required = ("smiles", "total", "molecule", "route", "route_source")
    missing = [name for name in required if name not in candidate]
    if missing:
        raise AgentRunError(f"agent memory record missing {','.join(missing)} at {memory_file}:{line_number}")
    return {
        "smiles": candidate["smiles"],
        "total": candidate["total"],
        "molecule": candidate["molecule"],
        "route": candidate["route"],
        "binding": candidate.get("binding"),
        "binding_source": candidate.get("binding_source"),
        "docking_energy": candidate.get("docking_energy"),
        "qed": candidate.get("qed"),
        "sa": candidate.get("sa"),
        "route_validity": candidate.get("route_validity"),
        "starting_material": candidate.get("starting_material"),
        "step": candidate.get("step"),
        "convergence": candidate.get("convergence"),
        "balance": candidate.get("balance"),
        "route_source": candidate["route_source"],
        "penalties": candidate.get("penalties", []),
        "round_vina_feedback": bool(candidate.get("round_vina_feedback")),
        "origin": "global_experience" if record.get("scope") == "global" else "long_term_memory",
        "source_run": record.get("run_dir"),
        "target_signature": record.get("target_signature"),
    }


def _experience_candidate_payload(candidate: Candidate) -> dict[str, Any]:
    return {
        "smiles": candidate.mol_smiles,
        "route_text": candidate.route.text,
        "total": candidate.score.total,
        "molecule": candidate.score.molecule_score,
        "route": candidate.score.route_score,
        "binding": candidate.score.binding_score,
        "binding_source": candidate.score.binding_source,
        "docking_energy": candidate.score.docking_energy,
        "qed": candidate.score.qed,
        "sa": candidate.score.sa,
        "route_validity": candidate.score.route_validity_score,
        "starting_material": candidate.score.starting_material_availability_score,
        "step": candidate.score.step_penalty_score,
        "convergence": candidate.score.convergence_score,
        "balance": candidate.score.balance_score,
        "route_source": candidate.route.source,
        "penalties": candidate.score.penalties,
        "round_vina_feedback": bool(candidate.metadata.get("round_vina_feedback")),
    }


def _elite_memory(candidates: list[Candidate], limit: int = 5, origin: str = "run_memory") -> list[dict[str, Any]]:
    ranked = sorted(candidates, key=lambda item: item.score.total, reverse=True)
    memory: list[dict[str, Any]] = []
    for candidate in ranked[:limit]:
        memory.append(
            {
                "smiles": candidate.mol_smiles,
                "total": candidate.score.total,
                "molecule": candidate.score.molecule_score,
                "route": candidate.score.route_score,
                "binding": candidate.score.binding_score,
                "binding_source": candidate.score.binding_source,
                "docking_energy": candidate.score.docking_energy,
                "qed": candidate.score.qed,
                "sa": candidate.score.sa,
                "route_validity": candidate.score.route_validity_score,
                "starting_material": candidate.score.starting_material_availability_score,
                "step": candidate.score.step_penalty_score,
                "convergence": candidate.score.convergence_score,
                "balance": candidate.score.balance_score,
                "route_source": candidate.route.source,
                "penalties": candidate.score.penalties,
                "round_vina_feedback": bool(candidate.metadata.get("round_vina_feedback")),
                "origin": origin,
            }
        )
    return memory


def _format_memory_for_prompt(memory: list[dict[str, Any]]) -> str:
    if not memory:
        return "No previous evaluated candidates yet."
    lines: list[str] = []
    for item in memory:
        penalties = ",".join(str(value) for value in item.get("penalties", [])) or "none"
        lines.append(
            "- "
            f"origin={item.get('origin', 'run_memory')} "
            f"smiles={item['smiles']} "
            f"total={item['total']} "
            f"molecule={item['molecule']} "
            f"route={item['route']} "
            f"binding={item.get('binding')} "
            f"binding_source={item.get('binding_source')} "
            f"docking_energy={item.get('docking_energy')} "
            f"qed={item.get('qed')} "
            f"sa={item.get('sa')} "
            f"route_validity={item.get('route_validity')} "
            f"starting_material={item.get('starting_material')} "
            f"step={item.get('step')} "
            f"convergence={item.get('convergence')} "
            f"balance={item.get('balance')} "
            f"route_source={item['route_source']} "
            f"round_vina_feedback={item.get('round_vina_feedback')} "
            f"penalties={penalties}"
        )
    return "\n".join(lines)


def _remember_rejection(rejection_memory: list[dict[str, Any]], smiles: str, reason: str, round_index: int, limit: int = 12) -> None:
    rejection_memory.append({"smiles": smiles, "reason": reason, "round": round_index})
    del rejection_memory[:-limit]


def _recent_rejections(rejection_memory: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    return rejection_memory[-limit:]


def _format_rejections_for_prompt(rejection_memory: list[dict[str, Any]]) -> str:
    recent = _recent_rejections(rejection_memory)
    if not recent:
        return "No rejected candidates yet."
    lines: list[str] = []
    for item in recent:
        lines.append(f"- round={item['round']} smiles={item['smiles']} reason={item['reason']}")
    return "\n".join(lines)


def _evaluate_candidate(
    smiles: str,
    target: TargetContext,
    out_dir: Path,
    mode: str,
    candidate_index: int,
    docking_limit: int,
) -> Candidate:
    route = _plan_route(smiles, out_dir / "routes" / safe_name(smiles))
    if mode in {"proxy", "competition"}:
        metrics, _ = property_metrics(smiles)
        binding = proxy_binding_score(metrics, smiles)
        docking_energy = None
        binding_source = "proxy" if mode == "proxy" else "proxy_search"
    else:
        if candidate_index >= docking_limit:
            raise AgentRunError("docking_budget_exhausted")
        binding, docking_energy = run_vina_binding_score(smiles, target, out_dir / "docking" / safe_name(smiles))
        binding_source = "vina"
    score = build_score(smiles=smiles, route=route, binding_score_value=binding, binding_source=binding_source, docking_energy=docking_energy)
    if score.route_score <= 0:
        raise AgentRunError("route_score_zero")
    if score.validity_score == 0 or score.molecule_score <= 0:
        raise AgentRunError("molecule_score_zero")
    return Candidate(mol_smiles=smiles, route=route, score=score, metadata={})


def _rerank_with_vina(
    candidates: list[Candidate],
    target: TargetContext,
    out_dir: Path,
    docking_limit: int,
    logs: list[str],
) -> list[Candidate]:
    if docking_limit <= 0:
        raise AgentRunError("competition mode requires AGENT_DOCKING_LIMIT > 0")
    docked: list[Candidate] = []
    ranked = sorted(candidates, key=lambda item: item.score.total, reverse=True)
    rerank_candidates = ranked[:docking_limit]
    for rerank_index, candidate in enumerate(rerank_candidates, start=1):
        _progress(
            "competition_dock_start",
            target=target.target_id,
            index=rerank_index,
            total=len(rerank_candidates),
            smiles=candidate.mol_smiles,
        )
        try:
            reused_vina_score = candidate.score.binding_source == "vina" and candidate.score.docking_energy is not None
            if reused_vina_score:
                score = candidate.score
            else:
                binding, docking_energy = run_vina_binding_score(
                    candidate.mol_smiles,
                    target,
                    out_dir / "docking" / safe_name(candidate.mol_smiles),
                )
                score = build_score(
                    smiles=candidate.mol_smiles,
                    route=candidate.route,
                    binding_score_value=binding,
                    binding_source="vina",
                    docking_energy=docking_energy,
                )
        except Exception as exc:
            reason = f"competition_vina_failed:{type(exc).__name__}:{exc}"
            _progress(
                "competition_dock_reject",
                target=target.target_id,
                index=rerank_index,
                total=len(rerank_candidates),
                reason=reason,
                smiles=candidate.mol_smiles,
            )
            logs.append(
                json_event(
                    "reject",
                    target_id=target.target_id,
                    smiles=candidate.mol_smiles,
                    reason=reason,
                )
            )
            continue
        if score.route_score <= 0 or score.validity_score == 0 or score.molecule_score <= 0:
            _progress(
                "competition_dock_reject",
                target=target.target_id,
                index=rerank_index,
                total=len(rerank_candidates),
                reason="competition_vina_score_zero",
                smiles=candidate.mol_smiles,
            )
            logs.append(json_event("reject", target_id=target.target_id, smiles=candidate.mol_smiles, reason="competition_vina_score_zero"))
            continue
        reranked = candidate.model_copy(deep=True)
        reranked.score = score
        reranked.metadata["competition_reranked"] = True
        docked.append(reranked)
        _progress(
            "competition_dock_accept",
            target=target.target_id,
            index=rerank_index,
            total=len(rerank_candidates),
            binding=score.binding_score,
            energy=score.docking_energy,
            total_score=score.total,
            reused=reused_vina_score,
            smiles=reranked.mol_smiles,
        )
        logs.append(
            json_event(
                "competition_dock",
                target_id=target.target_id,
                smiles=reranked.mol_smiles,
                vina_binding=score.binding_score,
                docking_energy=score.docking_energy,
                total_score=score.total,
                route_score=score.route_score,
                route_validity_score=score.route_validity_score,
                starting_material_availability_score=score.starting_material_availability_score,
                step_penalty_score=score.step_penalty_score,
                convergence_score=score.convergence_score,
                balance_score=score.balance_score,
                property_prior_score=score.property_prior_score,
                reused_vina_score=reused_vina_score,
            )
        )
    if not docked:
        raise AgentRunError("competition mode produced no Vina-scored candidates")
    return docked


def _plan_route(smiles: str, out_dir: Path) -> Route:
    engine = _route_engine()
    if engine == "aizynthfinder":
        return run_aizynthfinder_route(smiles, out_dir)
    raise AgentRunError(f"unknown route engine: {engine}")


def _route_engine() -> str:
    return os.getenv("AI4S_ROUTE_ENGINE", "aizynthfinder").strip().lower()


def _llm_enabled() -> bool:
    return os.getenv("CHEM_EVOLVE_LLM_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


def _extract_llm_candidates(payload: dict[str, Any], limit: int) -> list[dict[str, str]]:
    items = payload.get("candidates")
    if not isinstance(items, list):
        raise AgentRunError("LLM round planner response missing candidates list")
    candidates: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, str):
            smiles = item.strip()
            rationale = ""
        elif isinstance(item, dict):
            smiles_value = item.get("smiles")
            rationale_value = item.get("rationale", "")
            if not isinstance(smiles_value, str):
                raise AgentRunError("LLM candidate object missing smiles string")
            if not isinstance(rationale_value, str):
                raise AgentRunError("LLM candidate rationale must be a string")
            smiles = smiles_value.strip()
            rationale = rationale_value.strip()
        else:
            raise AgentRunError("LLM candidates must be strings or objects")
        if not smiles:
            raise AgentRunError("LLM candidate SMILES is empty")
        candidate = {"smiles": smiles}
        if rationale:
            candidate["rationale"] = rationale
        candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return candidates


def _brief_candidate_smiles(candidates: list[dict[str, str]]) -> list[str]:
    return [candidate["smiles"] for candidate in candidates]


def _candidate_rationales(candidates: list[dict[str, str]], limit: int = 5) -> list[dict[str, str]]:
    notes: list[dict[str, str]] = []
    for candidate in candidates[:limit]:
        notes.append({"smiles": candidate["smiles"], "rationale": candidate.get("rationale", "")})
    return notes


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise AgentRunError("LLM round planner focus/avoid fields must be lists")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise AgentRunError("LLM round planner focus/avoid entries must be non-empty strings")
        out.append(item.strip())
    return out[:3]


def _agent_context_paths() -> list[str]:
    raw = os.getenv(
        "AI4S_COMPETITION_CONTEXT_PATHS",
        "docs/competition_race5_description.md,docs/sota_tools_and_scoring.md,data/README.md,data/benchmarks/benchmark_prior.json",
    )
    return [item.strip() for item in raw.split(",") if item.strip()]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _agent_context() -> str:
    chunks: list[str] = []
    missing: list[str] = []
    empty: list[str] = []
    for item in _agent_context_paths():
        path = _resolve_repo_path(item)
        if not path.exists():
            missing.append(item)
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            chunks.append(f"## {item}\n{_trim_context(text)}")
        else:
            empty.append(item)
    if missing:
        raise AgentRunError(f"competition context file missing: {', '.join(missing)}")
    if empty:
        raise AgentRunError(f"competition context file is empty: {', '.join(empty)}")
    if not chunks:
        raise AgentRunError("no competition context files found for LLM agent")
    return "\n\n".join(chunks)


def _trim_context(text: str, limit: int = 9000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
