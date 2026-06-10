from __future__ import annotations

from pathlib import Path
from typing import Sequence

from chem_evolve_agent.agent.executor import ToolExecutor
from chem_evolve_agent.agent.judge import ChemistryJudge
from chem_evolve_agent.agent.memory import AgentMemory
from chem_evolve_agent.agent.planner import AgentPlanner, LlmPlanner
from chem_evolve_agent.agent.state import AgentBudget, AgentState
from chem_evolve_agent.evolution.retrospective import summarize_candidates
from chem_evolve_agent.logging_utils import json_event
from chem_evolve_agent.models import Candidate
from chem_evolve_agent.randomness import run_seed_from_env


def run_chemistry_agent(
    target_id: str,
    pocket_summary: str,
    out_dir: Path,
    rounds: int,
    per_round: int,
    mode: str = "proxy",
    docking_limit: int = 8,
    target_path: Path | None = None,
    pocket_center: Sequence[float] | None = None,
    pocket_box_size: Sequence[float] | None = None,
    planner: AgentPlanner | None = None,
    run_seed: int | None = None,
) -> tuple[list[Candidate], list[str]]:
    run_seed = run_seed_from_env() if run_seed is None else run_seed
    budget = AgentBudget(rounds=rounds, per_round=per_round, mode=mode, docking_limit=docking_limit)
    state = AgentState(target_id=target_id, pocket_summary=pocket_summary)
    planner = planner or LlmPlanner()
    executor = ToolExecutor(run_seed=run_seed)
    judge = ChemistryJudge()
    memory = AgentMemory()
    logs = [
        json_event(
            "agent_run_start",
            target_id=target_id,
            planner=planner.name,
            rounds=rounds,
            per_round=per_round,
            mode=mode,
            run_seed=run_seed,
        )
    ]

    for _ in range(rounds):
        # Agent 主循环：记忆检索 -> planner 选动作 -> executor 调工具 -> judge 验证评分 -> memory 写回。
        # 这个结构对应比赛要求的“agent 生成/改进/演化过程”，不是一次性固定 pipeline。
        memory_context = memory.retrieve(state)
        decision = planner.plan_next(state, budget, memory_context)
        action = decision.action
        logs.append(
            json_event(
                "agent_plan",
                target_id=target_id,
                round=state.round_index,
                action_id=action.action_id,
                action_type=action.action_type,
                planner=decision.planner_name,
                fallback_reason=decision.fallback_reason,
                available_actions=[
                    {
                        "action_type": available.action_type,
                        "enabled": available.enabled,
                        "reason": available.reason,
                    }
                    for available in decision.available_actions
                ],
                memory_used=decision.memory_used,
                rationale=action.rationale,
            )
        )
        execution = executor.execute(action, state)
        if execution.notes and action.action_type == "generate_llm":
            logs.append(json_event("llm_skipped", target_id=target_id, round=state.round_index, reason="disabled_or_failed"))
        logs.append(
            json_event(
                "generate",
                target_id=target_id,
                round=state.round_index,
                count=len(execution.generated_smiles),
                generator=execution.tool_name,
                action_id=action.action_id,
            )
        )
        observation, eval_logs = judge.evaluate_generation(execution, state)
        logs.extend(eval_logs)
        if hasattr(observation, "model_dump"):
            state.history.append(observation.model_dump())
        else:
            state.history.append(observation.dict())

        # memory 只保存本次 run 内的有效经验，用来影响后续 round 的动作选择。
        # 跨 run 的长期经验由 outputs/strategy_memory 和 auto_iterate 负责。
        memory.observe(state, decision, observation)
        logs.append(
            json_event(
                "agent_observe",
                target_id=target_id,
                round=state.round_index,
                action_id=observation.action_id,
                generated_count=observation.generated_count,
                accepted_count=observation.accepted_count,
                skipped_invalid=observation.skipped_invalid,
                skipped_duplicate=observation.skipped_duplicate,
                notes=observation.notes,
                best_score=state.best_score(),
            )
        )
        logs.append(
            json_event(
                "agent_memory",
                target_id=target_id,
                round=state.round_index,
                entries=len(memory.entries),
                latest=memory.entries[-1].text if memory.entries else "",
            )
        )
        logs.append(json_event("reflect", target_id=target_id, round=state.round_index, summary=summarize_candidates(state.candidates)))
        state.round_index += 1

    candidates = state.ranked_candidates()
    if mode in {"docking", "competition"} and target_path is not None and pocket_center is not None and pocket_box_size is not None:
        dock_dir = out_dir / "docking" / target_id
        logs.extend(
            judge.apply_docking(
                candidates,
                target_id=target_id,
                target_path=target_path,
                dock_dir=dock_dir,
                center=tuple(pocket_center),
                box_size=tuple(pocket_box_size),
                limit=docking_limit,
            )
        )
        candidates = state.ranked_candidates()

    logs.append(
        json_event(
            "agent_rank",
            target_id=target_id,
            candidate_count=len(candidates),
            best_score=candidates[0].score.total if candidates else 0.0,
        )
    )
    return candidates, logs
