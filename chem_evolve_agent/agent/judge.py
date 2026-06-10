from __future__ import annotations

from pathlib import Path
from typing import Sequence

from chem_evolve_agent.agent.actions import AgentObservation, ActionExecution
from chem_evolve_agent.agent.state import AgentState
from chem_evolve_agent.chemistry.smiles import canonicalize_smiles, is_valid_smiles
from chem_evolve_agent.evaluators.docking import dock_smiles_with_vina
from chem_evolve_agent.evaluators.local_proxy import score_smiles_proxy
from chem_evolve_agent.evaluators.retrosynthesis import fallback_route_for, route_consistency_score
from chem_evolve_agent.logging_utils import json_event
from chem_evolve_agent.models import Candidate


class ChemistryJudge:
    def evaluate_generation(self, execution: ActionExecution, state: AgentState) -> tuple[AgentObservation, list[str]]:
        action = execution.action
        logs: list[str] = []
        accepted = 0
        skipped_invalid = 0
        skipped_duplicate = 0

        for raw_smiles in execution.generated_smiles:
            # 所有 generator/工具产物都必须先过 RDKit 合法性和 canonical 化，
            # 这样后续去重、route 终产物比较、评分才在同一个 SMILES 表示上进行。
            if not is_valid_smiles(raw_smiles):
                skipped_invalid += 1
                logs.append(
                    json_event(
                        "skip_invalid_smiles",
                        target_id=state.target_id,
                        round=action.round_index,
                        smiles=raw_smiles,
                    )
                )
                continue

            smiles = canonicalize_smiles(raw_smiles)
            if smiles in state.seen_smiles:
                skipped_duplicate += 1
                logs.append(
                    json_event(
                        "skip_duplicate_smiles",
                        target_id=state.target_id,
                        round=action.round_index,
                        smiles=smiles,
                    )
                )
                continue

            state.seen_smiles.add(smiles)
            score = score_smiles_proxy(smiles)
            route = fallback_route_for(smiles)
            route_score, route_penalties = route_consistency_score(smiles, route)

            # 当前 route 仍是 fallback/轻量检查；它至少要保证最终产物和 mol_smiles 一致。
            # 比赛路线分更严格，后续接 AiZynthFinder 或反应模板时应替换这里。
            score.route_score = route_score
            score.penalties.extend(route_penalties)
            logs.append(
                json_event(
                    "evaluate",
                    target_id=state.target_id,
                    round=action.round_index,
                    smiles=smiles,
                    molecule_score=score.molecule_score,
                    route_score=score.route_score,
                    penalties=score.penalties,
                )
            )
            logs.append(
                json_event(
                    "retrosynthesis",
                    target_id=state.target_id,
                    smiles=smiles,
                    planner="fallback",
                    route=route.text,
                    route_score=route_score,
                    penalties=route_penalties,
                )
            )
            state.candidates.append(
                Candidate(
                    mol_smiles=smiles,
                    route=route,
                    score=score,
                    metadata={
                        "generator": execution.tool_name,
                        "round": action.round_index,
                        "action_id": action.action_id,
                        "action_type": action.action_type,
                    },
                )
            )
            accepted += 1

        observation = AgentObservation(
            action_id=action.action_id,
            action_type=action.action_type,
            tool_name=execution.tool_name,
            generated_count=len(execution.generated_smiles),
            accepted_count=accepted,
            skipped_invalid=skipped_invalid,
            skipped_duplicate=skipped_duplicate,
            notes=execution.notes,
        )
        return observation, logs

    def apply_docking(
        self,
        candidates: Sequence[Candidate],
        target_id: str,
        target_path: Path,
        dock_dir: Path,
        center: tuple[float, float, float] | list[float],
        box_size: tuple[float, float, float] | list[float],
        limit: int,
    ) -> list[str]:
        logs: list[str] = []
        for candidate in list(candidates)[:limit]:
            result = dock_smiles_with_vina(
                candidate.mol_smiles,
                target_path,
                dock_dir,
                center=center,
                box_size=box_size,
            )
            candidate.score.penalties.extend(result.penalties)
            if result.docking_energy is not None:
                candidate.score.docking_energy = result.docking_energy
                docking_component = max(0.0, min(1.0, (-result.docking_energy - 4.0) / 8.0))
                candidate.score.molecule_score = round(0.75 * candidate.score.molecule_score + 0.25 * docking_component, 4)
            elif result.penalties:
                candidate.score.molecule_score = round(max(0.0, candidate.score.molecule_score - 0.03), 4)
            logs.append(
                json_event(
                    "docking",
                    target_id=target_id,
                    smiles=candidate.mol_smiles,
                    attempted=result.attempted,
                    success=result.success,
                    docking_energy=result.docking_energy,
                    reason=result.reason,
                    penalties=result.penalties,
                    elapsed_seconds=round(result.elapsed_seconds, 3),
                )
            )
        return logs
