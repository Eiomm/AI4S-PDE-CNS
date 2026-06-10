from __future__ import annotations

from pathlib import Path

from chem_evolve_agent.agent import run_chemistry_agent
from chem_evolve_agent.evaluators.local_proxy import score_smiles_proxy
from chem_evolve_agent.evaluators.retrosynthesis import fallback_route_for, route_consistency_score
from chem_evolve_agent.evaluators.docking import dock_smiles_with_vina
from chem_evolve_agent.evolution.retrospective import summarize_candidates
from chem_evolve_agent.generators.base import GenerationContext
from chem_evolve_agent.generators.fragment_generator import FragmentCatalogGenerator
from chem_evolve_agent.generators.llm_generator import LlmGenerator
from chem_evolve_agent.generators.mutation_generator import MutationGenerator
from chem_evolve_agent.generators.seed_generator import SeedGenerator
from chem_evolve_agent.logging_utils import json_event
from chem_evolve_agent.models import Candidate
from chem_evolve_agent.randomness import offset_seed, run_seed_from_env
from chem_evolve_agent.chemistry.smiles import canonicalize_smiles, is_valid_smiles
from chem_evolve_agent.target.pdb_loader import load_pdb_target
from chem_evolve_agent.target.pocket import summarize_pocket


def run_target_smoke(
    target_id: str,
    rounds: int = 3,
    per_round: int = 16,
) -> tuple[list[Candidate], list[str]]:
    return _run_generation_loop(
        target_id=target_id,
        pocket_summary="smoke_mode_no_pocket",
        rounds=rounds,
        per_round=per_round,
        mode="proxy",
    )


def run_target_pipeline(
    target_path: Path,
    out_dir: Path,
    rounds: int = 3,
    per_round: int = 16,
    mode: str = "proxy",
    docking_limit: int = 8,
) -> tuple[list[Candidate], list[str]]:
    target = load_pdb_target(target_path)
    pocket = summarize_pocket(target)
    logs = [
        json_event(
            "target_loaded",
            target_id=target.target_id,
            path=str(target.path),
            atoms=target.atom_count,
            residues=len(target.residues),
            has_hetatm=target.has_hetatm,
        ),
        json_event(
            "pocket_summary",
            target_id=target.target_id,
            method=pocket.method,
            center=pocket.center,
            box_size=pocket.box_size,
            summary=pocket.summary,
        ),
    ]
    candidates, loop_logs = _run_generation_loop(
        target_id=target.target_id,
        pocket_summary=pocket.summary,
        rounds=rounds,
        per_round=per_round,
        mode=mode,
        run_seed=run_seed_from_env(),
    )
    logs.extend(loop_logs)
    if mode in {"docking", "competition"}:
        dock_dir = out_dir / "docking" / target.target_id
        for candidate in candidates[:docking_limit]:
            result = dock_smiles_with_vina(
                candidate.mol_smiles,
                target.path,
                dock_dir,
                center=pocket.center,
                box_size=pocket.box_size,
            )
            candidate.score.penalties.extend(result.penalties)
            if result.docking_energy is not None:
                candidate.score.docking_energy = result.docking_energy
                # Map better negative energies into a bounded contribution.
                docking_component = max(0.0, min(1.0, (-result.docking_energy - 4.0) / 8.0))
                candidate.score.molecule_score = round(0.75 * candidate.score.molecule_score + 0.25 * docking_component, 4)
            elif result.penalties:
                candidate.score.molecule_score = round(max(0.0, candidate.score.molecule_score - 0.03), 4)
            logs.append(
                json_event(
                    "docking",
                    target_id=target.target_id,
                    smiles=candidate.mol_smiles,
                    attempted=result.attempted,
                    success=result.success,
                    docking_energy=result.docking_energy,
                    reason=result.reason,
                    penalties=result.penalties,
                    elapsed_seconds=round(result.elapsed_seconds, 3),
                )
            )
        candidates.sort(key=lambda item: item.score.total, reverse=True)
    logs.append(
        json_event(
            "pipeline_complete",
            target_id=target.target_id,
            mode=mode,
            candidate_count=len(candidates),
            best_score=candidates[0].score.total if candidates else 0.0,
        )
    )
    return candidates, logs


def run_target_agent_pipeline(
    target_path: Path,
    out_dir: Path,
    rounds: int = 3,
    per_round: int = 16,
    mode: str = "proxy",
    docking_limit: int = 8,
) -> tuple[list[Candidate], list[str]]:
    target = load_pdb_target(target_path)
    pocket = summarize_pocket(target)
    logs = [
        json_event(
            "target_loaded",
            target_id=target.target_id,
            path=str(target.path),
            atoms=target.atom_count,
            residues=len(target.residues),
            has_hetatm=target.has_hetatm,
        ),
        json_event(
            "pocket_summary",
            target_id=target.target_id,
            method=pocket.method,
            center=pocket.center,
            box_size=pocket.box_size,
            summary=pocket.summary,
        ),
    ]
    candidates, agent_logs = run_chemistry_agent(
        target_id=target.target_id,
        pocket_summary=pocket.summary,
        out_dir=out_dir,
        rounds=rounds,
        per_round=per_round,
        mode=mode,
        docking_limit=docking_limit,
        target_path=target.path,
        pocket_center=pocket.center,
        pocket_box_size=pocket.box_size,
        run_seed=run_seed_from_env(),
    )
    logs.extend(agent_logs)
    logs.append(
        json_event(
            "pipeline_complete",
            target_id=target.target_id,
            mode=mode,
            runner="agent",
            candidate_count=len(candidates),
            best_score=candidates[0].score.total if candidates else 0.0,
        )
    )
    return candidates, logs


def _run_generation_loop(
    target_id: str,
    pocket_summary: str,
    rounds: int,
    per_round: int,
    mode: str,
    run_seed: int = 0,
) -> tuple[list[Candidate], list[str]]:
    # legacy runner 仍保留为实验基线：按固定 generator 顺序生成、过滤、评分、排序。
    # run_seed 让多次实验可以探索不同采样路径，同时保留可复现性。
    generators = [
        SeedGenerator(seed=offset_seed(run_seed, 17)),
        MutationGenerator(seed=offset_seed(run_seed, 31)),
        FragmentCatalogGenerator(seed=offset_seed(run_seed, 43)),
        LlmGenerator(),
    ]
    candidates: list[Candidate] = []
    logs: list[str] = []
    seen_smiles: set[str] = set()
    logs.append(json_event("run_start", targets=[target_id], rounds=rounds, per_round=per_round, mode=mode, run_seed=run_seed))
    for round_index in range(rounds):
        generator = generators[round_index % len(generators)]
        context = GenerationContext(
            target_id=target_id,
            pocket_summary=pocket_summary,
            round_index=round_index,
            run_seed=run_seed,
        )
        generated = generator.generate(context, limit=per_round)
        if not generated and generator.name == "llm_generator":
            logs.append(json_event("llm_skipped", target_id=target_id, round=round_index, reason="disabled_or_failed"))
            generator = generators[0]
            generated = generator.generate(context, limit=per_round)
        logs.append(json_event("generate", target_id=target_id, round=round_index, count=len(generated), generator=generator.name))
        for raw_smiles in generated:
            if not is_valid_smiles(raw_smiles):
                logs.append(json_event("skip_invalid_smiles", target_id=target_id, round=round_index, smiles=raw_smiles))
                continue
            smiles = canonicalize_smiles(raw_smiles)
            if smiles in seen_smiles:
                logs.append(json_event("skip_duplicate_smiles", target_id=target_id, round=round_index, smiles=smiles))
                continue
            seen_smiles.add(smiles)
            score = score_smiles_proxy(smiles)
            route = fallback_route_for(smiles)
            route_score, route_penalties = route_consistency_score(smiles, route)
            score.route_score = route_score
            score.penalties.extend(route_penalties)
            logs.append(
                json_event(
                    "evaluate",
                    target_id=target_id,
                    round=round_index,
                    smiles=smiles,
                    molecule_score=score.molecule_score,
                    route_score=score.route_score,
                    penalties=score.penalties,
                )
            )
            logs.append(
                json_event(
                    "retrosynthesis",
                    target_id=target_id,
                    smiles=smiles,
                    planner="fallback",
                    route=route.text,
                    route_score=route_score,
                    penalties=route_penalties,
                )
            )
            candidates.append(
                Candidate(
                    mol_smiles=smiles,
                    route=route,
                    score=score,
                    metadata={"generator": generator.name, "round": round_index},
                )
            )
        logs.append(json_event("reflect", target_id=target_id, round=round_index, summary=summarize_candidates(candidates)))
    candidates.sort(key=lambda item: item.score.total, reverse=True)
    logs.append(json_event("rank", target_id=target_id, candidate_count=len(candidates), best_score=candidates[0].score.total if candidates else 0.0))
    return candidates, logs
