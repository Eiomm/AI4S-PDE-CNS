from __future__ import annotations

from chem_evolve_agent.agent.actions import ActionExecution, AgentAction
from chem_evolve_agent.agent.state import AgentState
from chem_evolve_agent.chemistry.mutations import rdkit_guided_mutations
from chem_evolve_agent.generators.base import GenerationContext, MoleculeGenerator
from chem_evolve_agent.generators.fragment_generator import FragmentCatalogGenerator
from chem_evolve_agent.generators.llm_generator import LlmGenerator
from chem_evolve_agent.generators.mutation_generator import MutationGenerator
from chem_evolve_agent.generators.seed_generator import SeedGenerator
from chem_evolve_agent.randomness import offset_seed, run_seed_from_env


class ToolExecutor:
    def __init__(self, run_seed: int | None = None) -> None:
        # run_seed 由 auto_iterate 每轮注入；同一实验可复现，不同 iteration 不会完全重放。
        self.run_seed = run_seed_from_env() if run_seed is None else run_seed
        self._fallback_generator = SeedGenerator(seed=offset_seed(self.run_seed, 17))
        self._generators: dict[str, MoleculeGenerator] = {
            "generate_seed": self._fallback_generator,
            "generate_mutation": MutationGenerator(seed=offset_seed(self.run_seed, 31)),
            "generate_fragment": FragmentCatalogGenerator(seed=offset_seed(self.run_seed, 43)),
            "generate_llm": LlmGenerator(),
        }

    def execute(self, action: AgentAction, state: AgentState) -> ActionExecution:
        if action.action_type == "generate_guided_mutation":
            return self._execute_guided_mutation(action, state)

        generator = self._generators[action.action_type]
        context = GenerationContext(
            target_id=state.target_id,
            pocket_summary=state.pocket_summary,
            round_index=action.round_index,
            run_seed=self.run_seed,
        )
        generated = generator.generate(context, limit=action.limit)
        notes: list[str] = []
        tool_name = generator.name
        if not generated and action.action_type == "generate_llm":
            notes.append("llm_generator_empty_or_disabled")
            generator = self._fallback_generator
            generated = generator.generate(context, limit=action.limit)
            tool_name = generator.name
        return ActionExecution(
            action=action,
            tool_name=tool_name,
            generated_smiles=generated,
            notes=notes,
        )

    def _execute_guided_mutation(self, action: AgentAction, state: AgentState) -> ActionExecution:
        ranked = state.ranked_candidates()
        if not ranked:
            # guided mutation 需要 parent；没有候选时退回 seed，保证 agent 不会空跑。
            generated = self._fallback_generator.generate(
                GenerationContext(
                    target_id=state.target_id,
                    pocket_summary=state.pocket_summary,
                    round_index=action.round_index,
                    run_seed=self.run_seed,
                ),
                limit=action.limit,
            )
            return ActionExecution(
                action=action,
                tool_name="seed_generator",
                generated_smiles=generated,
                notes=["guided_mutation_no_candidates_fallback_seed"],
            )

        requested_parent = str(action.params.get("parent_smiles") or "")
        notes: list[str] = []
        if requested_parent and requested_parent in state.seen_smiles:
            parent_smiles = requested_parent
        else:
            # LLM planner 可以建议 parent，但只能使用本 run 已见过的 SMILES。
            # 未知 parent 会被忽略，防止 planner 幻觉出一个不存在的优化起点。
            parent_smiles = ranked[0].mol_smiles
            if requested_parent:
                notes.append(f"ignored_unknown_parent={requested_parent}")
        notes.append(f"parent_smiles={parent_smiles}")
        generated = rdkit_guided_mutations(
            parent_smiles,
            seed=offset_seed(self.run_seed, action.round_index + 101),
            limit=action.limit,
        )
        return ActionExecution(
            action=action,
            tool_name="guided_mutation_tool",
            generated_smiles=generated,
            notes=notes,
        )
