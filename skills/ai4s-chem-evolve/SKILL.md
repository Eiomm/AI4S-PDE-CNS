---
name: ai4s-chem-evolve
description: Use when iterating the AI4S chemistry molecule-generation pipeline, running inference experiments, comparing strategies, logging results, and reusing the best strategy memory.
---

# AI4S Chem Evolve

Use this skill when improving `/data/wangjunao/AI4S` for the AI4S small-molecule generation task.

## Operating Loop

1. Keep the submission contract intact: `result.csv`, `result.log`, `result.zip` with `mol_smiles,route` columns.
2. Prefer inference-time strategy experiments first: prompt changes, generator mix, filtering, scoring, docking probes, and retrosynthesis checks.
3. Change one main variable per run, then execute `scripts/run_harness_once.sh` or `scripts/auto_iterate.py`.
4. Judge progress by objective, best score, average score, valid SMILES, route consistency, scaffold diversity, and penalties.
5. Record every run under `outputs/strategy_memory`; promote only strategies that improve objective or reveal a reusable failure rule.
6. If LLM is enabled, use the OpenAI-compatible Apifox/GPT.GE route from `.env`: model `openai/claude-opus-4-8`, provider `openai`, base URL `https://api.gpt.ge/v1`.

## Current Best Strategies

1. `agent_main` objective=0.8431 best=0.7383 avg=0.6975 scaffolds=10 llm_calls=5 runner=agent
   Run: `/data/wangjunao/AI4S/outputs/goal_agent_01_i01_agent_main/20260611_001443`
2. `agent_main` objective=0.731 best=0.6995 avg=0.694 scaffolds=1 llm_calls=4 runner=agent
   Run: `/data/wangjunao/AI4S/outputs/goal_agent_02_i01_agent_main/20260611_002357`
3. `agent_main` objective=0.731 best=0.6995 avg=0.694 scaffolds=1 llm_calls=4 runner=agent
   Run: `/data/wangjunao/AI4S/outputs/goal_agent_02_i02_agent_main/20260611_002446`
4. `agent_main` objective=0.731 best=0.6995 avg=0.694 scaffolds=1 llm_calls=4 runner=agent
   Run: `/data/wangjunao/AI4S/outputs/goal_agent_02_i03_agent_main/20260611_002522`
5. `agent_main` objective=0.731 best=0.6995 avg=0.694 scaffolds=1 llm_calls=4 runner=agent
   Run: `/data/wangjunao/AI4S/outputs/goal_agent_02_i04_agent_main/20260611_002551`

## Commands

```bash
python scripts/auto_iterate.py --experiment goal_iter --iterations 1 --skip-tests
bash scripts/run_harness_once.sh --name official --target target.pdb
python scripts/check_llm_connectivity.py
```

## Memory Files

- `outputs/strategy_memory/experiment_index.csv`: compact experiment table.
- `outputs/strategy_memory/experiment_index.jsonl`: full machine-readable records.
- `outputs/strategy_memory/best_strategies.md`: promoted strategy notes.
- `outputs/strategy_memory/failed_strategies.md`: reusable failure notes.
