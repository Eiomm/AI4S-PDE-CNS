---
name: ai4s-chem-evolve
description: Use when improving the simplified AI4S CNS molecule-generation agent, preserving the result.zip contract while avoiding silent fallback paths.
---

# AI4S Chem Evolve

Use this skill when improving `/data/wangjunao/AI4S` for the AI4S CNS small-molecule generation task.

## Operating Rules

1. Preserve the submission contract: `result.csv`, `result.log`, `result.zip` with `mol_smiles,route` columns.
2. Keep one main runtime path: `cli.py -> core.py -> chem_ops.py/runtime_tools.py -> submitter.py`.
3. Do not add silent fallback behavior. If a required tool is unavailable for the selected mode, fail clearly.
4. Do not add `A>>A` self-reaction routes. Invalid routes must reject candidates.
5. Optimize against the competition-shaped score:
   - molecule = `0.8 binding + 0.1 validity + 0.1 SA`
   - route = `0.55 validity + 0.30 starting material + 0.05 step + 0.05 convergence + 0.05 balance`
   - total = `0.60 molecule + 0.40 route`
6. Prefer real tools:
   - SBDD external generator via `AI4S_SBDD_GENERATOR_CMD`
   - Vina/OpenBabel for docking mode
   - AiZynthFinder when route parsing is enabled
   - RDKit for SMILES/property/scoring support only; do not reintroduce a template route engine

## Commands

```bash
bash scripts/run_harness_once.sh --name smoke --target examples/target.pdb --rounds 1 --per-round 8 --top-k 5 --mode proxy --skip-tests
python scripts/check_tools.py
python scripts/check_llm_connectivity.py
```
