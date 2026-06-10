# Chem-Evolve Agent Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and debug an autonomous small-molecule design and synthesis-planning agent for the AI4S CNS challenge that can generate valid `result.csv`/`result.log`/`result.zip` outputs from one or more PDB targets.

**Architecture:** Use a hybrid architecture: LangGraph-style deterministic workflow for reliability, AIDE/MLEvolve-style experiment evolution for autonomous improvement, and chemistry tools for molecule generation, docking, property filtering, and retrosynthesis. The first milestone uses lightweight local scoring and mockable tool adapters so the full loop can be tested before expensive docking or retrosynthesis are integrated.

**Tech Stack:** Python 3.10+ preferred, RDKit, AutoDock Vina or GNINA, AiZynthFinder or compatible retrosynthesis wrapper, pydantic, pandas, pytest, optional LangGraph, optional LLM API.

---

## 1. Research Summary

### 1.1 Relevant Open-Source Agent Architectures

- **AIDE / aideml:** Best reference for code-space exploration. Each experiment branch is a code/config node; the agent generates changes, runs experiments, evaluates results, and expands promising branches.
- **MLEvolve:** Best reference for autonomous scientific iteration. Its useful ideas are Progressive MCGS, retrospective memory, cross-branch information flow, and separation between strategy planning and implementation.
- **ChemCrow:** Relevant as a chemistry tool-calling reference, but not enough as the main competition agent because it is more of a conversational chemistry assistant than an autonomous optimization loop.
- **LangGraph:** Useful for the deterministic execution graph: target preparation, molecule generation, filtering, docking, retrosynthesis, ranking, logging, and submission.
- **AutoGen/OpenHands:** Useful references for multi-agent execution and coding-agent behavior, but they should not define the chemistry loop.

### 1.2 Chemistry Tool Stack

- **RDKit:** Must-have for SMILES validation, descriptors, QED, substructure filters, molecular mutations, sanitization, and canonicalization.
- **AutoDock Vina:** Lightweight docking evaluator. Good default for first competition-grade prototype.
- **GNINA:** Stronger neural docking/rescoring option if GPU/runtime allows.
- **AiZynthFinder:** Practical retrosynthesis route planner. Good default for route generation and route scoring.

### 1.3 Core Design Conclusion

Do not directly fork ChemCrow as the final system. Use ChemCrow only as a reference for chemistry tool wrappers and prompt style. The main agent should be:

```text
ExperimentEvolutionController  # AIDE/MLEvolve ideas
  -> ChemWorkflowGraph         # LangGraph-style deterministic DAG
     -> TargetProfiler
     -> MoleculeGeneratorPool
     -> PropertyFilter
     -> DockingEvaluator
     -> RetrosynthesisPlanner
     -> CompositeScorer
     -> MemoryBank
     -> Submitter
```

## 2. Competition Objective

### 2.1 Inputs

- Preliminary round: one target file, usually `target.pdb`.
- Final round: `/saisdata/37/target1.pdb`, `/saisdata/37/target2.pdb`, `/saisdata/37/target3.pdb`.

### 2.2 Outputs

- Preliminary round: `result.zip` containing `result.csv` and `result.log`.
- Final round: `/saisresult/result.zip` containing `result1.csv`, `result2.csv`, and `result3.csv`.

Each CSV must contain:

```csv
mol_smiles,route
```

The final product of the last reaction step in `route` must match `mol_smiles` after canonicalization.

### 2.3 Ranking Objective

For final round scoring, optimize both molecule and route:

```text
composite_score = 0.60 * molecule_score + 0.40 * route_score
```

Molecule score should include docking energy, property quality, structural reasonableness, and filter penalties. Route score should include route existence, step count, starting material availability, reaction confidence, route economy, and final-product consistency.

## 3. Repository Layout

Create this structure:

```text
chem_evolve_agent/
  __init__.py
  cli.py
  config.py
  logging_utils.py
  models.py
  target/
    __init__.py
    pdb_loader.py
    pocket.py
  chemistry/
    __init__.py
    smiles.py
    filters.py
    mutations.py
    scoring.py
  generators/
    __init__.py
    base.py
    seed_generator.py
    mutation_generator.py
    llm_generator.py
  evaluators/
    __init__.py
    local_proxy.py
    docking.py
    retrosynthesis.py
  evolution/
    __init__.py
    memory.py
    graph_search.py
    retrospective.py
  workflow/
    __init__.py
    runner.py
  submitter.py
configs/
  default.yaml
  competition_final.yaml
tests/
  test_smiles.py
  test_filters.py
  test_scoring.py
  test_submitter.py
  test_memory.py
  test_runner_smoke.py
scripts/
  run_local_smoke.sh
  inspect_result_zip.py
```

## 4. Milestone 0: Runnable Skeleton and Output Contract

Purpose: make the agent produce valid files before adding expensive chemistry tools.

### Task 0.1: Create project metadata

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`

- [ ] **Step 1: Add minimal Python project config**

```toml
[project]
name = "chem-evolve-agent"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
  "pydantic>=2",
  "pandas>=2",
  "pyyaml>=6",
]

[project.optional-dependencies]
dev = [
  "pytest>=8",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Add README with the local command**

```markdown
# Chem-Evolve Agent

Local smoke run:

```bash
python -m chem_evolve_agent.cli --targets examples/target.pdb --out runs/smoke
```
```

- [ ] **Step 3: Verify packaging smoke**

Run:

```bash
python3 -m pytest --version
```

Expected: prints pytest version if installed, or fails clearly so the environment setup step can install it.

### Task 0.2: Define shared models

**Files:**
- Create: `chem_evolve_agent/models.py`
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Write tests for candidate and score ordering**

```python
from chem_evolve_agent.models import Candidate, Route, Score

def test_score_total_prefers_better_composite():
    weak = Score(molecule_score=0.4, route_score=0.2)
    strong = Score(molecule_score=0.7, route_score=0.5)
    assert strong.total > weak.total

def test_candidate_requires_smiles_and_route():
    candidate = Candidate(
        mol_smiles="CCO",
        route=Route(steps=["CCBr.O>>CCO"]),
        score=Score(molecule_score=0.5, route_score=0.5),
        metadata={"source": "test"},
    )
    assert candidate.mol_smiles == "CCO"
    assert candidate.route.steps[-1].endswith(">>CCO")
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python3 -m pytest tests/test_scoring.py -v
```

Expected: FAIL because `chem_evolve_agent.models` does not exist.

- [ ] **Step 3: Implement models**

```python
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class Route(BaseModel):
    steps: list[str] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return ",".join(self.steps)


class Score(BaseModel):
    molecule_score: float
    route_score: float
    docking_energy: float | None = None
    qed: float | None = None
    sa: float | None = None
    penalties: list[str] = Field(default_factory=list)

    @property
    def total(self) -> float:
        return 0.60 * self.molecule_score + 0.40 * self.route_score


class Candidate(BaseModel):
    mol_smiles: str
    route: Route
    score: Score
    metadata: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Run tests and confirm pass**

Run:

```bash
python3 -m pytest tests/test_scoring.py -v
```

Expected: PASS.

### Task 0.3: Implement submitter

**Files:**
- Create: `chem_evolve_agent/submitter.py`
- Test: `tests/test_submitter.py`

- [ ] **Step 1: Write tests for CSV and zip contract**

```python
import zipfile
from pathlib import Path

import pandas as pd

from chem_evolve_agent.models import Candidate, Route, Score
from chem_evolve_agent.submitter import write_single_target_result


def test_write_single_target_result_creates_csv_log_zip(tmp_path: Path):
    candidate = Candidate(
        mol_smiles="CCO",
        route=Route(steps=["CCBr.O>>CCO"]),
        score=Score(molecule_score=0.8, route_score=0.7),
        metadata={"source": "unit"},
    )
    zip_path = write_single_target_result(tmp_path, "result", [candidate], ["event one"])
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        assert sorted(zf.namelist()) == ["result.csv", "result.log"]
        zf.extractall(tmp_path / "unzipped")
    df = pd.read_csv(tmp_path / "unzipped" / "result.csv")
    assert list(df.columns) == ["mol_smiles", "route"]
    assert df.iloc[0]["mol_smiles"] == "CCO"
    assert (tmp_path / "unzipped" / "result.log").read_text().strip() == "event one"
```

- [ ] **Step 2: Implement submitter**

```python
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from chem_evolve_agent.models import Candidate


def write_single_target_result(
    out_dir: Path,
    stem: str,
    candidates: list[Candidate],
    log_lines: list[str],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    log_path = out_dir / f"{stem}.log"
    zip_path = out_dir / f"{stem}.zip"

    if not candidates:
        raise ValueError("at least one candidate is required")

    rows = [
        {"mol_smiles": candidate.mol_smiles, "route": candidate.route.text}
        for candidate in sorted(candidates, key=lambda item: item.score.total, reverse=True)
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log_path.write_text("\n".join(log_lines) + "\n")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=csv_path.name)
        zf.write(log_path, arcname=log_path.name)
    return zip_path
```

- [ ] **Step 3: Run submitter test**

Run:

```bash
python3 -m pytest tests/test_submitter.py -v
```

Expected: PASS.

## 5. Milestone 1: Chemistry Baseline Without External Docking

Purpose: make the agent chemically sane before integrating slow tools.

### Task 1.1: Add SMILES utilities

**Files:**
- Create: `chem_evolve_agent/chemistry/smiles.py`
- Test: `tests/test_smiles.py`

- [ ] **Step 1: Write tests**

```python
from chem_evolve_agent.chemistry.smiles import canonicalize_smiles, is_valid_smiles


def test_canonicalize_smiles_normalizes_valid_smiles():
    assert canonicalize_smiles("C(C)O") in {"CCO", "OCC"}


def test_invalid_smiles_returns_false():
    assert is_valid_smiles("not-a-smiles") is False
```

- [ ] **Step 2: Implement RDKit-backed functions with fallback**

```python
from __future__ import annotations


def _rdkit():
    try:
        from rdkit import Chem
    except Exception:
        return None
    return Chem


def is_valid_smiles(smiles: str) -> bool:
    Chem = _rdkit()
    if Chem is None:
        return bool(smiles) and " " not in smiles and ">>" not in smiles
    return Chem.MolFromSmiles(smiles) is not None


def canonicalize_smiles(smiles: str) -> str:
    Chem = _rdkit()
    if Chem is None:
        if not is_valid_smiles(smiles):
            raise ValueError(f"invalid SMILES: {smiles}")
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)
```

- [ ] **Step 3: Run tests**

Run:

```bash
python3 -m pytest tests/test_smiles.py -v
```

Expected: PASS in RDKit and fallback environments. If RDKit canonicalizes ethanol to `CCO`, the first test passes.

### Task 1.2: Add local proxy evaluator

**Files:**
- Create: `chem_evolve_agent/evaluators/local_proxy.py`
- Test: `tests/test_filters.py`

- [ ] **Step 1: Write tests for deterministic scoring**

```python
from chem_evolve_agent.evaluators.local_proxy import score_smiles_proxy


def test_proxy_score_is_deterministic():
    first = score_smiles_proxy("CCO")
    second = score_smiles_proxy("CCO")
    assert first == second


def test_proxy_penalizes_invalid_smiles():
    score = score_smiles_proxy("not-a-smiles")
    assert score.molecule_score == 0
    assert "invalid_smiles" in score.penalties
```

- [ ] **Step 2: Implement deterministic local score**

```python
from __future__ import annotations

from chem_evolve_agent.chemistry.smiles import is_valid_smiles
from chem_evolve_agent.models import Score


def score_smiles_proxy(smiles: str) -> Score:
    if not is_valid_smiles(smiles):
        return Score(molecule_score=0.0, route_score=0.0, penalties=["invalid_smiles"])
    heavy_atom_proxy = sum(1 for char in smiles if char.isalpha() and char.isupper())
    size_score = min(heavy_atom_proxy / 35.0, 1.0)
    simplicity_bonus = max(0.0, 1.0 - max(0, len(smiles) - 80) / 80.0)
    molecule_score = round(0.65 * size_score + 0.35 * simplicity_bonus, 4)
    return Score(molecule_score=molecule_score, route_score=0.0)
```

- [ ] **Step 3: Run tests**

Run:

```bash
python3 -m pytest tests/test_filters.py -v
```

Expected: PASS.

## 6. Milestone 2: Generator Pool and Smoke Workflow

Purpose: produce candidates through an agent loop instead of static copying.

### Task 2.1: Implement seed generator

**Files:**
- Create: `chem_evolve_agent/generators/base.py`
- Create: `chem_evolve_agent/generators/seed_generator.py`

- [ ] **Step 1: Define generator interface**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationContext:
    target_id: str
    pocket_summary: str
    round_index: int


class MoleculeGenerator:
    name = "base"

    def generate(self, context: GenerationContext, limit: int) -> list[str]:
        raise NotImplementedError
```

- [ ] **Step 2: Add non-static seed generator**

```python
from __future__ import annotations

import random

from chem_evolve_agent.generators.base import GenerationContext, MoleculeGenerator


class SeedGenerator(MoleculeGenerator):
    name = "seed_generator"

    def __init__(self, seed: int = 13):
        self._random = random.Random(seed)
        self._cores = ["c1ccccc1", "c1ccncc1", "O=C(N)Nc1ccccc1", "Cc1ncccc1"]
        self._tails = ["C", "OC", "Cl", "F", "N", "C(=O)N"]

    def generate(self, context: GenerationContext, limit: int) -> list[str]:
        molecules: list[str] = []
        for _ in range(limit):
            core = self._random.choice(self._cores)
            tail = self._random.choice(self._tails)
            molecules.append(f"{tail}{core}")
        return molecules
```

### Task 2.2: Implement workflow runner

**Files:**
- Create: `chem_evolve_agent/workflow/runner.py`
- Test: `tests/test_runner_smoke.py`

- [ ] **Step 1: Write smoke test**

```python
from chem_evolve_agent.workflow.runner import run_target_smoke


def test_run_target_smoke_returns_ranked_candidates():
    candidates, logs = run_target_smoke(target_id="target", rounds=2, per_round=4)
    assert candidates
    assert candidates[0].score.total >= candidates[-1].score.total
    assert any("round=1" in line for line in logs)
```

- [ ] **Step 2: Implement workflow**

```python
from __future__ import annotations

import json

from chem_evolve_agent.evaluators.local_proxy import score_smiles_proxy
from chem_evolve_agent.generators.base import GenerationContext
from chem_evolve_agent.generators.seed_generator import SeedGenerator
from chem_evolve_agent.models import Candidate, Route


def _naive_route_for(smiles: str) -> Route:
    return Route(steps=[f"START.O>>{smiles}"])


def run_target_smoke(
    target_id: str,
    rounds: int = 3,
    per_round: int = 16,
) -> tuple[list[Candidate], list[str]]:
    generator = SeedGenerator(seed=17)
    candidates: list[Candidate] = []
    logs: list[str] = []
    for round_index in range(rounds):
        context = GenerationContext(
            target_id=target_id,
            pocket_summary="smoke_mode_no_pocket",
            round_index=round_index,
        )
        generated = generator.generate(context, limit=per_round)
        logs.append(json.dumps({
            "event": "generate",
            "target_id": target_id,
            "round": round_index,
            "count": len(generated),
        }, ensure_ascii=False))
        for smiles in generated:
            score = score_smiles_proxy(smiles)
            route = _naive_route_for(smiles)
            score.route_score = 0.2
            candidates.append(Candidate(
                mol_smiles=smiles,
                route=route,
                score=score,
                metadata={"generator": generator.name, "round": round_index},
            ))
    candidates.sort(key=lambda item: item.score.total, reverse=True)
    return candidates, logs
```

- [ ] **Step 3: Run smoke test**

Run:

```bash
python3 -m pytest tests/test_runner_smoke.py -v
```

Expected: PASS.

### Task 2.3: Add CLI

**Files:**
- Create: `chem_evolve_agent/cli.py`
- Create: `scripts/run_local_smoke.sh`

- [ ] **Step 1: Implement CLI**

```python
from __future__ import annotations

import argparse
from pathlib import Path

from chem_evolve_agent.submitter import write_single_target_result
from chem_evolve_agent.workflow.runner import run_target_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--per-round", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.out)
    all_logs: list[str] = []
    for index, target in enumerate(args.targets, start=1):
        stem = "result" if len(args.targets) == 1 else f"result{index}"
        candidates, logs = run_target_smoke(
            target_id=Path(target).stem,
            rounds=args.rounds,
            per_round=args.per_round,
        )
        all_logs.extend(logs)
        write_single_target_result(out_dir, stem, candidates[:20], all_logs)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add shell smoke command**

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p examples runs/smoke
if [ ! -f examples/target.pdb ]; then
  printf 'HEADER    MOCK TARGET\nEND\n' > examples/target.pdb
fi
python3 -m chem_evolve_agent.cli --targets examples/target.pdb --out runs/smoke --rounds 2 --per-round 8
python3 scripts/inspect_result_zip.py runs/smoke/result.zip
```

Make it executable:

```bash
chmod +x scripts/run_local_smoke.sh
```

## 7. Milestone 3: Real Chemistry Adapters

Purpose: replace proxy scoring with real chemistry tools while keeping the same interfaces.

### Task 3.1: RDKit filter adapter

**Files:**
- Create: `chem_evolve_agent/chemistry/filters.py`

Implementation requirements:

- Validate SMILES with RDKit.
- Compute molecular weight, logP, HBD, HBA, rotatable bonds, QED.
- Reject or penalize invalid molecules, extreme MW, extreme logP, PAINS-like alerts if available, and molecules that fail sanitization.
- Return both numeric metrics and named penalties.

Minimum acceptance command:

```bash
python3 -m pytest tests/test_smiles.py tests/test_filters.py -v
```

Expected: PASS.

### Task 3.2: Docking adapter

**Files:**
- Create: `chem_evolve_agent/evaluators/docking.py`
- Create: `chem_evolve_agent/target/pdb_loader.py`
- Create: `chem_evolve_agent/target/pocket.py`

Implementation requirements:

- Prepare ligand from SMILES to 3D conformer.
- Prepare receptor from PDB.
- Select docking box from known ligand if present; otherwise use pocket predictor or conservative center/box fallback.
- Run Vina first. Add GNINA later if runtime allows.
- Capture command, stdout, stderr, docking energy, output path, elapsed time.
- Never crash the full loop on one docking failure; return a scored failure with penalty.

Minimum acceptance command:

```bash
python3 -m chem_evolve_agent.cli --targets examples/target.pdb --out runs/docking-smoke --rounds 1 --per-round 2
```

Expected: CLI exits 0 and writes a log event for each docking attempt or skipped docking reason.

### Task 3.3: Retrosynthesis adapter

**Files:**
- Create: `chem_evolve_agent/evaluators/retrosynthesis.py`

Implementation requirements:

- Wrap AiZynthFinder if installed.
- Use a deterministic fallback route only in smoke mode.
- Parse route steps into competition format: `reactants>>product`, comma-delimited for multi-step route.
- Verify the final product canonicalizes to `mol_smiles`.
- Score route by existence, step count, starting material availability, reaction confidence if available, and route consistency.

Minimum acceptance command:

```bash
python3 -m pytest tests/test_submitter.py tests/test_runner_smoke.py -v
```

Expected: PASS.

## 8. Milestone 4: Evolution Controller

Purpose: make the system improve instead of just running a fixed pipeline.

### Task 4.1: Memory bank

**Files:**
- Create: `chem_evolve_agent/evolution/memory.py`
- Test: `tests/test_memory.py`

Data model:

```python
class MemoryRecord(BaseModel):
    lesson_id: str
    context: dict[str, Any]
    rule: str
    evidence: dict[str, Any]
    confidence: float
    applies_when: list[str]
    fails_when: list[str]
```

Memory promotion rule:

```text
Promote an observation to retrospective memory only if it appears in at least two branches or improves composite score by at least 10% within one branch.
```

### Task 4.2: Progressive graph search

**Files:**
- Create: `chem_evolve_agent/evolution/graph_search.py`

Search node:

```text
node = {
  "strategy": generator + filter + docking + retro config,
  "parent_ids": list[str],
  "candidate_ids": list[str],
  "mean_score": float,
  "best_score": float,
  "cost_seconds": float,
  "status": "running|complete|failed"
}
```

Expansion policy:

- First 30% budget: diversity exploration.
- Middle 50% budget: mixed exploration and exploitation with UCB.
- Final 20% budget: exploit best branch and route feasibility.

### Task 4.3: Retrospective planner

**Files:**
- Create: `chem_evolve_agent/evolution/retrospective.py`

Planner behavior:

- Summarize top and failed molecules by structural motifs and route failures.
- Convert repeated findings into memory records.
- Use memory records to bias the next generator configuration.
- Keep every suggestion grounded in observed score deltas.

## 9. Milestone 5: Competition Mode

Purpose: run in the official container path and produce exactly the expected output.

### Task 5.1: Final-round runner

**Files:**
- Modify: `chem_evolve_agent/cli.py`
- Create: `scripts/run_competition_final.sh`

Script:

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p /saisresult
python3 -m chem_evolve_agent.cli \
  --targets /saisdata/37/target1.pdb /saisdata/37/target2.pdb /saisdata/37/target3.pdb \
  --out /saisresult \
  --rounds "${AGENT_ROUNDS:-8}" \
  --per-round "${AGENT_PER_ROUND:-64}"
cd /saisresult
zip -f result.zip result1.csv result2.csv result3.csv || zip result.zip result1.csv result2.csv result3.csv
```

Acceptance:

```bash
bash scripts/run_competition_final.sh
python3 scripts/inspect_result_zip.py /saisresult/result.zip
```

Expected:

- `/saisresult/result.zip` exists.
- Zip contains `result1.csv`, `result2.csv`, `result3.csv`.
- Each CSV has `mol_smiles,route`.
- Every route final product matches the CSV SMILES after canonicalization, unless external retrosynthesis failed and the log explicitly records the fallback reason.

## 10. Self-Debugging Protocol

Use this protocol for "go mode" execution.

### 10.1 Before each run

Log:

```json
{"event":"run_start","targets":["target1.pdb"],"rounds":8,"per_round":64,"mode":"proxy|docking|competition"}
```

### 10.2 On failure

Do not patch randomly. Follow this root-cause sequence:

1. Reproduce the exact failing command.
2. Capture full traceback or stderr.
3. Identify the failing layer: input, generation, validation, docking, retrosynthesis, scoring, submitter.
4. Write a minimal failing test or a one-command reproduction.
5. Patch only that layer.
6. Re-run the minimal test.
7. Re-run the smoke command.

### 10.3 Required verification commands

After each milestone:

```bash
python3 -m pytest -v
bash scripts/run_local_smoke.sh
python3 scripts/inspect_result_zip.py runs/smoke/result.zip
```

Before final submission:

```bash
bash scripts/run_competition_final.sh
python3 scripts/inspect_result_zip.py /saisresult/result.zip
```

## 11. Experiment Schedule

### Day 1: Output contract and smoke loop

- Implement Milestone 0 and Milestone 1.
- Goal: local smoke run writes `result.zip`.
- Do not add LLM calls yet.

### Day 2: Generator pool

- Add seed, mutation, fragment, and optional LLM generators.
- Goal: each generated molecule has provenance and valid log entries.

### Day 3: Real property filtering and docking

- Add RDKit filters and Vina adapter.
- Goal: docking result is captured per molecule without crashing the loop.

### Day 4: Retrosynthesis

- Add AiZynthFinder wrapper.
- Goal: route score affects ranking, and final route product is verified.

### Day 5: Evolution and memory

- Add graph search, branch selection, retrospective memory, and strategy biasing.
- Goal: log shows at least two generation/evaluation/reflection cycles.

### Day 6: Official path hardening

- Add final-round runner and strict zip inspection.
- Goal: `/saisresult/result.zip` is generated from `/saisdata/37/*.pdb`.

### Day 7: Ablation and score tuning

- Compare proxy-only, RDKit+docking, RDKit+docking+retro, and full evolution modes.
- Goal: keep the best mode under time budget.

## 12. Risk Register

- **RDKit unavailable:** Use fallback for smoke tests, but install RDKit before serious molecule scoring.
- **Docking too slow:** Use two-stage ranking: proxy/RDKit top 100, docking top 20, retrosynthesis top 10.
- **No pocket predictor:** Use conservative docking box fallback and log it.
- **Retrosynthesis fails often:** Penalize but do not discard all high-affinity molecules; keep a route-feasibility balance.
- **LLM hallucinated chemistry:** LLM suggestions must pass RDKit validation and route-product verification before entering the candidate pool.
- **Static-result suspicion:** Log parent molecule, generator, mutation reason, score delta, route planner result, and branch id for every selected molecule.

## 13. Final Acceptance Checklist

- [ ] `python3 -m pytest -v` exits 0.
- [ ] `bash scripts/run_local_smoke.sh` exits 0.
- [ ] Local `result.zip` contains CSV and log files.
- [ ] Final competition script writes `/saisresult/result.zip`.
- [ ] Every CSV has exactly `mol_smiles,route` columns.
- [ ] Logs prove generation, scoring, retrosynthesis, ranking, and at least one reflective iteration.
- [ ] No final molecule is copied from the sample.
- [ ] Route final products match corresponding `mol_smiles` after canonicalization.
