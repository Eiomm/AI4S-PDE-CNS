# Chem-Evolve Agent

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate ai4s-chem-evolve
```

Local smoke run:

```bash
python -m chem_evolve_agent.cli --targets examples/target.pdb --out runs/smoke
```

Official target minimal harness:

```bash
bash scripts/run_harness_once.sh --name official --target target.pdb
```

This keeps the competition files (`result.csv`, `result.log`, `result.zip`) and also writes easier local aliases:

- `candidates.csv`
- `pipeline.log`
- `submission.zip`

Run the same loop in tmux:

```bash
bash scripts/run_tmux_training.sh official
bash scripts/run_tmux_training.sh official --status
```

Automatic strategy iteration:

```bash
python scripts/auto_iterate.py --experiment goal_smoke --iterations 1 --strategies seed_baseline llm_diverse --skip-tests
```

Agent-loop runner:

```bash
python -m chem_evolve_agent.cli --targets examples/target.pdb --out runs/agent-smoke --runner agent --rounds 2 --per-round 8
bash scripts/run_harness_once.sh --name agent_smoke --target target.pdb --runner agent --skip-tests
```

The default runner remains `legacy`. The `agent` runner uses a planner/action/memory loop: the planner selects one enabled action per round, the executor runs the chemistry tool or generator, the judge scores and filters molecules, and memory feeds the next planner decision. If LLM planning is disabled or fails, the agent falls back to a deterministic heuristic planner.

Competition final run:

```bash
bash scripts/run_competition_final.sh
```

Final-round container contract:

```text
input:  /saisdata/37/target1.pdb /saisdata/37/target2.pdb /saisdata/37/target3.pdb
output: /saisresult/result.zip containing result1.csv, result2.csv, result3.csv
score:  molecule:route = 6:4 in the local composite ranking
```

Local final-mode dry run without `/saisresult` permissions:

```bash
SAISRESULT_DIR=runs/final AGENT_ROUNDS=1 AGENT_PER_ROUND=4 bash scripts/run_competition_final.sh
python scripts/inspect_result_zip.py runs/final/result.zip result1.csv result2.csv result3.csv
```

## LiteLLM API Layer

The agent uses a small `LiteLlmClient` wrapper around LiteLLM's Python SDK. The design follows the AI4S-PDE-CNS Task1 pattern: model/base URL/key settings come from `.env`, calls use an OpenAI-compatible endpoint, provider noise is suppressed by default, and every request/response can be audited as JSONL under `runs/llm_io`.

LLM generation is disabled by default. Enable it only when API credentials are configured:

```bash
cp .env.example .env
export CHEM_EVOLVE_LLM_ENABLED=1
export AI4S_AGENT_MODEL=openai/claude-opus-4-8
export AI4S_AGENT_PROVIDER=openai
export AI4S_AGENT_BASE_URL=https://api.gpt.ge/v1
export APIFOX_GPT_GE_API_KEY=...
```

Relevant knobs:

- `AI4S_AGENT_MODEL`: LiteLLM model string, for example `openai/claude-opus-4-8` for the Apifox/GPT.GE OpenAI-compatible route.
- `AI4S_AGENT_PROVIDER`: set to `openai` for third-party OpenAI-compatible APIs; use `anthropic` only for direct Anthropic official keys.
- `AI4S_AGENT_BASE_URL`: OpenAI-compatible base URL, for example `https://api.gpt.ge/v1` or `http://127.0.0.1:8080/v1` when using a local logging proxy.
- `AI4S_PROXY_PORT`, `AI4S_PROXY_TARGET`, `AI4S_PROXY_LOG_DIR`: local logging proxy settings kept compatible with the referenced AI4S-PDE-CNS Task1 `.env.example`.
- `AI4S_AGENT_API_KEY_ENVS`: comma-separated API key fallback order. Defaults to `APIFOX_GPT_GE_API_KEY,OPENAI_API_KEY,ANTHROPIC_API_KEY,VAPI_API_KEY,AIGC_API_KEY,...`.
- `AI4S_AGENT_LLM_LOG_DIR`: JSONL audit directory. Defaults to `runs/llm_io`.
- `AI4S_PROVIDER_RAW_DEBUG`: set to `1` only when debugging provider payloads.
- `CHEM_EVOLVE_LLM_*`: project-specific overrides are still supported and take precedence for model/base URL/temperature/max tokens/retries/log dir.

The LLM generator asks for a JSON array of SMILES, strips fenced `json` responses, then validates and canonicalizes every molecule before it can enter the candidate pool. If LiteLLM is missing, disabled, or the call fails, the deterministic seed generator continues the run.

## Docs

- `docs/task_overview.md`: simple Chinese explanation of the task and current pipeline.
- `docs/competition_race5_description.md`: local summary of the official Race 5 task description, scoring, semifinal contract, and code-review requirements.
- `docs/harness_prompt.md`: reusable Codex harness prompt for iterative generate/test/review/fix loops.
- `docs/auto_iteration.md`: automatic `/goal`-style experiment loop with terminal dashboard and skill memory.
