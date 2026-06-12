# AI4S CNS Agent Submission

## Entry Point

Run inside the competition container. The Docker image places the official launcher at `/app/run.sh`:

```bash
bash /app/run.sh
```

Default paths:

- Input targets: `/saisdata/target1.pdb`, `/saisdata/target2.pdb`, `/saisdata/target3.pdb`
- Output archive: `/saisresult/result.zip`

For local compatibility, the entry point also accepts the older nested layout `/saisdata/37/target1-3.pdb` when the direct `/saisdata/target1-3.pdb` files are not present.

The archive contains only `result1.csv`, `result2.csv`, and `result3.csv`.
At startup, the entry point removes previous agent-managed `result*.csv`, `result*.log`, `result*.zip`, plus scratch directories including `generation/`, `routes/`, `docking/`, `docking_feedback/`, `llm_io/`, and `work/`, so the submission is generated fresh inside the container.
Before exit, `Code/main.py` validates `/saisresult/result.zip` and requires exactly `result1.csv`, `result2.csv`, and `result3.csv`.
For multi-target runs, per-target scratch files are isolated under `work/resultN/` while CSV/LOG/ZIP submission artifacts stay at the output root.

## Agent Flow

The simplified agent has one runtime path:

```text
target.pdb
  -> pocket summary
  -> competition/scoring context
  -> generation tools
  -> strict route planning
  -> proxy search scoring
  -> long-term experience memory + elite memory + molecule evolution
  -> final Vina reranking in competition mode
  -> result zip
```

Tools do not silently degrade. If `AGENT_MODE=docking` or `competition`, Vina/OpenBabel must work. In `competition` mode, proxy scoring is used only during search; final candidates are reranked with Vina. If route planning cannot produce a route whose final product matches `mol_smiles`, that candidate is rejected.
When no `.env` is present, `competition` mode defaults to `AI4S_ROUTE_ENGINE=aizynthfinder` and `AIZYNTHFINDER_CONFIG=data/aizynthfinder/config.yml`; explicit environment variables still take precedence.

Each round logs the evaluated candidates. When LLM reasoning is enabled, the agent reads the competition/scoring context, target pocket, persistent experience memory, and current-run elite memory, then asks the LLM to produce the round `agent_strategy`, focus items, avoid items, and candidate molecules. The agent also uses top scored candidates as memory for RDKit-guided structural evolution.
The log records `agent_strategy` with its source, so reviewers can audit whether a round used API reasoning or explicit local heuristic mode.
The log also starts with `agent_runtime_config`, recording non-secret runtime configuration such as LLM model, LLM audit log directory, route engine, route budget, docking budget, and output cleanup. By default, LLM IO logs are kept under the current output directory's `llm_io/`.
`--per-round` is a global round candidate-pool size. Before expensive route planning, candidates are ranked by a light RDKit/proxy molecule score; in competition mode the Docker default `AI4S_ROUTE_LIMIT_PER_ROUND=10` limits how many candidates enter AiZynthFinder each round.
In competition mode, `AI4S_VINA_FEEDBACK_PER_ROUND=1` docks the best candidate from each round with real Vina and feeds that score back into the next round's memory.
For multi-target semifinal runs, only the final `result.zip` archive is created; per-target LOG files remain on disk for audit, receive a final `final_submit` event after the zip is assembled, and are not packed into the final submission zip.

## Main Dependencies

- Python 3.10+
- RDKit
- pydantic
- Optional LLM API via LiteLLM
- AutoDock Vina and OpenBabel for docking/competition mode

## Runtime Options

- `AGENT_ROUNDS`, Docker default `8`, must be positive
- `AGENT_PER_ROUND`, Docker default `32`, must be positive
- `AGENT_TOP_K`, default `10`, must be positive
- `AGENT_MODE`, default `competition`
- `AGENT_DOCKING_LIMIT`, default equals `AGENT_TOP_K`; must be positive in `docking` and `competition` modes
- `CHEM_EVOLVE_LLM_ENABLED`
- `AI4S_AGENT_MEMORY_FILE`, default is configured by the Docker image as `/workspace/data/agent_experience.jsonl`
- `AI4S_SBDD_GENERATOR_CMD`
- `AI4S_ROUTE_ENGINE`, default `aizynthfinder`; obsolete values such as `template` fail during preflight
- `AI4S_ROUTE_LIMIT_PER_ROUND`, Docker default `10` for competition entrypoints
- `AI4S_VINA_FEEDBACK_PER_ROUND`, default `1` for competition entrypoints

API key placeholder for optional LLM molecule generation:

```bash
export OPENAI_API_KEY="<your-api-key>"
export AI4S_AGENT_BASE_URL="<optional-openai-compatible-base-url>"
```

## Verification

Before packaging, run:

```bash
bash scripts/check_competition_ready.sh
python scripts/check_tools.py
python scripts/check_data.py
python scripts/check_llm_connectivity.py
bash scripts/run_real_competition_smoke.sh
```

The one-step readiness check runs data/review-artifact checks, tool checks, LLM connectivity, and a fast submission smoke. Set `AI4S_REQUIRE_SBDD=1` to require the external SBDD probe. The real competition smoke uses AiZynthFinder for routes and Vina for final reranking on a minimal local target.
