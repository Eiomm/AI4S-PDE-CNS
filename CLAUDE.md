# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AI4S CNS Challenge Race 7 — neural operator PDE Agent for 1D Burgers equation prediction. The framework wraps an LLM-driven observe-plan-act-record loop with sandboxed tool execution, submission validation, and auditable JSONL logging.

Full competition rules and data layout are in `AGENT.md`. Read it first if you haven't.

## Essential commands

```powershell
# Preferred local experiment Python
$PY = "D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe"

# Run tests
& $PY -m pytest -q

# Run a single test file
& $PY -m pytest tests/test_submission.py -q

# Mock agent smoke test
& $PY -m agent.run --task task1 --config configs\task1_mock.yaml

# Real agent run
& $PY -m agent.run --task task1 --config configs\task1.yaml

# Validate a submission directory
& $PY -m agent.validate_submission --path runs\<run_id>

# Pack a validated submission
& $PY -m agent.pack_submission --run runs\<run_id>
# Default output: runs\<run_id>\pred.zip

# Check LLM connectivity
& $PY -m agent.check_llm --config configs\deepseek.example.yaml

# Zero-train baseline (copies initial conditions, no model)
& $PY -m agent.zero_submission --input data\data_and_sample_submission\train_val_test_init\task1_test.hdf5 --output-dir runs\task1-zero --code-dir code

# Validate initial condition match (first 10 frames)
& $PY -c "from agent.submission import validate_initial_condition; validate_initial_condition(r'runs\...\task1_pred.hdf5', r'data\...\task1_test.hdf5'); print('ok')"

# Evaluate Task 1 predictions
& $PY code\evaluate_task1.py --prediction runs\...\task1_pred.hdf5 --target data\...\task1_val.hdf5 --output runs\...\metrics.json
```

## Architecture

```
agent/          Agent framework — main loop, LLM client, tools, submission, logging
code/           Competition submission code — NOT a package (no __init__.py)
configs/        YAML configs for tasks and LLM provider profiles
data/           Competition HDF5 data (not in git)
runs/           Experiment outputs — each run has manifest.json + artifacts
tests/          pytest test suite
```

New autonomous studies should use the classified run layout instead of writing many top-level run folders:

```text
runs/task1/autonomous/YYYYMMDD/<study_name>/
runs/task2/autonomous/YYYYMMDD/<study_name>/
runs/experiment_registry.jsonl
```

Older top-level `runs/<name>` folders are historical artifacts and should not be moved during normal development.

### Local Python environment

Prefer the existing `Hwpytorch` conda environment for local PDE experiments:

```text
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe
```

It is the current working environment for PyTorch/HDF5/checkpoint inference and has been verified to run Python 3.10.18. `configs/task1.yaml` already points `python_executable` to this path. When running tests, validation, official checkpoint inference, final submission packaging, or any script that imports `torch`, prefer this interpreter over bare `python` to avoid dependency drift.

### Data directory as Agent context

`data/` is not only for HDF5 datasets and checkpoints. Treat it as the immutable input context that the Agent reads before planning. The official sample logs show the Agent first loads the race description, sample submission, sample code, HDF5 structure, task log rules, and PDE knowledge/solver API notes before writing any solution code.

Expected context categories:

- `data/赛题要求.md`: consolidated competition rules and local interpretation.
- `data/task_log_sample/`: official-style `task1_logs.log`, `task2_logs.log`, README, and `openai-log/` proxy reference for compliant LLM logging.
- `data/data_and_sample_submission/`: official competition bundle root. It currently contains `train_val_test_init/` for task HDF5 files and `sample_submission/sample_submission/` for the official sample submission files.
- `data/Task1/` and `data/Task2/`: task-specific train/val/test HDF5 files.
- `data/pdebench_burgers/`: raw PDEBench Burgers data used for allowed training or fine-tuning.
- `data/knowledge_base/` (planned): PDE and model knowledge docs such as `knowledge-api.md`, `solve-api.md`, Burgers equation notes, FNO/DeepONet/PI-DeepONet summaries, PDEBench data format, checkpoint protocol, scoring proxy, and rule-risk notes.
- `data/registry/` (planned): machine-readable path/config indexes such as `data_registry.yaml`, `checkpoint_registry.yaml`, `submission_registry.yaml`, and `knowledge_registry.yaml`.

The Agent should not guess important paths or rules from code. It should read registries and knowledge files first, then inspect HDF5 files. Keep operational run settings in `configs/`; keep official/static input context in `data/`.

The sample logs also imply a knowledge service interface may exist, e.g. endpoints like `/equations/burgers_equation/solvers` and `/ai-models/fno/profile`. If no service is running, provide equivalent Markdown knowledge files under `data/knowledge_base/` so the Observer and Planner still have stable context.

### Agent loop (`agent/run.py`)

The core is `run_agent()`: **observe** → **plan** (LLM call via `logged_completion`) → **act** (tool dispatch) → **record** (JSONL log + manifest). Tools are restricted by path whitelist and command whitelist.

### Target Agent architecture: PDE-Research-Agent

The long-term direction is a PDE-specific research Agent, inspired by ML-Master/AIDE but not copied from it. ML-Master's useful ideas are MCTS-style solution-tree search, separate draft/improve/debug branches, memory from prior experiments, and automatic promotion of the best candidate. Its Kaggle CSV assumptions should not be carried over directly because this competition requires HDF5 predictions, `pred.zip`, `methodology.pdf`, task logs, first-10-frame checks, and code-log consistency.

The Agent is not a chat wrapper around scripts. It should behave like a controlled scientific experiment system: it observes the current PDE state, proposes one auditable experiment, calls a whitelisted tool, evaluates the result deterministically, records the evidence, and decides whether to expand, debug, stop, or submit.

The intended closed loop is:

```
Rule Guard -> Observer -> Memory -> Planner -> Tool Router -> Controlled Executor -> Evaluator -> Reviewer -> Memory/Submitter
```

- `Rule Guard`: encode competition constraints, no numerical solvers for extra data, Task 1/Task 2 isolation, official checkpoint rules, and time budgets.
- `Observer`: inspect HDF5 data, checkpoints, predictions, validation errors, segment metrics, runtime, and failure modes before planning.
- `Memory`: summarize experiment history, best candidates, failed branches, code hashes, validation artifacts, and platform feedback.
- `Planner`: use the LLM to propose exactly one atomic experiment action as structured JSON.
- `Tool Router`: map the action JSON to a whitelisted tool instead of letting the LLM run arbitrary scripts.
- `Controlled Executor`: run only approved actions such as diagnostics, checkpoint validation, ensemble search, postprocess search, fine-tuning, baseline training, refiner training, and submission packaging.
- `Evaluator`: deterministically compute local metrics, official-style proxy score, first-10-frame match, shape checks, and runtime.
- `Reviewer`: decide whether the result improves the frontier, is compliant, should be debugged, should be expanded, or should trigger submission.
- `Submitter`: build `pred.zip`, validate it locally, and write official-style logs and methodology artifacts.

The loop should run in this order for each research iteration:

1. Observe the task state: data shape, checkpoint availability, current best prediction, validation score, runtime, and previous failures.
2. Build compact memory: summarize only the relevant best candidates, rejected candidates, and rule constraints.
3. Ask the LLM planner for exactly one atomic action in JSON.
4. Validate the action schema before execution.
5. Route the action to one whitelisted tool.
6. Execute with time limits and deterministic artifact paths.
7. Evaluate with local metrics, official-style proxy score, first-10-frame check, and submission validator when needed.
8. Review whether the result is better, buggy, risky, or worth expanding.
9. Write journal records, LLM logs, code hashes, artifacts, and human-readable reports.
10. Promote the best candidate or package `pred.zip` only after validation passes.

Preferred action types for this repository:

```
inspect_data
diagnose_error
validate_baseline
weight_search
postprocess_search
finetune_checkpoint
train_refiner
baseline_zoo
task2_train_model
task2_submit_best
evaluate_candidate
submit_best
validate_submission
code_patch
stop
```

A good action contains:

```json
{
  "intent": "draft|improve|debug|submit|stop",
  "hypothesis": "one concrete scientific reason for this experiment",
  "action_type": "one whitelisted action type",
  "params": {},
  "expected_effect": "what metric, runtime, or reliability change should happen",
  "risk": "what can go wrong and how it will be checked"
}
```

Keep actions atomic and auditable. One action should change one idea only: a weight search, a postprocess search, a short fine-tune, a refiner branch, a baseline validation, or a packaging step. The Agent should optimize the competition objective, but every submitted result must remain traceable to journal records, LLM logs, code files, and validation artifacts.

Implementation preference: evolve the current `agent/pde_*` modules rather than importing ML-Master wholesale. Reuse the current controlled executor, journal, final submission builder, and validator. Add a stronger `Observer`, a stricter action schema, and an MCTS/LLM planner that generates tool calls dynamically instead of relying only on static YAML action lists.

#### PDE-specific Observer requirements

The Observer should give the Planner a compact, factual state report. For Task 1 it should include:

- Available data files and HDF5 dataset keys.
- Prediction shape and first-10-frame max error.
- Overall MSE, forecast-only MSE, segment-wise Rel-MSE/proxy score, and long-horizon RMSE.
- FNO vs U-Net vs ensemble comparison when cached predictions exist.
- Error concentration by time segment and by sample.
- Runtime and train-time budget status.
- Whether the current candidate is official-checkpoint-only, fine-tuned, or uses additional trained components.

For Task 2 it should include:

- Data shape, available train/test fields, and whether `Nu` is available.
- Confirmation that Task 1 data/checkpoints are not used.
- Current baseline type, validation split strategy, and inference runtime.
- Whether the result is only a persistence scaffold or a trained multi-Nu model.

#### Planner requirements

The Planner should not output free-form code as its first choice. It should output structured action JSON. Use free-form code patches only when an existing tool cannot express the experiment, and require validation after every patch.

Planner priority order:

1. Reproduce and validate the current baseline.
2. Diagnose where the baseline fails.
3. Try cheap search actions: ensemble, segment weights, persistence/postprocess, cached prediction blending.
4. Try controlled training actions: short checkpoint fine-tune, residual refiner, segment-weighted loss, spectral/physics losses.
5. Try model-family branches: FNO variants, DeepONet, PI-DeepONet, U-Net-style predictors, Baseline Zoo.
6. Submit only when the candidate passes shape, first-10-frame, runtime, code-log consistency, and local validator checks.

#### Executor and safety requirements

The Executor must stay deterministic and narrow. It should call repository tools with explicit parameters, not let the LLM improvise shell scripts. Whitelisted actions should write outputs under `runs/<study>/<node_id>/...`, and every artifact path should be recorded in the journal.

Never allow these through an Agent action:

- Calling numerical solvers to generate new labels.
- Mixing Task 1 checkpoint/data into Task 2.
- Writing predictions without first-10-frame validation.
- Packaging a submission without `methodology.pdf`.
- Bypassing `LLMCallLogger` for real LLM decisions.
- Creating untracked code changes that cannot be referenced from logs.

#### MCTS adaptation

Use ML-Master's search idea, not its Kaggle execution contract. In this repository, each MCTS node should be a structured PDE experiment, not an arbitrary generated Python script. Rewards should come from deterministic evaluator output:

- maximize official-style proxy score when available;
- otherwise minimize validation MSE;
- penalize failed validation, missing artifacts, non-finite metrics, excessive runtime, or compliance risk.

Expansion strategy:

- root nodes create baseline or diagnostic branches;
- successful nodes get `improve` children;
- failed nodes get `debug` children only when the failure is actionable;
- repeated low-improvement branches should terminate;
- best validated candidates are promoted to submission candidates.

#### Logging and submission traceability

Official scoring checks logs before metrics. Every Agent run should produce:

- `journal.json`: structured experiment tree.
- `journal_report.md`: readable experiment trajectory.
- `candidate_comparison.csv`: comparable metrics across candidates.
- `task{N}_logs.log`: official-style JSONL log for submitted tasks.
- code hash records for files copied into `submission/code`.
- prediction metrics and validation artifacts.
- final `pred.zip` path and validator result.

The final package should be reproducible from recorded action parameters. If a result cannot be traced from log records to code and artifacts, do not submit it.

For the stricter official code provenance requirement, final packaging may enable `require_llm_code_trace` in `submission.json` by running:

```powershell
& $PY -m agent.final_submission --run-name final-strict --require-llm-code-trace --provenance-log runs\task1\autonomous\<YYYYMMDD>\<study>\planner_logs.log
```

When this flag is enabled, `validate_submission()` only accepts real LLM `code_patch` records as proof for files copied into `code/`; synthetic records from `Task1FNOWorkflow`, `Task2PersistenceWorkflow`, `bootstrap`, `mock`, or static replay do not count. Strict code-trace packaging no longer appends `provider=codex` synthetic trace records, so submitted JSONL logs remain real LLM-call records. `--provenance-log` appends autonomous planner JSONL records into the final task logs. That means the Agent must first generate or rewrite submitted `code/` files through a logged real-LLM `code_patch` action, with validation artifacts attached.

For stricter autonomy review, final packaging can also require an audit:

```powershell
& $PY -m agent.final_submission --run-name final-strict --task1-run runs\<task1_run> --task2-run runs\<task2_run> --require-llm-code-trace --require-autonomy-audit --task1-study-dir runs\task1\autonomous\<YYYYMMDD>\<task1_study> --task2-study-dir runs\task2\autonomous\<YYYYMMDD>\<task2_study> --provenance-log runs\task1\autonomous\<YYYYMMDD>\<task1_study>\planner_logs.log --provenance-log runs\task2\autonomous\<YYYYMMDD>\<task2_study>\planner_logs.log
```

`--require-autonomy-audit` runs `agent.autonomy_audit.audit_autonomous_study()` before packing. The audit rejects `bootstrap/mock/static` planner providers, too-few LLM calls, missing source/baseline reading trace, missing completed `code_patch`, fewer than two metric-bearing experiments, and missing failed-experiment analysis. Task 1 finetune nodes must trace to the official Nu0.001 FNO checkpoint and include a planner-selected `temporal_stride=5`; Task 2 studies must not reference Task 1 assets.

### LLM client (`agent/llm.py`)

`OpenAICompatibleClient` talks to any OpenAI-compatible API. Provider profiles are in `configs/llm_providers.yaml`. `build_llm_client()` picks the right client based on `provider` key. All calls go through `logged_completion()` which writes JSONL records with `timestamp` and `elapsed_seconds`.

### Tools (`agent/tools.py`)

`ToolRunner` is a security sandbox: only paths under `project_root` (or `allowed_roots`) can be read/written; only whitelisted shell commands (`python`, `pytest`, `pip`, `git`) can execute. All writes and shell runs are tracked in a manifest.

### Submission pipeline (`agent/submission.py`, `agent/task1_submission.py`)

`validate_submission()` checks: `submission.json` with `problem_id: PDE_Burgers`, `task{N}_pred.hdf5` shape `(N, 200, 256)`, `task{N}_time.csv` with `train_time`/`inference_time`, `task{N}_logs.log` as valid JSONL with `timestamp`/`elapsed_seconds`. `validate_initial_condition()` verifies first 10 frames match the official input within tolerance. If `require_llm_code_trace` is true, it additionally verifies that each submitted code file can be traced to a real LLM `code_patch` response in the task logs.

### Code directory (`code/`)

Flat directory of standalone scripts — never add `__init__.py` (shadows Python stdlib `code`). Key files:
- `fno_inference.py` — FNO model definition + autoregressive inference (primary)
- `model_task1_fno.py` — Alternative FNO with `rollout()`
- `fno_ensemble.py` — Ensemble across Nu checkpoints
- `evaluate_task1.py` — MSE metrics
- `baseline_stub.py` — Zero-train baseline

## Critical constraints

- **`code/` must NOT have `__init__.py`** — it shadows Python's stdlib `code` module and breaks pytest
- Prediction shape must be `(N, 200, 256)`
- First 10 time steps must match the official initial condition
- Task 1 may use PDEBench checkpoints (`checkpoints/burgers_FNO.tar`); Task 2 must train from scratch with no Task 1 data/checkpoint leakage
- No numerical solver generated extra training data
- All LLM calls go through `LLMCallLogger` — never bypass it
- Agent code in `code/` must be traceable from logs

## Config system

Task configs (e.g. `configs/task1.yaml`) set `llm_profile` which resolves from `configs/llm_providers.yaml`. API keys come from env vars (`DEEPSEEK_API_KEY`, `KIMI_CODE_API_KEY`, `SILICONFLOW_API_KEY`, etc.). Use `provider: mock` for smoke tests without real API calls.

## Git ignoring

Data, checkpoints, model weights, logs, and submission zips stay out of git. The `.gitignore` is pre-configured — don't commit large binaries.
