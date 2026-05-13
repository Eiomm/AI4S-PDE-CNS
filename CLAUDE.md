# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AI4S CNS Challenge Race 7 — neural operator PDE Agent for 1D Burgers equation prediction. The framework wraps an LLM-driven observe-plan-act-record loop with sandboxed tool execution, submission validation, and auditable JSONL logging.

Full competition rules and data layout are in `AGENT.md`. Read it first if you haven't.

## Essential commands

```powershell
# Run tests
python -m pytest -q

# Run a single test file
python -m pytest tests/test_submission.py -q

# Mock agent smoke test
python -m agent.run --task task1 --config configs\task1_mock.yaml

# Real agent run
python -m agent.run --task task1 --config configs\task1.yaml

# Validate a submission directory
python -m agent.validate_submission --path runs\<run_id>

# Pack a validated submission
python -m agent.pack_submission --run runs\<run_id>
# Default output: runs\<run_id>\pred.zip

# Check LLM connectivity
python -m agent.check_llm --config configs\deepseek.example.yaml

# Zero-train baseline (copies initial conditions, no model)
python -m agent.zero_submission --input data\data_and_sample_submission\data_and_sample_submission\train_val_test_init\task1_test.hdf5 --output-dir runs\task1-zero --code-dir code

# Validate initial condition match (first 10 frames)
python -c "from agent.submission import validate_initial_condition; validate_initial_condition(r'runs\...\task1_pred.hdf5', r'data\...\task1_test.hdf5'); print('ok')"

# Evaluate Task 1 predictions
python code\evaluate_task1.py --prediction runs\...\task1_pred.hdf5 --target data\...\task1_val.hdf5 --output runs\...\metrics.json
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

### Agent loop (`agent/run.py`)

The core is `run_agent()`: **observe** → **plan** (LLM call via `logged_completion`) → **act** (tool dispatch) → **record** (JSONL log + manifest). Tools are restricted by path whitelist and command whitelist.

### LLM client (`agent/llm.py`)

`OpenAICompatibleClient` talks to any OpenAI-compatible API. Provider profiles are in `configs/llm_providers.yaml`. `build_llm_client()` picks the right client based on `provider` key. All calls go through `logged_completion()` which writes JSONL records with `timestamp` and `elapsed_seconds`.

### Tools (`agent/tools.py`)

`ToolRunner` is a security sandbox: only paths under `project_root` (or `allowed_roots`) can be read/written; only whitelisted shell commands (`python`, `pytest`, `pip`, `git`) can execute. All writes and shell runs are tracked in a manifest.

### Submission pipeline (`agent/submission.py`, `agent/task1_submission.py`)

`validate_submission()` checks: `submission.json` with `problem_id: PDE_Burgers`, `task{N}_pred.hdf5` shape `(N, 200, 256)`, `task{N}_time.csv` with `train_time`/`inference_time`, `task{N}_logs.log` as valid JSONL with `timestamp`/`elapsed_seconds`. `validate_initial_condition()` verifies first 10 frames match the official input within tolerance.

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
