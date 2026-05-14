# AI4S-PDE-CNS

Auditable Agent framework for the AI4S CNS challenge race 7: neural operator PDE Agent.

## Goal

This repository starts with the compliance and automation layer, not leaderboard chasing. The first working version records every LLM call as JSONL, restricts tool execution to the project workspace, validates official submission structure, and gives later FNO or DeepONet baselines a clean place to plug in.

## Environment

The local Windows project environment is `Hwpytorch`; use it for GPU/MCTS runs:

```powershell
cd D:\Study\AI4S-PDE-CNS
$env:AI4S_PROJECT_PYTHON="D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe"
.\scripts\run-task1-mcts.ps1 -Config configs\task1_mcts_full.yaml -Reset
```

For a fresh environment, use Python 3.10, 3.11, or 3.12. The system Python discovered during planning was 3.13, so create a dedicated environment before installing training dependencies.

```powershell
cd D:\Study\AI4S-PDE-CNS
conda create -n ai4s-pde-cns python=3.10 -y
conda activate ai4s-pde-cns
pip install -e ".[dev]"
```

For GPU training, install a PyTorch build that matches the local CUDA driver, then add baseline dependencies such as `neuraloperator`.

## Agent Commands

Run a mock smoke test loop:

```powershell
python -m agent.run --task task1 --config configs\task1.yaml
```

Run the Task 1 MCTS validation-plus-submission workflow with the default local GPU environment:

```powershell
.\scripts\run-task1-mcts.ps1 -Config configs\task1_mcts_full.yaml -Reset
```

Current scope notes:

- Task 1 is the only optimized track at the moment.
- Task 2 now has a persistence-baseline scaffold for data-flow and packaging checks, but no competitive model yet.
- Current experiment steps are documented in `docs/current_experiment_steps.md`.

For a fast MCTS structure-only check:

```powershell
.\scripts\run-task1-mcts.ps1 -Config configs\task1_mcts_mock.yaml -MaxSteps 3 -Reset
```

Validate a submission directory:

```powershell
python -m agent.validate_submission --path submission
```

Pack a validated run directory:

```powershell
python -m agent.pack_submission --run runs\<run_id>
```

Default output: `runs\<run_id>\pred.zip`.

Run the AIDE-style autonomous Task 1 experiment loop:

```powershell
python -m agent.run_task1_autonomous_experiment --config configs\kimi.example.yaml --study-name task1-autonomous --max-iterations 3
```

This loop writes `runs\<study_name>\journal.json`, `planner_logs.log`, and `autonomous_summary.json`. The LLM must emit one structured atomic plan per iteration with `action_type` equal to `weight_search`, `finetune`, `code_patch`, `submit_best`, or `stop`. `code_patch` is allowed to rewrite files under `code/`, but paths outside `code/` are rejected and every node is reviewed back into the journal.

For a local autonomous smoke test without real API calls:

```powershell
python -m agent.run_task1_autonomous_experiment --config configs\task1_mock.yaml --study-name autonomous-smoke --max-iterations 1
```

The autonomous loop also writes `journal_report.md`. In the CLI runner, every `code_patch` must include either a `validation_command` list or `submission_validation_path`. `validation_command` is limited to validation-oriented commands such as `python` or `pytest`; `submission_validation_path` calls the local submission validator, including code-log consistency checks. Any failure marks the node as failed and the reviewer recommends a debug step.

Run a real lightweight Task 1 autonomous weight-search node:

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.run_task1_autonomous_experiment --config configs\task1_mock.yaml --study-name task1-autonomous-bootstrap --max-iterations 1 --bootstrap-weight-search --bootstrap-grid-step 0.01 --bootstrap-grid-radius 2 --metric competition_score_proxy --maximize
```

The bootstrap grid shifts weight locally between the official `nu0.001` FNO checkpoint and the official `unet_pf20_nu0.001` checkpoint. The current compliant default is a FNO/Unet-PF prediction-level blend selected on local validation. Every candidate's validation metrics are stored in the journal artifact under `candidate_results` and flattened into `candidate_comparison.csv`, while `experiment_comparison.csv` and the global registry keep the node-level best for quick ranking.

Compare recorded experiments:

```powershell
python -m agent.compare_experiments --metric competition_score_proxy --maximize --top-k 10
```

Run the Task 1 Baseline Zoo on GPU:

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.run_task1_baseline_zoo --study-name task1-zoo-prototype --models fno_ensemble,unet1d,deeponet_lite,residual_refiner --max-samples 1024 --steps 200 --device cuda
```

Run a Task 1 physics/spectral-loss training branch:

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.run_task1_baseline_zoo --study-name task1-physics-loss-v1 --models pino_fno --max-samples 4096 --steps 2000 --batch-size 16 --device cuda --physics-loss-weight 0.001 --spectral-loss-weight 0.01 --spectral-high-weight 4.0
```

Search validation-only combinations without creating a submission zip:

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.run_task1_combo_search --study-dir runs\task1-zoo-medium-finetuned-fno --target data\Task1\task1_val.hdf5
```

Train a DeepONetLite long-horizon specialist, using validation-only gating before any packaging:

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.run_task1_baseline_zoo --study-name task1-zoo-tail120-deeponet-v2 --models fno_ensemble,deeponet_lite --max-samples 12000 --steps 8000 --batch-size 4 --lr 0.0003 --hidden 96 --device cuda --loss-start-step 120 --loss-end-step 200 --fno-weight nu0.001=0.12 --fno-weight unet_pf20_nu0.001=0.88
```

Update the experiment ledger and clean stale submission zips:

```powershell
python -m agent.update_experiment_results --runs-root runs --output docs\results\task1_experiment_results.md --cleanup-zips
```

Run the zero-train Task 1 smoke baseline on the official test initial condition:

```powershell
python code\baseline_stub.py --input data\data_and_sample_submission\data_and_sample_submission\train_val_test_init\task1_test.hdf5 --output runs\task1-official-smoke\task1_pred.hdf5
```

Create a complete zero-train Task 1 submission bundle:

```powershell
python -m agent.zero_submission --input data\data_and_sample_submission\data_and_sample_submission\train_val_test_init\task1_test.hdf5 --output-dir runs\task1-zero-submission --code-dir code
python -m agent.validate_submission --path runs\task1-zero-submission
```

Run the Task 2 persistence scaffold:

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe code\task2_persistence_baseline.py --input data\Task2\task2_test.h5 --output runs\task2-persistence\task2_pred.hdf5
```


Evaluate a Task 1 validation prediction against `task1_val.hdf5`:

```powershell
python code\evaluate_task1.py --prediction runs\task1-val-zero\task1_val_pred.hdf5 --target data\data_and_sample_submission\data_and_sample_submission\train_val_test_init\task1_val.hdf5 --output runs\task1-val-zero\metrics.json
```
## LLM Provider

The default configs use `provider: mock` for safe local testing. Real providers are configured through `configs/llm_providers.yaml`, and task configs switch by setting `llm_profile`.

Available profiles:

- `deepseek`: DeepSeek official API, default model `deepseek-v4-pro`.
- `deepseek_flash`: DeepSeek official API, lower-cost `deepseek-v4-flash`.
- `kimi`: Kimi Code API, fixed model `kimi-for-coding`.
- `kimi_open_platform`: Moonshot/Kimi Open Platform API, default model `kimi-k2.6`.
- `siliconflow_glm`: SiliconFlow GLM profile.
- `siliconflow_deepseek`: SiliconFlow DeepSeek profile.

Provider smoke checks:

```powershell
$env:DEEPSEEK_API_KEY="..."
python -m agent.check_llm --config configs\deepseek.example.yaml

$env:KIMI_CODE_API_KEY="..."
python -m agent.check_llm --config configs\kimi.example.yaml

$env:SILICONFLOW_API_KEY="..."
python -m agent.check_llm --config configs\siliconflow.example.yaml
```

All model calls must go through `LLMCallLogger` so task logs contain JSON lines with `timestamp` and `elapsed_seconds`.

Autonomous experiment planning uses `ExperimentJournal` and `CandidateNode` records so each attempt has a hypothesis, action, parent node, metrics, artifacts, and review decision. This is the ML-Master/AIDE-inspired layer above the lower-level PDE workflows.

## Competition Constraints

- At least one task must be submitted.
- Each submitted task must include `task{N}_pred.hdf5`, `task{N}_time.csv`, and `task{N}_logs.log`.
- Prediction shape must be `(N, 200, 256)`.
- The final prediction HDF5 must store predictions in dataset `tensor`, matching the official sample submission.
- The first 10 time steps must match the given initial condition in the final prediction file.
- `agent.submission.validate_initial_condition(...)` can check the first 10 frames against the official HDF5 input before packaging.
- Task 1 may use official PDEBench checkpoints.
- Task 2 must train from scratch and must not reuse Task 1 data or checkpoint.
- Numerical solver generated extra training data is disallowed.
- Agent code and logs must be mutually traceable.

## Next Baseline Step

Start with Task 1 FNO via `neuraloperator`. Implement the training/inference files in `code/`, write predictions into `runs/<run_id>/`, validate, then copy the selected artifacts into `submission/`.
