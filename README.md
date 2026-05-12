# AI4S-PDE-CNS

Auditable Agent framework for the AI4S CNS challenge race 7: neural operator PDE Agent.

## Goal

This repository starts with the compliance and automation layer, not leaderboard chasing. The first working version records every LLM call as JSONL, restricts tool execution to the project workspace, validates official submission structure, and gives later FNO or DeepONet baselines a clean place to plug in.

## Environment

Use Python 3.11 or 3.12. The system Python discovered during planning was 3.13, so create a dedicated environment before installing training dependencies.

```powershell
cd D:\Study\AI4S-PDE-CNS
conda create -n ai4s-pde-cns python=3.12 -y
conda activate ai4s-pde-cns
pip install -e ".[dev]"
```

For GPU training, install a PyTorch build that matches the local CUDA driver, then add baseline dependencies such as `neuraloperator`.

## Agent Commands

Run a mock smoke test loop:

```powershell
python -m agent.run --task task1 --config configs\task1.yaml
```

Validate a submission directory:

```powershell
python -m agent.validate_submission --path submission
```

Pack a validated run directory:

```powershell
python -m agent.pack_submission --run runs\<run_id>
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
