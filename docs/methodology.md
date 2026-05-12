# Methodology Draft

## Agent Architecture

The Agent follows an observe-plan-act-record loop. It observes project files, race notes, experiment outputs, and logs. It asks an API-hosted LLM to plan the next step, executes only approved tools, and records every LLM response and action.

## LLM Logging

Every LLM call is written as one JSON line in `task{N}_logs.log`. Each line contains `timestamp`, `elapsed_seconds`, `provider`, `model`, `messages`, and `response`.

## Tool Use

The first version allows file reads, file writes inside approved roots, and allowlisted shell commands. Every write and shell command is recorded in the run manifest.

## PDE Modeling Plan

Task 1 will start from FNO through `neuraloperator` or the official PDEBench checkpoint. Task 2 will be trained from scratch and kept isolated from Task 1 data and checkpoint artifacts.

## Task 1 Real-LLM Closure Test

This section records the verified end-to-end test flow for Task 1 with a real DeepSeek API-backed Agent.

### Inputs and Secrets

- API keys are stored in the project-root `.env`, which is ignored by git.
- `configs/task1.yaml` loads `.env` through `env_file: .env`.
- Do not print API key values. Only check whether `DEEPSEEK_API_KEY` is present.

Connectivity check:

```powershell
python -m agent.check_llm --config configs\task1.yaml
```

Expected result: a run directory under `runs/api-check-deepseek-*` with `response.json` containing `API_OK`.

### Model Strategy

The current Task 1 model is a weighted PDEBench FNO checkpoint ensemble:

```text
Nu0.001 weight 0.01
Nu0.01  weight 0.31
Nu0.1   weight 0.66
Nu1.0   weight 0.02
```

Validation metrics from `runs/task1-weighted-val/metrics.json`:

```text
mse:              0.001679162949145986
forecast_mse:     0.0017675399464694265
long_horizon_mse: 0.001573041832931417
```

The weighted ensemble improves over the earlier equal three-model ensemble (`mse` about `0.002843`).

### Real Agent Run

Run the real Agent:

```powershell
python -m agent.run --task task1 --config configs\task1.yaml --project-root .
```

Verified run:

```text
runs/task1-20260512-222949
provider: deepseek
model: deepseek-v4-pro
```

The Agent ran weighted test inference with `Hwpytorch` through the configured Python alias:

```text
prediction: runs/task1-weighted-test-v2/task1_pred.hdf5
inference_time: 15.888068 seconds
```

### Failure Found During Real Test

The first real run (`runs/task1-20260512-222402`) completed weighted inference but failed packaging. Root cause:

- The LLM supplied a non-existent or fabricated `log_path`.
- The packaging step then used an invalid task log instead of the real Agent JSONL log.

Fix:

- `create_task1_submission` execution now always packages the current run's real `task1_logs.log`.
- This prevents the LLM from accidentally or intentionally replacing the official trace log.

### Final Packaging

The clean final package was regenerated from:

```text
prediction: runs/task1-weighted-test-v2/task1_pred.hdf5
log:        runs/task1-20260512-222949/task1_logs.log
code:       code/
```

Final artifacts:

```text
runs/task1-real-submission
runs/task1-real-submission.zip
```

Timing written to `task1_time.csv`:

```text
train_time,inference_time
177.541932,15.888068
```

Here `train_time` includes real Agent planning and non-inference overhead. No model training was performed.

### Independent Verification

Validate the submission directory:

```powershell
python -c "from agent.submission import validate_submission; import json; r=validate_submission('runs/task1-real-submission'); print(json.dumps(r.__dict__, ensure_ascii=False, indent=2))"
```

Expected:

```json
{
  "valid": true,
  "tasks": ["task1"],
  "messages": ["ok"]
}
```

Check prediction shape and initial frames:

```powershell
python code\check_pred_shape.py runs\task1-real-submission\task1_pred.hdf5 --input data\data_and_sample_submission\data_and_sample_submission\train_val_test_init\task1_test.hdf5
```

Verified output:

```text
shape: (1000, 200, 256)
dtype: float32
first_ten_match: True
max_initial_error: 2.384185791015625e-07
finite: True
```

Run the unit and smoke suite:

```powershell
python -m pytest -q
```

The closure is considered valid only when all three checks pass: API check, Agent run, and independent submission validation.
