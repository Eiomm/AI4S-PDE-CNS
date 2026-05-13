# AI4S PDE CNS Task 1 Methodology

## Official Constraint Check

This submission is for Task 1 only. It follows the public race description and local
submission validator:

- Prediction file: `task1_pred.hdf5`
- Prediction dataset: `tensor`
- Prediction shape: `(1000, 200, 256)`
- The first 10 time steps are copied from `task1_test.hdf5`
- The submitted forecast covers the future 190 time steps
- `task1_time.csv`, `task1_logs.log`, `submission.json`, `methodology.pdf`, and `code/`
  are included in `pred.zip`

Task 1 allows fine-tuning from the official PDEBench checkpoint. No numerical solver
generated extra data is used.

## Data And Checkpoints

The final Task 1 model uses:

- Official Task 1 initial condition: `data/Task1/task1_test.hdf5`
- Local validation target: `data/Task1/task1_val.hdf5`
- PDEBench 1D Burgers training data for `nu=0.1`
- Official PDEBench FNO checkpoint for `nu=0.1`
- Official PDEBench FNO checkpoint for `nu=0.01`

The selected fine-tuned checkpoint is:

```text
runs/task1-finetune-nu0.1-lr3e-6-short-proxy/best.pt
```

## Fine-Tuning Procedure

The `nu=0.1` FNO checkpoint is fine-tuned with one-step supervised windows from the
PDEBench 1D Burgers `nu=0.1` data. Two controlled runs were compared:

```text
Short proxy-selected run:
  steps: 250
  learning_rate: 3e-6
  batch_size: 8
  max_samples: 2048
  validation_every: 25
  best checkpoint metric: competition_score_proxy, maximize

Larger-data robustness run:
  steps: 1200
  learning_rate: 3e-6
  batch_size: 16
  max_samples: 8000
  validation_every: 50
  best checkpoint metric: competition_score_proxy, maximize

Common settings:
  grad_clip: 0.1
  weight_decay: 0.0
  sample_start: 0
```

Before any update, the script evaluates the base checkpoint on local Task 1 validation.
`best.pt` is saved only when validation improves over the base checkpoint. This prevents
degraded fine-tuning runs from replacing the official checkpoint.

The larger-data run did not improve validation proxy score. It peaked early and then
degraded, indicating over-training relative to the Task 1 target distribution. The final
submission therefore uses the short proxy-selected checkpoint, chosen by explicit
validation evidence rather than by convenience.

Measured local elapsed time used for timing records:

```text
fine-tune short run:       10.968621 seconds
weight selection/search:   about 37 seconds
reported train_time:       48.000000 seconds
```

The local GPU is an NVIDIA GeForce RTX 5070, which is not listed in the published A100
time conversion table. The reported time is local measured time before an official RTX
5070-to-A100 conversion coefficient is known.

## Ensemble And Selection

The final prediction is a weighted ensemble:

```text
nu0.01 official FNO checkpoint:         0.085
nu0.1 fine-tuned FNO checkpoint:        0.915
nu0.001 official FNO checkpoint:        0.00
nu1.0 official FNO checkpoint:          0.00
```

Selection uses a local proxy for the public scoring rule instead of plain MSE. The proxy
removes the first 10 initial-condition frames, then scores the remaining 190 frames in
three segments:

- Segment 1: relative MSE on steps 0-47, converted by `100 * exp(-20 * rel_mse)`
- Segment 2: relative MSE on steps 47-95, converted by `100 * exp(-10 * rel_mse)`
- Segment 3: RMSE on steps 95-190, converted by the Lorentzian component
  `100 / (1 + 10 * rmse)`

The official description also mentions a Frechet component for the third segment. Since
the page does not provide enough implementation detail to reproduce it exactly, this
submission uses the Lorentzian component as a transparent local proxy and keeps raw MSE,
forecast MSE, segment relative MSE, and proxy score in local experiment records.

## Local Validation Result

The selected candidate is:

```text
runs/task1-finetune-nu0.1-short-proxy-weight-search/rank-13-mse-0.0016034222
```

Local validation metrics:

```text
mse:                       0.001603422098378363
forecast_mse:              0.0016878127351351185
competition_score_proxy:   58.18577978794908
segment1_rel_mse:          0.058978976426412894
segment2_rel_mse:          0.07040701233226918
segment3_rmse:             0.031108425466246738
```

## Baseline Zoo Extension

The next Task 1 iteration adds a controlled Baseline Zoo around the current FNO
fallback. This does not replace the selected submission unless validation improves.
The zoo records every candidate in the same experiment format:

```text
FNO ensemble:        current checkpoint/fine-tuned fallback
TFNO:                optional neuraloperator branch when neuralop is installed
U-Net 1D:            trajectory-to-trajectory prototype
DeepONetLite:        branch/trunk PyTorch prototype
PINO-FNO:            FNO-style prototype with physics/spectral losses
Residual Refiner:    correction model over a base rollout
```

Each Baseline Zoo run writes `metrics.json`, `run_result.json`,
`experiment_memory.json`, and `baseline_manifest.json`. Successful validation
predictions can be combined by global convex ensemble and feature-cluster EM
ensemble. Only a candidate that beats the current validation fallback and passes
`validate_submission` may produce a new `pred.zip`.

## Reproducibility Commands

Fine-tune:

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe code\train_task1_fno_finetune.py --train-hdf5 data\pdebench_burgers\raw\1D_Burgers_Sols_Nu0.1.hdf5 --base-checkpoint checkpoints\extracted\1D_Burgers_Sols_Nu0.1_FNO.pt --run-dir runs\task1-finetune-nu0.1-lr3e-6-short-proxy --steps 250 --batch-size 8 --max-samples 2048 --sample-start 0 --val-every 25 --log-every 25 --lr 3e-6 --weight-decay 0.0 --grad-clip 0.1 --val-max-samples 100 --selection-metric competition_score_proxy --selection-direction max
```

Evaluate:

```powershell
python code\evaluate_task1.py --prediction runs\task1-finetune-nu0.1-short-proxy-weight-search\rank-13-mse-0.0016034222\task1_val_pred.hdf5 --target data\Task1\task1_val.hdf5
```

Validate final submission:

```powershell
python -m agent.validate_submission --path runs\task1-finetune-nu0.1-short-proxy-final
```

Run Baseline Zoo prototype:

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.run_task1_baseline_zoo --study-name task1-zoo-prototype --models fno_ensemble,unet1d,deeponet_lite,residual_refiner --max-samples 1024 --steps 200
```
