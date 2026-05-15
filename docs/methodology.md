# AI4S PDE CNS Methodology

## Current Status

This repository now has two clearly separated tracks:

- **Task 1:** implemented and locally validated. The active compliant baseline uses only the two official Task 1 checkpoints:
  `1D_Burgers_Sols_Nu0.001_FNO.pt` and `1D_Burgers_Sols_Nu0.001_Unet-PF-20.pt`.
- **Task 2:** scaffolded only. The final package may include a persistence prediction so the official file flow is valid, but this is not a competitive model yet.

Older multi-`nu` and `nu=0.1` fine-tuning experiments are historical exploration records. They should not be treated as active final-submission evidence unless the race rules are clarified to allow those assets.

## Task 1 Active Pipeline

Input: `data/Task1/task1_test.hdf5` with shape `(1000, 10, 256)`.

Output: `task1_pred.hdf5` with shape `(1000, 200, 256)`. The first 10 frames are copied from the input and the remaining 190 frames are forecast.

Current active validation candidates:

```text
postprocessed blend:  segment FNO weights = 0.17 / 0.03 / 0.11
                      persistence alpha   = 0.89 / 0.95 / 0.41
final static blend:   nu0.001 FNO = 0.12, Unet-PF-20 = 0.88
proxy-score blend:    nu0.001 FNO = 0.04, Unet-PF-20 = 0.96
Unet-only baseline:   nu0.001 FNO = 0.00, Unet-PF-20 = 1.00
```

Local validation on `data/Task1/task1_val.hdf5`:

| Candidate | MSE | Proxy Score |
| --- | ---: | ---: |
| segment blend + persistence `0.17/0.03/0.11`, `0.89/0.95/0.41` | `0.0315804837` | `18.86073947` |
| best-MSE blend `0.12/0.88` | `0.0582305179` | `13.03828949` |
| proxy-score blend `0.04/0.96` | `0.0611382528` | `13.83318215` |
| Unet-only `0.00/1.00` | `0.0648512424` | `13.39284111` |
| FNO-only `1.00/0.00` | `0.4238491430` | `5.13830407` |

The postprocessed blend still uses only the two official Task 1 checkpoints.
The additional persistence component repeats the last observed input frame and
is applied as a conservative long-horizon stabilizer. No numerical solver is
called.

The autonomous loop now exposes this as a first-class Agent action:
`postprocess_search`. The planner can propose the action, the controlled
executor searches validation candidates, and the journal records both the
validation metrics and the exact `task1_extra_inference_args` needed by
`submit_best` or the final packaging CLI.

## Task 2 Scaffold

Input: `data/Task2/task2_test.h5` with shape `(1000, 10, 256)`.

Output: `task2_pred.hdf5` with shape `(1000, 200, 256)`. The official sample submission keeps the first 10 output frames equal to the input, so the current scaffold follows that convention and repeats the last observed frame for the remaining 190 frames.

Implemented scaffold:

- `code/task2_persistence_baseline.py`
- `agent/task2_workflow.py`
- `agent/task2_submission.py`

This is only a correctness baseline for validation and packaging. The competitive Task 2 model still needs a dedicated training strategy using `task2_part0_train.h5`, `task2_part1_train.h5`, and `task2_part2_train.h5`.

## Final Packaging Flow

The official packaging path is:

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.final_submission --run-name final-official-ensemble-postprocess-task2-persistence --task1-weights 0.12 0.88 --task1-segment-fno-weights 0.17 0.03 0.11 --task1-persistence-segment-alpha 0.89 0.95 0.41 --task2 persistence
```

This creates:

```text
runs/final-official-ensemble-postprocess-task2-persistence/
  submission.json
  task1_pred.hdf5
  task1_time.csv
  task1_logs.log
  task2_pred.hdf5
  task2_time.csv
  task2_logs.log
  methodology.pdf
  code/
  pred.zip
```

The final validator is:

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.validate_submission --path runs\final-official-ensemble-postprocess-task2-persistence
```

## Loss Design For New Training

`agent/task1_baseline_train.py` now supports optional loss terms:

- supervised trajectory MSE over a chosen time window
- initial-condition consistency loss
- spectral MSE with extra high-frequency weight
- Burgers residual loss `u_t + u u_x - nu u_xx`

Example:

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.run_task1_baseline_zoo --study-name task1-physics-loss-v1 --models residual_refiner --max-samples 4096 --steps 2000 --batch-size 16 --device cuda --loss-start-step 10 --loss-end-step 200 --physics-loss-weight 0.001 --spectral-loss-weight 0.01 --spectral-high-weight 4.0 --base-train-hdf5 runs\base_cache\task1_train_base.hdf5 --base-validation-prediction-path runs\_verify_official_ensemble\task1_val_fno_unet_012_088.hdf5
```

## Reproducibility Commands

Validate the active Task 1 MCTS path:

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.run_task1_mcts_experiment --config configs\task1_mcts_validation_smoke.yaml --reset
```

Run the minimal autonomous postprocess-search loop:

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.run_task1_autonomous_experiment --config configs\task1_mock.yaml --study-name task1-autonomous-postprocess-bootstrap --max-iterations 1 --metric competition_score_proxy --maximize --bootstrap-postprocess-search
```

Run direct Task 1 official checkpoint inference:

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe code\official_checkpoint_ensemble.py --input data\Task1\task1_test.hdf5 --output runs\task1-official-ensemble\task1_pred.hdf5 --batch-size 64 --models fno=checkpoints\extracted\1D_Burgers_Sols_Nu0.001_FNO.pt unet_pf20=checkpoints\extracted\1D_Burgers_Sols_Nu0.001_Unet-PF-20.pt --weights 0.12 0.88
```

Validate Task 2 scaffold:

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe code\task2_persistence_baseline.py --input data\Task2\task2_test.h5 --output runs\task2-persistence\task2_pred.hdf5
```
