# Current Experiment Steps

## Scope

The project is no longer treated as a single Task 1-only codebase. Current status is:

- Task 1 final packaging uses the compliant official checkpoint ensemble only.
- Task 2 final packaging keeps only a persistence scaffold, so the data flow is valid but not competitive yet.
- Old multi-`nu` Task 1 experiments are archived as historical context only.

## Step 1: Confirm Data

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -c "import h5py; [print(p, h5py.File(p,'r')['tensor'].shape) for p in ['data/Task1/task1_test.hdf5','data/Task1/task1_val.hdf5','data/Task2/task2_test.h5','data/Task2/task2_val.h5']]"
```

Expected:

- Task 1 test: `(1000, 10, 256)`
- Task 1 validation: `(100, 200, 256)`
- Task 2 test: `(1000, 10, 256)`
- Task 2 validation: `(100, 210, 256)`

## Step 2: Validate Task 1 Official Checkpoint Ensemble

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.run_task1_mcts_experiment --config configs\task1_mcts_validation_smoke.yaml --reset
```

This compares:

- `nu0.001=0.12`, `unet_pf20_nu0.001=0.88`
- `nu0.001=0.0`, `unet_pf20_nu0.001=1.0`
- `nu0.001=0.04`, `unet_pf20_nu0.001=0.96`

## Step 3: Build Final pred.zip

The final official-format package is generated in one run directory. Task 1 uses the active official ensemble (`Nu0.001_FNO=0.12`, `Unet-PF-20=0.88`), and Task 2 uses the persistence scaffold:

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.final_submission --run-name final-official-ensemble-task2-persistence --task1-weights 0.12 0.88 --task2 persistence
```

Expected final archive:

```text
runs\final-official-ensemble-task2-persistence\pred.zip
```

## Step 4: Validate Final Submission

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe -m agent.validate_submission --path runs\final-official-ensemble-task2-persistence
```

The local validator checks:

- `submission.json`
- non-empty `code/`
- `methodology.pdf`
- task prediction shapes `(N, 200, 256)`
- task time CSV files
- task JSONL logs
- code-log consistency

## Step 5: Optional Task 2 Scaffold Only

Use this only when debugging the Task 2 data flow separately:

```powershell
D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe code\task2_persistence_baseline.py --input data\Task2\task2_test.h5 --output runs\task2-persistence\task2_pred.hdf5
```

This verifies Task 2 output shape and first-10-frame copying. A competitive Task 2 model still needs to be trained from `data/Task2/task2_part*_train.h5`.

## Step 6: Read Results

Use `docs/results/task1_experiment_results.md` for the active Task 1 table. It deliberately excludes old multi-`nu` and `nu=0.1` experiments from the active ranking.
