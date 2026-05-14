# Task2 MiniFNO U-Net Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two train-from-scratch Task2 candidates, MiniFNO and Temporal U-Net, then keep the final Task2 submission on persistence unless validation improves.

**Architecture:** Add a self-contained Task2 module under `code/` with allowlisted HDF5 loading, two small PyTorch models, training/evaluation helpers, and a CLI that can smoke train or compare candidates against persistence. Add a separate inference CLI that only reads `task2_test.h5/tensor` plus a Task2 checkpoint and always copies the first 10 frames into the `(N, 200, 256)` output.

**Tech Stack:** Python, h5py, numpy, pytest, optional PyTorch for model training/inference.

---

### Task 1: Contract Tests

**Files:**
- Create: `tests/test_task2_train_from_scratch.py`

- [ ] **Step 1: Write failing tests**

Add tests that create tiny synthetic Task2 HDF5 files and assert:
- only Task2 allowlisted filenames are accepted;
- Task1-like data and checkpoint paths are rejected;
- batches expose `(input, target)` as `(N, 10, 256)` and `(N, 200, 256)`;
- both `minifno` and `unet` model outputs are `(N, 200, 256)`;
- model outputs copy the initial 10 frames exactly;
- inference does not require a `Nu` dataset in `task2_test.h5`.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_task2_train_from_scratch.py -q`

Expected: import errors for missing `code/train_task2_models.py` or missing functions.

### Task 2: Training Module

**Files:**
- Create: `code/train_task2_models.py`

- [ ] **Step 1: Implement allowlisted data IO**

Create `TASK2_ALLOWED_FILES`, `Task2TrajectoryDataset`, `load_task2_tensor`, and path validation that rejects paths outside the Task2 filename allowlist.

- [ ] **Step 2: Implement models**

Create `Task2MiniFNO`, `Task2TemporalUNet`, and `build_model(name, hidden)` with a shared output contract: return `(N, 200, 256)` and force `out[:, :10, :] = initial`.

- [ ] **Step 3: Implement train/eval CLI**

Add `train_one_model`, `evaluate_checkpoint`, `persistence_metrics`, and `main` with small defaults and explicit `--model {minifno,unet}`.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_task2_train_from_scratch.py -q`

Expected: all tests pass.

### Task 3: Inference Module

**Files:**
- Create: `code/infer_task2_model.py`

- [ ] **Step 1: Implement checkpoint-only inference**

Load model metadata from a Task2 checkpoint, read only `task2_test.h5/tensor`, run batched inference, and write `prediction`.

- [ ] **Step 2: Verify with smoke fixture**

Run: `pytest tests/test_task2_train_from_scratch.py -q`

Expected: all tests pass.

### Task 4: Smoke and Comparison

**Files:**
- No new files required.

- [ ] **Step 1: Run smoke training**

Run one tiny candidate training job with `--sample-limit` and `--epochs 1` to prove the real Task2 files can be read.

- [ ] **Step 2: Compare to persistence**

Run a short validation comparison. If neither candidate improves validation MSE/forecast MSE over persistence, do not replace the existing Task2 submission.
