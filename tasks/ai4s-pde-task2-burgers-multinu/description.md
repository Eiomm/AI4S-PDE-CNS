# Task 2: Multi-Nu 1D Burgers Prediction

You are solving Task 2 of the AI4S PDE neural-operator challenge.

Your script must produce **two separate output files** in CWD:

```text
task2_pred.hdf5             # the (1000, 200, 256) prediction tensor — and nothing else
task2_inference_time.txt    # the test-set rollout wall-clock, in seconds
```

Keep them strictly separate: the HDF5 file contains only the prediction
tensor (no attributes), and the txt file contains only the
elapsed-seconds number (one line, e.g. `42.150`).

In addition, also **print** `INFERENCE_TIME=<seconds, float>` to stdout
on its own line. The submission packager prefers the txt file, falls
back to stdout, and re-measures locally as last resort. Do **not**
create `task2_time.csv` yourself.

The 2-minute hard cap is checked against this measured value, so the
agent should `assert inference_time < 120` and, if it fails, rerun with
a faster configuration (larger batch, smaller model, GPU).

## Objective

Predict full 1D Burgers trajectories under **multiple viscosity coefficients
`Nu` simultaneously**. Unlike Task 1 (single `Nu=0.001`), training data spans
`Nu ∈ [1e-4, 1e-2]` (roughly two orders of magnitude). The model must
generalize across this range.

Training and validation data carry a per-sample `nu` value. The test data
does **not**. At test time, predict from the first 10 observed frames only.

The prediction must come from a neural model or neural-operator style method.
Do not use a numerical PDE solver to generate the prediction.

## ❗ Hard Constraints (read before writing any code)

Four zero-tolerance rules. Violating any one of them causes the entire
submission to be scored **0** by the official grader.

1. **Train from scratch — no pretrained checkpoints, no public weights.**
   Do not load `*.pt` files from PDEBench, HuggingFace, Task-1's checkpoint
   directory, or any other source. Initialize every parameter randomly and
   train only on the three `task2_part*_train.h5` files. This is the
   single biggest difference from Task 1.
2. **`data/task2_val.h5` is for evaluation only.** Do not pass any tensor
   from this file into a training loss, optimizer step, or backward pass.
   Validation `nu` may be read for stratified metrics but not for gradients.
3. **`data/task2_test.h5` first-10-frame copy is mandatory.** Frames
   `[:, :10, :]` of `task2_pred.hdf5` must equal `task2_test.h5["tensor"]`
   within `1e-3`. Predict only frames `[:, 10:200, :]`.
4. **Test-time inference is `nu`-free.** `task2_test.h5` does not contain
   `nu`. Your inference path may infer `nu` from the 10-frame prefix or
   ignore `nu` entirely, but it must not assume `nu` is given.

Time/compute limits enforced by the grader:

- Total training wall-clock (Agent thinking + actual training) ≤ **12 h**.
- Inference time on the full test set ≤ **2 min**. Exceeding it scores 0.

## Input Data

```text
data/task2_part0_train.h5
data/task2_part1_train.h5
data/task2_part2_train.h5
data/task2_val.h5
data/task2_test.h5
```

HDF5 schemas (note the **`t_coordinate` / `x_coordinate` underscores** — Task 1
used hyphens):

```text
task2_part{0,1,2}_train.h5
  tensor        float32  (1000, 320, 256)
  nu            float32  (1000,)            ← per-sample viscosity
  t_coordinate  float32  (320,)
  x_coordinate  float32  (256,)

task2_val.h5
  tensor        float32  (100, 210, 256)
  nu            float32  (100,)
  t_coordinate  float32  (210,)
  x_coordinate  float32  (256,)

task2_test.h5
  tensor        float32  (1000, 10, 256)    ← only first 10 frames
  t_coordinate  float32  (10,)
  x_coordinate  float32  (256,)
  (no `nu` field)
```

Three training parts concatenate to **3000 trajectories**, each with **320
timesteps**. You predict only 200 frames at test time, so training
trajectories are 1.6× longer than you need — that buffer is for sampling
random rollout windows.

## Workspace Layout

Your script will be executed with **CWD = task root**, but the script itself
may live in a `run/` subdirectory:

```text
<task root>             ← os.getcwd() points here
├── data/
│   ├── task2_part0_train.h5
│   ├── task2_part1_train.h5
│   ├── task2_part2_train.h5
│   ├── task2_val.h5
│   └── task2_test.h5
├── config.yaml
└── run/
    └── _sandbox_script_*.py    ← your script lives here, __file__.parent = run/
```

Resolve every path against `os.getcwd()` or use plain relative strings
(`"data/task2_part0_train.h5"`), never against `__file__`.

There is no `src/` directory and no helper package — the model code is
inlined below. Copy it into your script.

## Domain Knowledge

### Why this task is different from Task 1

| | Task 1 | Task 2 |
|---|---|---|
| Viscosity | fixed `Nu=0.001` | `Nu ∈ [1e-4, 1e-2]`, per-sample |
| Pretrained checkpoint | allowed and recommended | **forbidden** |
| Training data | optional 8 GB external file | mandatory 3 × 1 GB local files |
| Train trajectory length | 200 raw / 40 model frames | 320 frames (no downsampling needed) |
| HDF5 coordinate keys | `t-coordinate` / `x-coordinate` (hyphens) | `t_coordinate` / `x_coordinate` (**underscores**) |
| Inference time budget | informal | hard 2-minute cap |

Do not copy Task-1's data-loading code verbatim — the coordinate-key names
and tensor shapes are different.

### Multi-Nu strategy: condition or not?

`nu` is available at training time but not at test time. Three legal
approaches, in increasing complexity:

1. **`nu`-agnostic FNO.** Ignore `nu` entirely. Train one model on the
   mixed-`Nu` data; the network implicitly learns to identify the regime
   from the 10-frame input window. Simplest, lowest risk, often
   competitive when training data is plentiful.
2. **Inferred-`nu` conditioning.** Add a tiny auxiliary head that
   regresses `nu` from the first-10-frame window (trained with the known
   `nu` labels). At test time, run this head first to estimate `nu`, then
   condition the main FNO on the estimate.
3. **`nu`-as-extra-channel.** During training, broadcast `log(nu)` (or
   `nu` itself) to a `(256,)` spatial map and feed it as an additional
   input channel to the FNO. At test time, predict `nu` from the prefix
   and feed the predicted value. Strictly more powerful than #2 because
   `nu` participates in every spatial feature map, not just the bottleneck.

Start with **approach #1** — it is one less moving part and one less
failure mode (e.g., a bad `nu` estimator can hurt more than no `nu`).
Escalate to #2 / #3 only if validation `nu` stratification shows large
per-`Nu` errors.

### Inline FNO model (paste into your script)

Same architecture as Task 1 (FNO1d, 4 spectral layers). You may freely
modify it — change `width`, `modes`, `n_layers`, add `nu` conditioning,
etc. — since you are training from scratch. The version below is a
reasonable starting point for a `nu`-agnostic baseline.

```python
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat)
        )

    def forward(self, x):
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            x.shape[0], self.out_channels, x.size(-1) // 2 + 1,
            device=x.device, dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = torch.einsum(
            "bix,iox->box", x_ft[:, :, : self.modes1], self.weights1
        )
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNO1d(nn.Module):
    """Set extra_channels=0 for nu-agnostic. Set =1 to feed a log-nu broadcast map."""
    def __init__(self, num_channels=1, modes=16, width=64, initial_step=10, extra_channels=0):
        super().__init__()
        self.padding = 2
        self.fc0 = nn.Linear(initial_step * num_channels + 1 + extra_channels, width)
        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x, grid, extras=None):
        """x: (B, X, 10*num_channels); grid: (B, X, 1); extras: (B, X, extra_channels) or None"""
        feats = [x, grid]
        if extras is not None:
            feats.append(extras)
        x = torch.cat(feats, dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = F.pad(x, [0, self.padding])
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)
        x = x[..., : -self.padding]
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.unsqueeze(-2)
```

### Rollout helper

```python
@torch.no_grad()
def rollout_fno(model, initial, x_coords, total_steps, device, batch_size,
                extras_per_sample=None):
    """
    initial: (N, 10, 256) np.float32.
    extras_per_sample: optional (N, extra_channels) for nu conditioning; will be
        broadcast to (N, 256, extra_channels) inside.
    Returns (N, total_steps, 256) np.float32.
    """
    n_samples, initial_step, spatial_size = initial.shape
    x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
    grid = torch.tensor(x_norm, dtype=torch.float32, device=device).view(1, spatial_size, 1)
    prediction = np.zeros((n_samples, total_steps, spatial_size), dtype=np.float32)
    prediction[:, :initial_step, :] = initial
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = initial[start:end]
        xx = torch.tensor(batch, dtype=torch.float32, device=device).permute(0, 2, 1).contiguous()
        batch_grid = grid.expand(end - start, -1, -1)
        if extras_per_sample is not None:
            e = torch.tensor(extras_per_sample[start:end], dtype=torch.float32, device=device)
            batch_extras = e.view(end - start, 1, -1).expand(-1, spatial_size, -1)
        else:
            batch_extras = None
        frames = [batch[:, i, :] for i in range(initial_step)]
        for _ in range(initial_step, total_steps):
            pred = model(
                xx.reshape(xx.shape[0], xx.shape[1], -1), batch_grid, batch_extras
            ).squeeze(-1).squeeze(-1)
            frames.append(pred.cpu().numpy())
            xx = torch.cat((xx[:, :, 1:], pred.unsqueeze(-1)), dim=2)
        prediction[start:end] = np.stack(frames, axis=1)
    return prediction
```

### Training from scratch — recipe

```python
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load and concatenate the 3 training parts.
train_chunks = []
nu_chunks = []
for i in range(3):
    with h5py.File(f"data/task2_part{i}_train.h5", "r") as f:
        train_chunks.append(f["tensor"][:].astype(np.float32))   # (1000, 320, 256)
        nu_chunks.append(f["nu"][:].astype(np.float32))
        x_coords = f["x_coordinate"][:].astype(np.float32)        # underscores!
train_u  = np.concatenate(train_chunks, axis=0)                   # (3000, 320, 256)
train_nu = np.concatenate(nu_chunks, axis=0)                      # (3000,)
del train_chunks, nu_chunks
print(f"train: {train_u.shape}, nu range [{train_nu.min():.2e}, {train_nu.max():.2e}]")

# 2. Optional input normalization (compute on train ONLY).
u_mean = float(train_u.mean())
u_std  = float(train_u.std())
train_u = (train_u - u_mean) / u_std    # normalize → denorm before metric

# 3. Build model. Start nu-agnostic (extra_channels=0).
model = FNO1d(num_channels=1, modes=16, width=64, initial_step=10,
              extra_channels=0).to(device)
print(f"params={sum(p.numel() for p in model.parameters()):,}")

# 4. Train. The full 3000-sample / 320-frame trajectories let you sample
#    rollout windows freely. Use random t0 to expose the model to every
#    temporal regime, and curriculum-step the horizon up over epochs.
EPOCHS    = 30
BATCH     = 32
LR        = 5e-4
HORIZON_SCHEDULE = [1] * 5 + [5] * 10 + [10] * 15      # warm up then deepen
optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

train_t = torch.from_numpy(train_u)                    # CPU; move to GPU per batch
grid = torch.tensor(
    (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min()),
    dtype=torch.float32, device=device,
).view(1, len(x_coords), 1)

n_samples, n_time, _ = train_t.shape
torch.manual_seed(0)
for epoch in range(EPOCHS):
    H = HORIZON_SCHEDULE[min(epoch, len(HORIZON_SCHEDULE)-1)]
    perm = torch.randperm(n_samples)
    model.train()
    running, batches = 0.0, 0
    for s in range(0, n_samples, BATCH):
        idx   = perm[s:s+BATCH]
        batch = train_t[idx].to(device)                # (B, 320, 256)
        t0w   = int(torch.randint(0, n_time - 10 - H, (1,)).item())
        window = batch[:, t0w:t0w+10, :]
        target = batch[:, t0w+10:t0w+10+H, :]

        xx = window.permute(0, 2, 1).contiguous()
        bgrid = grid.expand(xx.shape[0], -1, -1)
        preds = []
        for _ in range(H):
            p = model(xx.reshape(xx.shape[0], xx.shape[1], -1),
                      bgrid, None).squeeze(-1).squeeze(-1)
            preds.append(p)
            xx = torch.cat((xx[:, :, 1:], p.unsqueeze(-1)), dim=2)
        loss = F.mse_loss(torch.stack(preds, dim=1), target)
        optim.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        running += loss.item(); batches += 1
    sched.step()
    print(f"epoch {epoch}: H={H} train_mse={running/batches:.6f} lr={sched.get_last_lr()[0]:.2e}")

# 5. Evaluate on val (READ-ONLY; nu may be used for stratified reporting only).
def weighted_val(pred, true):
    s1 = np.mean((pred[:, 10:57]   - true[:, 10:57])   ** 2)
    s2 = np.mean((pred[:, 57:105]  - true[:, 57:105])  ** 2)
    s3 = np.mean((pred[:, 105:200] - true[:, 105:200]) ** 2)
    return 0.25 * s1 + 0.25 * s2 + 0.50 * s3

with h5py.File("data/task2_val.h5", "r") as f:
    val_true_raw = f["tensor"][:, :200, :].astype(np.float32)     # (100, 200, 256)
    val_x        = f["x_coordinate"][:].astype(np.float32)        # underscores!

# Run rollout on normalized inputs, denormalize before scoring.
model.eval()
val_pred_norm = rollout_fno(
    model,
    (val_true_raw[:, :10, :] - u_mean) / u_std,
    val_x, total_steps=200, device=device, batch_size=50,
)
val_pred = val_pred_norm * u_std + u_mean
print(f"val weighted_score = {weighted_val(val_pred, val_true_raw):.6f}")

# 6. Test rollout + save. NO nu available, NO normalization stats from test.
with h5py.File("data/task2_test.h5", "r") as f:
    test_initial = f["tensor"][:].astype(np.float32)              # (1000, 10, 256)
    test_x       = f["x_coordinate"][:].astype(np.float32)

import time
_t0 = time.perf_counter()
test_pred_norm = rollout_fno(
    model, (test_initial - u_mean) / u_std, test_x,
    total_steps=200, device=device, batch_size=100,
)
inference_time = time.perf_counter() - _t0
print(f"INFERENCE_TIME={inference_time:.3f}")                       # parsed by packager
with open("task2_inference_time.txt", "w") as f:
    f.write(f"{inference_time:.3f}\n")
test_pred = test_pred_norm * u_std + u_mean
test_pred[:, :10, :] = test_initial                                # mandatory copy

assert test_pred.shape == (1000, 200, 256)
assert np.all(np.isfinite(test_pred))
assert np.max(np.abs(test_pred[:, :10, :] - test_initial)) < 1e-3
assert inference_time < 120, f"inference_time={inference_time:.1f}s exceeds 2-min cap"

with h5py.File("task2_pred.hdf5", "w") as f:
    f.create_dataset("tensor", data=test_pred.astype(np.float32))
```

### Hyperparameter notes

- **Architecture**: `width=64, modes=16` is a good starting point. Going
  to `width=96, modes=24` adds capacity for the multi-`Nu` distribution
  but doubles training time. `modes` larger than 16 sees diminishing
  returns on this resolution.
- **Epochs**: 30 cosine-annealed epochs is the floor; 50–80 is more
  competitive. Watch val score plateau and stop.
- **Horizon curriculum** (`HORIZON_SCHEDULE`): `1 → 5 → 10` is the
  minimum. Increasing the deeper-stage horizon to 20 helps `seg3` if you
  have wall-clock budget.
- **Normalization**: subtract train-set mean/std on inputs and targets,
  denormalize predictions before scoring. Skipping this can leave the
  model unable to handle the `Nu`-dependent amplitude variation.
- **`Nu` conditioning** (if you choose to add it): use
  `log10(nu)` rather than `nu` directly (range is ~[-4, -2], nicely
  centered). Normalize to `[-1, 1]` before broadcasting.

## Inference Efficiency (2-minute hard cap)

Test set is `(1000, 200, 256)` — 1000 samples × 190 model forwards = 190K
single-frame rollouts. Reference timing for the FNO above on CPU is
~55 s; with `width=96` it can climb to ~120 s. Stay safely under 2 min:

1. Run the rollout on **GPU** if available (`torch.device("cuda")`).
2. **Batch size 100** is a good default; larger if memory permits.
3. `torch.compile(model)` before rollout — typically 20–40% faster.
4. **`model.half()`** is risky for autoregressive stability; verify val
   score is unchanged before using.
5. Always time the full 1000-sample rollout once before submitting.
   Never extrapolate timing from a subset.

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `KeyError: 't-coordinate'` | Copied Task-1 loader; Task-2 keys use underscores | Use `f["t_coordinate"]`, `f["x_coordinate"]` |
| Loaded only the first part file | Forgot to concatenate part0/part1/part2 | Concatenate all three `task2_part*_train.h5` |
| Disqualification: "used pretrained weights" | Loaded a `.pt` from `tasks/ai4s-pde-task1-*/burgers_FNO/` or elsewhere | Train from scratch only; random init |
| Predictions diverge past frame ~50 | 1-step training only; no horizon curriculum | Schedule `H = 1 → 5 → 10`, keep `clip_grad_norm_=1.0` |
| First 10 frames mismatch test input | Predicted from frame 0 instead of copying | Force `test_pred[:, :10] = test_initial` after rollout |
| Inference exceeds 2 min on test | Single-sample rollout, no batching, no GPU | Batch 100+, use GPU, `torch.compile` |
| Crash trying to read `task2_test.h5["nu"]` | Tried to use test-time `nu` (which doesn't exist) | Test rollout must be `nu`-free; predict `nu` if needed |
| Val score regresses across epochs | LR too high, no scheduler, no early stop | AdamW `lr≈5e-4` + cosine annealing; checkpoint best val |
| Score worse than expected | Forgot to denormalize predictions before metric | If you normalized inputs/targets, `pred = pred*std + mean` |
| Train loss low but val score bad | Overfit to first part of trajectories (only sampled `t0=0`) | Sample `t0` uniformly in `[0, n_time-10-H)` |

## Output Files

Your script must create **two separate files in CWD**:

### 1. `task2_pred.hdf5` — prediction tensor only

```text
Required HDF5 dataset:
  tensor   float32-compatible   shape (1000, 200, 256)
```

Hard requirements:

- `tensor[:, :10, :]` must copy `data/task2_test.h5["tensor"]` within
  tolerance `1e-3`.
- `tensor[:, 10:200, :]` must contain your 190 predicted future frames.
- All values must be finite.
- Test-set `nu` is not provided and must not be assumed.
- **Do not** attach extra metadata, attributes, or auxiliary datasets.
  This file holds the prediction tensor and nothing else.

### 2. `task2_inference_time.txt` — rollout wall-clock

A plain text file containing only the floating-point seconds value
measured around the rollout call. Valid contents:

```text
42.150
```

or:

```text
INFERENCE_TIME=42.150
```

Both formats are accepted by the packager. The 2-minute hard cap is
checked against this value. Wrap **only** the `rollout_fno(...)` call
(not data loading or file writing) with `time.perf_counter()`:

```python
import time
t0 = time.perf_counter()
test_pred = rollout_fno(model, test_initial, test_x, total_steps=200,
                        device=device, batch_size=100)
inference_time = time.perf_counter() - t0
with open("task2_inference_time.txt", "w") as f:
    f.write(f"{inference_time:.3f}\n")
print(f"INFERENCE_TIME={inference_time:.3f}")
assert inference_time < 120, f"exceeds 2-min cap: {inference_time:.1f}s"
```

Do **not** create `task2_time.csv` yourself — the submission packager
assembles it post-run from `task2_inference_time.txt` plus the LLM
call-log timestamps.

## Local Validation Guidance

`task2_val.h5` is the only honest signal you have for multi-`Nu`
generalization. Strategy:

- **Always evaluate the official weighted metric** (segment weights
  25/25/50 over frames 10–57, 57–105, 105–200).
- **Stratify by `nu`** to spot regimes where the model fails. A common
  pattern: low-viscosity (`Nu < 5e-4`) samples have sharper shocks and
  are harder. If you see `seg3` errors concentrated there, increase
  training horizon or add `nu` conditioning.
- **Never use val tensors in any backward pass.** The `nu` array can be
  read for stratified metrics or for selecting model variants. The
  `tensor` array is read only for computing errors.

The official Task 2 metric scores only the 190 predicted frames:

| Segment | Forecast range (post-input frame index) | Weight |
|---|---:|---:|
| 1 | 0–47 | 25% |
| 2 | 47–95 | 25% |
| 3 | 95–190 | 50% |

## Constraints (summary)

- Train from scratch — no public or pre-existing pretrained weights.
- Do not use numerical PDE solvers in the prediction path.
- Do not use any extra data beyond the five files in `data/`.
- Do not produce a file with missing `tensor`, wrong shape, non-finite
  values, mismatched first 10 frames, or anything dependent on test-time
  `nu`.
- Stay within 12 h training wall-clock and 2 min inference.
