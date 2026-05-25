# Task 1: Fixed-Nu 1D Burgers Prediction

You are solving Task 1 of the AI4S PDE neural-operator challenge.

Your script must produce **two separate output files** in CWD:

```text
task1_pred.hdf5             # the (1000, 200, 256) prediction tensor — and nothing else
task1_inference_time.txt    # the test-set rollout wall-clock, in seconds
```

Keep them strictly separate: the HDF5 file contains only the prediction
tensor (no attributes, no extra datasets), and the txt file contains
only the elapsed-seconds number (one line). Detailed schemas are in
*Output Files* near the end of this description.

In addition, also **print** `INFERENCE_TIME=<seconds, float>` to stdout
on its own line so the value is captured even if the txt file is lost.
The submission packager prefers the txt file, falls back to stdout, and
finally re-measures locally if both are missing. Do **not** create
`task1_time.csv` yourself.

## Objective

Predict full 1D Burgers trajectories under the fixed physical setting
`Nu=0.001`.

Given the first 10 observed time steps for each test sample, generate the full
200-step trajectory.

The prediction must come from a neural model or neural-operator style method.
Do not use a numerical PDE solver to generate the prediction.

## ❗ Hard Constraints (read before writing any code)

These four rules are zero-tolerance. Violating any one of them causes the
entire submission to be scored **0** by the official grader. The automatic
judge in this pipeline also flags violators as `is_buggy=true`.

1. **`data/task1_val.hdf5` is for *evaluation only*.** Do not pass any
   tensor from this file into a training loss, optimizer step, or backward
   pass. Use it solely to compute the validation metric. Fine-tuning on
   val data is the most common way to get disqualified.
2. **`data/task1_test.hdf5` first-10-frame copy is mandatory.** Frames
   `[:, :10, :]` of `task1_pred.hdf5` must equal `task1_test.hdf5["tensor"]`
   within `1e-3`. Predict only frames `[:, 10:200, :]`.
3. **No external data.** The only training source you may use is the
   official PDEBench file `1D_Burgers_Sols_Nu0.001.hdf5` (10000 samples,
   downloaded separately from DARUS). Do not use any other dataset, any
   synthetic data you generate yourself, or `task1_val.hdf5`.
4. **No numerical PDE solvers in the prediction path.** The predictor must
   be a neural network / neural operator. No finite-difference rollouts,
   no spectral solvers, no fallback estimators.

The official training file is pre-downloaded in this workspace at
`data/1D_Burgers_Sols_Nu0.001.hdf5` (8 GB, 10000 samples). Your script
**must** fine-tune from it — zero-shot inference scores ~0.446 on val,
fine-tuning brings it to ~0.000340 (>1300× improvement). If you find your
fine-tune branch was skipped because `os.path.exists(...)` returned
`False`, you used the wrong path — the file is under `data/`, not at the
workspace root.

## Input Data

Use these local files:

```text
data/task1_test.hdf5
data/task1_val.hdf5
```

File structure:

```text
task1_test.hdf5
  tensor        float32  (1000, 10, 256)
  t-coordinate  float32  (10,)
  x-coordinate  float32  (256,)

task1_val.hdf5
  tensor        float32  (100, 200, 256)
  t-coordinate  float32  (200,)
  x-coordinate  float32  (256,)
```

`task1_test.hdf5` contains only the first 10 observed frames. You must predict
frames 10 through 199.

Two official PDEBench artifacts at `Nu=0.001`. **The names look almost the
same — they are different things:**

| File | What it is | Always in workspace? |
|---|---|---|
| `burgers_FNO/1D_Burgers_Sols_Nu0.001_FNO.pt` | trained FNO weights (~533 KB) | **yes** — use for inference and warm-start |
| `burgers_FNO/1D_Burgers_Sols_Nu0.001_Unet-PF-20.pt` | trained UNet-PF weights | yes — alternate baseline |
| `data/1D_Burgers_Sols_Nu0.001.hdf5` | raw PDEBench training data, `(10000, 201, 1024)` (~8 GB) | **yes, in this workspace** — pre-downloaded under `data/` |

All three artifacts are present in this workspace. The training file lives
under `data/`, NOT at the workspace root. Always reference it as
`data/1D_Burgers_Sols_Nu0.001.hdf5`. The agent script should still gate
the fine-tune branch on `os.path.exists("data/1D_Burgers_Sols_Nu0.001.hdf5")`
defensively (in case a future workspace omits it), but in this workspace
that check will return `True` and fine-tuning MUST run.

When using official checkpoints, keep the official resolution convention:

```text
reduced_resolution_t = 5
reduced_resolution = 4
```

That means one model time step corresponds to 5 raw PDEBench time indices, and
the 256-point grid corresponds to the 1024-point raw grid downsampled by 4.
Task-1 inputs are already on the 256-point grid, so no spatial resampling is
needed. The temporal axis you must produce (200 frames) matches the checkpoint
convention directly — each forward pass of the FNO advances exactly one of
these 200 frames.

## Workspace Layout

Your script will be executed with **CWD = task root**, but the script
itself may live in a `run/` subdirectory:

```text
<task root>             ← os.getcwd() points here
├── data/
│   ├── task1_test.hdf5
│   └── task1_val.hdf5
├── burgers_FNO/
│   └── 1D_Burgers_Sols_Nu0.001_FNO.pt
├── config.yaml
└── run/
    └── _sandbox_script_*.py    ← your script lives here, __file__.parent = run/
```

This means **`pathlib.Path(__file__).parent` ≠ task root**. Resolve every
path against `os.getcwd()` (or just use plain relative strings like
`"data/task1_test.hdf5"`), never against `__file__`.

You do **not** need to import any helper package. The entire model is
inlined in the next section — copy it into your script. There is no
`src/` to add to `sys.path`, no `ai4sv2_task1` to install.

## Domain Knowledge: Using the FNO Checkpoint

This section captures what is needed to actually load and use
`burgers_FNO/1D_Burgers_Sols_Nu0.001_FNO.pt`. Treat `Nu=0.001` as the only
valid checkpoint for Task 1; do not substitute `Nu=0.01` or other viscosity
checkpoints — they were trained on a different physical regime.

### Checkpoint shape and meaning

The state_dict has these load-bearing tensors:

- `fc0.weight` shape `(width, initial_step * num_channels + 1)` — for the
  released checkpoint this is `(20, 11)`, i.e. `width=20`, `initial_step=10`,
  `num_channels=1`, plus 1 channel for the normalized x-grid.
- `conv{0..3}.weights1` shape `(width, width, modes)` — for the released
  checkpoint this is `(20, 20, 12)`, i.e. `modes=12`.

These shapes are how you should infer hyperparameters at load time, not from
a config file.

### Loader compatibility (important)

`neuralop.models.FNO` from PyPI `neuraloperator>=2.0` is **not** state-dict
compatible with this checkpoint — the module/parameter names differ and
the v2 architecture adds a `channel_mlp` block that does not exist in the
checkpoint. Do not try to `load_state_dict` into `neuralop.models.FNO`.

Use the PDEBench-aligned implementation below. Paste it directly into
your script — it is the entire model in ~80 lines, depends only on
`torch` + `numpy`, and the matching loader infers `width`, `modes`,
`initial_step` from the checkpoint shape automatically.

```python
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path


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
    def __init__(self, num_channels=1, modes=16, width=64, initial_step=10):
        super().__init__()
        self.padding = 2
        self.fc0 = nn.Linear(initial_step * num_channels + 1, width)
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

    def forward(self, x, grid):
        x = torch.cat((x, grid), dim=-1)
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


def load_fno_checkpoint(checkpoint_path, device):
    ckpt = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    width = int(state_dict["fc0.weight"].shape[0])
    input_features = int(state_dict["fc0.weight"].shape[1])
    modes = int(state_dict["conv0.weights1"].shape[2])
    model = FNO1d(num_channels=1, modes=modes, width=width, initial_step=input_features - 1)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


@torch.no_grad()
def rollout_fno(model, initial, x_coords, t_coords, device, batch_size):
    """initial: (N, 10, 256) np.float32 → returns (N, 200, 256) np.float32."""
    n_samples, initial_step, spatial_size = initial.shape
    total_steps = len(t_coords)
    x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
    grid = torch.tensor(x_norm, dtype=torch.float32, device=device).view(1, spatial_size, 1)
    prediction = np.zeros((n_samples, total_steps, spatial_size), dtype=np.float32)
    prediction[:, :initial_step, :] = initial
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = initial[start:end]
        xx = torch.tensor(batch, dtype=torch.float32, device=device).permute(0, 2, 1).contiguous()
        batch_grid = grid.expand(end - start, -1, -1)
        frames = [batch[:, i, :] for i in range(initial_step)]
        for _ in range(initial_step, total_steps):
            pred = model(xx.reshape(xx.shape[0], xx.shape[1], -1), batch_grid).squeeze(-1).squeeze(-1)
            frames.append(pred.cpu().numpy())
            xx = torch.cat((xx[:, :, 1:], pred.unsqueeze(-1)), dim=2)
        prediction[start:end] = np.stack(frames, axis=1)
    return prediction
```

Minimum usage:

```python
import h5py, numpy as np, torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_fno_checkpoint("burgers_FNO/1D_Burgers_Sols_Nu0.001_FNO.pt", device)

with h5py.File("data/task1_test.hdf5", "r") as f:
    initial = f["tensor"][:].astype(np.float32)   # (1000, 10, 256)
    x = f["x-coordinate"][:].astype(np.float32)

import time
t_full = np.arange(200, dtype=np.float32)
_t0 = time.perf_counter()
pred = rollout_fno(model, initial, x, t_full, device, batch_size=100)  # (1000, 200, 256)
inference_time = time.perf_counter() - _t0
print(f"INFERENCE_TIME={inference_time:.3f}")
with open("task1_inference_time.txt", "w") as f:
    f.write(f"{inference_time:.3f}\n")
pred[:, :10, :] = initial   # verbatim copy required

with h5py.File("task1_pred.hdf5", "w") as f:
    f.create_dataset("tensor", data=pred, dtype="float32")
```

Do **not** wrap the checkpoint load in `try/except` and fall back to a
non-neural extrapolator on failure — that produces output with the right
shape but a score around 29 (vs. ≈ 0.44 for the real model) and fails the
"neural model only" constraint. If `load_fno_checkpoint` raises, fix the
path and rerun.

### Complete end-to-end example (load → optional fine-tune → predict → save)

Below is one runnable script. It assumes you already pasted the `FNO1d`,
`load_fno_checkpoint`, `rollout_fno` block above. It does, in order:

1. Load the pretrained checkpoint.
2. Evaluate zero-shot on `task1_val.hdf5` to get a baseline.
3. If a training file `1D_Burgers_Sols_Nu0.001.hdf5` is available in the
   runtime (PDEBench format, full `(N, 200, 1024)`), fine-tune the model.
4. Re-evaluate on val, keep the better of {zero-shot, fine-tuned}.
5. Roll out on `task1_test.hdf5` and write `task1_pred.hdf5`.

```python
import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy


# ---------- helpers ----------------------------------------------------------

def weighted_val_score(pred, true):
    """Official-style metric on (N, 200, 256) arrays."""
    seg1 = np.mean((pred[:, 10:57]   - true[:, 10:57])   ** 2)
    seg2 = np.mean((pred[:, 57:105]  - true[:, 57:105])  ** 2)
    seg3 = np.mean((pred[:, 105:200] - true[:, 105:200]) ** 2)
    return 0.25 * seg1 + 0.25 * seg2 + 0.50 * seg3


def load_training_data(path, reduced_t=5, reduced_x=4):
    """PDEBench-style 1D Burgers: raw shape (N, 200, 1024) → (N, 40, 256)."""
    with h5py.File(path, "r") as f:
        u = f["tensor"][:].astype(np.float32)
        x = f["x-coordinate"][:].astype(np.float32)
    u = u[:, ::reduced_t, ::reduced_x]
    x = x[::reduced_x]
    return u, x


# ---------- 1. load pretrained ----------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = "burgers_FNO/1D_Burgers_Sols_Nu0.001_FNO.pt"
model = load_fno_checkpoint(ckpt_path, device)
print(f"loaded {ckpt_path} | device={device}")

# ---------- 2. zero-shot val baseline ---------------------------------------

with h5py.File("data/task1_val.hdf5", "r") as f:
    val_true = f["tensor"][:].astype(np.float32)               # (100, 200, 256)
    val_x    = f["x-coordinate"][:].astype(np.float32)
    val_t    = f["t-coordinate"][:].astype(np.float32)

val_pred_zs = rollout_fno(
    model, val_true[:, :10, :], val_x, val_t, device, batch_size=50
)
score_zs = weighted_val_score(val_pred_zs, val_true)
print(f"zero-shot weighted val score: {score_zs:.6f}")

best_state = deepcopy(model.state_dict())
best_score = score_zs

# ---------- 3. optional fine-tune -------------------------------------------

train_path = "data/1D_Burgers_Sols_Nu0.001.hdf5"   # pre-downloaded under data/
if os.path.exists(train_path):
    print(f"found {train_path} → fine-tuning")
    train_u, train_x = load_training_data(train_path)         # (N, 40, 256)
    train_u = torch.from_numpy(train_u).to(device)
    grid = torch.tensor(
        (train_x - train_x.min()) / (train_x.max() - train_x.min()),
        dtype=torch.float32, device=device,
    ).view(1, len(train_x), 1)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    n_epochs = 5
    batch_size = 32
    horizon = 5                                                # short rollout
    n_samples = train_u.shape[0]

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples, device=device)
        running = 0.0
        for s in range(0, n_samples, batch_size):
            idx = perm[s : s + batch_size]
            batch = train_u[idx]                              # (B, 40, 256)
            # random start so we cover all temporal regimes
            t0 = int(torch.randint(0, batch.shape[1] - 10 - horizon, (1,)).item())
            window = batch[:, t0 : t0 + 10, :].clone()         # (B, 10, 256)
            target = batch[:, t0 + 10 : t0 + 10 + horizon, :]  # (B, horizon, 256)

            xx = window.permute(0, 2, 1).contiguous()          # (B, 256, 10)
            batch_grid = grid.expand(xx.shape[0], -1, -1)
            preds = []
            for _ in range(horizon):
                pred = model(
                    xx.reshape(xx.shape[0], xx.shape[1], -1), batch_grid
                ).squeeze(-1).squeeze(-1)                      # (B, 256)
                preds.append(pred)
                xx = torch.cat((xx[:, :, 1:], pred.unsqueeze(-1)), dim=2)

            pred_stack = torch.stack(preds, dim=1)              # (B, horizon, 256)
            loss = F.mse_loss(pred_stack, target)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += loss.item() * idx.shape[0]
        avg = running / n_samples
        print(f"  epoch {epoch}: train_mse={avg:.6f}")

    # ---------- 4. keep best on val -----------------------------------------
    val_pred_ft = rollout_fno(
        model, val_true[:, :10, :], val_x, val_t, device, batch_size=50
    )
    score_ft = weighted_val_score(val_pred_ft, val_true)
    print(f"fine-tuned weighted val score: {score_ft:.6f}")
    if score_ft < best_score:
        best_score = score_ft
        best_state = deepcopy(model.state_dict())
        print("  → keeping fine-tuned weights")
    else:
        print("  → fine-tune did not improve; reverting")
        model.load_state_dict(best_state)
else:
    print(f"{train_path} not available → using zero-shot weights")

# ---------- 5. predict test + save ------------------------------------------

with h5py.File("data/task1_test.hdf5", "r") as f:
    test_initial = f["tensor"][:].astype(np.float32)          # (1000, 10, 256)
    test_x       = f["x-coordinate"][:].astype(np.float32)

import time
t_full = np.arange(200, dtype=np.float32)
_t0 = time.perf_counter()
test_pred = rollout_fno(model, test_initial, test_x, t_full, device, batch_size=100)
inference_time = time.perf_counter() - _t0
print(f"INFERENCE_TIME={inference_time:.3f}")
with open("task1_inference_time.txt", "w") as f:
    f.write(f"{inference_time:.3f}\n")
test_pred[:, :10, :] = test_initial                            # mandatory copy

assert test_pred.shape == (1000, 200, 256)
assert np.all(np.isfinite(test_pred))
assert np.max(np.abs(test_pred[:, :10, :] - test_initial)) < 1e-3

with h5py.File("task1_pred.hdf5", "w") as f:
    f.create_dataset("tensor", data=test_pred.astype(np.float32))

print(f"saved task1_pred.hdf5 + task1_inference_time.txt | best val score: {best_score:.6f} | inference_time: {inference_time:.3f}s")
```

Notes about this example:

- It is intentionally minimal — no AutoML loop, no spectral loss, no
  multi-horizon curriculum. Treat it as the **starting baseline** and add
  the items from *AutoML Hyperparameter Search* on top.
- Training data is at `data/1D_Burgers_Sols_Nu0.001.hdf5` in this workspace
  (8 GB, pre-downloaded). The bare-root path `"1D_Burgers_Sols_Nu0.001.hdf5"`
  will NOT find it — always prefix with `data/`.
- `task1_val.hdf5` is used **only** for selecting between zero-shot and
  fine-tuned weights, never as training data.
- The script must not catch the checkpoint-loading exception. If the load
  fails, the script should crash so you notice — never silently produce
  predictions from a fallback estimator.

### Rollout convention

- Input window: the last 10 frames, packed as `(batch, 256, 10)` for the
  spatial-then-time layout `FNO1d.forward` expects.
- One forward pass produces one new frame; concatenate it to the window,
  drop the oldest frame, and repeat until 200 frames exist.
- The first 10 output frames must be a verbatim copy of
  `task1_test.hdf5["tensor"]` (tolerance `1e-3`); only frames 10..199 come
  from rollout.
- Use the normalized x-grid `(x - x.min()) / (x.max() - x.min())` as the
  extra input channel, matching the loader.

## Training and Fine-tuning Strategy

The released checkpoint is a strong starting point but is not specialized to
the Task-1 test distribution. Two regimes are allowed:

1. **Zero-shot rollout.** Load the checkpoint with `load_fno_checkpoint` and
   roll out directly on `task1_test.hdf5`. Measured `weighted_score ≈
   0.4457` on `task1_val.hdf5` (baseline).
2. **Fine-tune from checkpoint.** Initialize `FNO1d` from
   `load_fno_checkpoint`, then continue training on the official
   `1D_Burgers_Sols_Nu0.001.hdf5` data (downsampled with
   `reduced_resolution_t=5`, `reduced_resolution=4`). With the recipe
   below, measured `weighted_score ≈ 0.000340` — a **>1300×** improvement
   over zero-shot, finishing in ~70 s on CPU.

Use `task1_val.hdf5` only for model selection, never as training data
(see Hard Constraints).

### Verified fine-tune recipe (empirically measured on this workspace)

The recipe below has been run end-to-end on this exact workspace. Numbers
are reproducible:

| Metric on `task1_val.hdf5` | Before | After |
|---|---:|---:|
| `weighted_score` | 0.4457 | **0.000340** |
| `seg1 MSE` (frames 10–57)    | 0.0471 | 0.0003 |
| `seg2 MSE` (frames 57–105)   | 0.2113 | 0.0003 |
| `seg3 MSE` (frames 105–200)  | 0.7623 | 0.0004 |
| L2 norm at frame 199 (pred / true) | 193.5 / 91.9 (**diverging**) | 92.1 / 91.9 |

Most of the score comes from the long-horizon segment (`seg3`, 50%
weight). The pretrained checkpoint **diverges past ~50 frames** because
it was trained with 1-step loss only — error accumulates exponentially.
A short multi-step rollout loss fixes the stability and shrinks all three
segments together.

The full procedure:

```python
import time
from copy import deepcopy

import h5py
import numpy as np
import torch
import torch.nn.functional as F

# 1. Load pretrained checkpoint (assumes FNO1d/load_fno_checkpoint/rollout_fno
#    have been pasted from the section above).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = load_fno_checkpoint("burgers_FNO/1D_Burgers_Sols_Nu0.001_FNO.pt", device)

# 2. Baseline on val (READ-ONLY use of task1_val).
with h5py.File("data/task1_val.hdf5", "r") as f:
    val_true = f["tensor"][:].astype(np.float32)        # (100, 200, 256)
    val_x    = f["x-coordinate"][:].astype(np.float32)
    val_t    = f["t-coordinate"][:].astype(np.float32)

def weighted_val(pred, true):
    s1 = np.mean((pred[:, 10:57]   - true[:, 10:57])   ** 2)
    s2 = np.mean((pred[:, 57:105]  - true[:, 57:105])  ** 2)
    s3 = np.mean((pred[:, 105:200] - true[:, 105:200]) ** 2)
    return 0.25 * s1 + 0.25 * s2 + 0.50 * s3

baseline_pred = rollout_fno(model, val_true[:, :10, :], val_x, val_t, device, batch_size=50)
baseline_score = weighted_val(baseline_pred, val_true)
print(f"baseline val weighted={baseline_score:.6f}")
best_state, best_score = deepcopy(model.state_dict()), baseline_score

# 3. Load PDEBench training data and downsample to the checkpoint's resolution.
#    Raw shape is (10000, 201, 1024). Take frames [0:200] strided by 5 → 40
#    frames (this matches reduced_resolution_t=5), and spatial stride 4 → 256.
with h5py.File("data/1D_Burgers_Sols_Nu0.001.hdf5", "r") as f:
    train = f["tensor"][:, :200:5, ::4].astype(np.float32)   # (10000, 40, 256)
    train_x = f["x-coordinate"][::4].astype(np.float32)      # (256,)
train_u = torch.from_numpy(train)
n_samples, n_time, _ = train_u.shape

# 4. Fine-tune with a 5-step rollout loss. Hyperparameters that worked:
EPOCHS, BATCH, LR, HORIZON = 2, 32, 1e-4, 5

grid = torch.tensor(
    (train_x - train_x.min()) / (train_x.max() - train_x.min()),
    dtype=torch.float32, device=device,
).view(1, train_x.shape[0], 1)
optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

model.train()
torch.manual_seed(0)
for epoch in range(EPOCHS):
    perm = torch.randperm(n_samples)
    running, batches = 0.0, 0
    t0 = time.perf_counter()
    for s in range(0, n_samples, BATCH):
        idx   = perm[s:s + BATCH]
        batch = train_u[idx].to(device)                        # (B, 40, 256)
        t0w   = int(torch.randint(0, n_time - 10 - HORIZON, (1,)).item())
        window = batch[:, t0w:t0w + 10, :]                      # (B, 10, 256)
        target = batch[:, t0w + 10:t0w + 10 + HORIZON, :]       # (B, h, 256)

        xx = window.permute(0, 2, 1).contiguous()               # (B, 256, 10)
        bgrid = grid.expand(xx.shape[0], -1, -1)
        preds = []
        for _ in range(HORIZON):
            p = model(xx.reshape(xx.shape[0], xx.shape[1], -1),
                      bgrid).squeeze(-1).squeeze(-1)            # (B, 256)
            preds.append(p)
            xx = torch.cat((xx[:, :, 1:], p.unsqueeze(-1)), dim=2)

        loss = F.mse_loss(torch.stack(preds, dim=1), target)
        optim.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # CRITICAL
        optim.step()
        running += loss.item(); batches += 1
    print(f"epoch {epoch}: train_mse={running/batches:.6f}  ({time.perf_counter()-t0:.1f}s)")
model.eval()

# 5. Re-evaluate and keep the better weights.
ft_pred  = rollout_fno(model, val_true[:, :10, :], val_x, val_t, device, batch_size=50)
ft_score = weighted_val(ft_pred, val_true)
print(f"fine-tuned val weighted={ft_score:.6f}  delta={ft_score - baseline_score:+.6f}")
if ft_score < best_score:
    best_score, best_state = ft_score, deepcopy(model.state_dict())
model.load_state_dict(best_state)

# 6. Predict the test set + persist inference_time. The rollout call is the
#    ONLY thing wrapped by time.perf_counter() — not data loading, not the
#    file write. Write inference_time to its own dedicated txt file (do not
#    embed it in task1_pred.hdf5 as an attribute).
with h5py.File("data/task1_test.hdf5", "r") as f:
    test_initial = f["tensor"][:].astype(np.float32)              # (1000, 10, 256)
    test_x       = f["x-coordinate"][:].astype(np.float32)

t_full = np.arange(200, dtype=np.float32)
_t0 = time.perf_counter()
test_pred = rollout_fno(model, test_initial, test_x, t_full, device, batch_size=100)
inference_time = time.perf_counter() - _t0
print(f"INFERENCE_TIME={inference_time:.3f}")
with open("task1_inference_time.txt", "w") as f:
    f.write(f"{inference_time:.3f}\n")
test_pred[:, :10, :] = test_initial                                # mandatory copy

assert test_pred.shape == (1000, 200, 256)
assert np.all(np.isfinite(test_pred))
assert np.max(np.abs(test_pred[:, :10, :] - test_initial)) < 1e-3

with h5py.File("task1_pred.hdf5", "w") as f:
    f.create_dataset("tensor", data=test_pred.astype(np.float32))
```

Five things that make this recipe work — drop any one and the score
regresses sharply:

- **Data prep `[:, :200:5, ::4]`.** Raw is `(10000, 201, 1024)`; you want
  `(10000, 40, 256)` so each model step equals one val frame. Skipping the
  `[:200]` slice leaves you with 41 frames and shifts the time alignment.
- **5-step rollout loss, not 1-step.** Pure 1-step training over-fits to
  next-frame accuracy and the model diverges past frame ~50 at inference.
- **`torch.nn.utils.clip_grad_norm_(., 1.0)`.** Autoregressive gradients
  explode silently otherwise — you may see `train_mse` look fine while val
  predictions blow up.
- **Random start `t0w`.** Sampling the 10-frame window from anywhere in
  `[0, n_time - 10 - HORIZON)` exposes the model to all temporal regimes
  of the trajectory, not just the initial transient.
- **`lr=1e-4`, `weight_decay=1e-5`, AdamW.** Higher LR (e.g. `1e-3`)
  blows up; lower LR (`1e-5`) needs many more epochs.

After fine-tuning, always **keep the better of {zero-shot, fine-tuned}**
based on val score (`best_state` above). Never overwrite the working
baseline blindly.

### Practical engineering notes

The following items are not optional details — each one of them has shown up
as a failure mode on prior runs. They apply to both full fine-tuning and the
AutoML inner-loop trainer below.

- **Normalization.** Compute `mean`/`std` from the *training* split only.
  Normalize inputs and targets, and denormalize predictions before computing
  any physically meaningful metric (forecast MSE/MAE on val).
- **Rollout curriculum.** Start with 1-step teacher forcing for a few warm-up
  epochs, then increase the unrolled horizon (1 → 5 → 20 → 50). Pure
  1-step training over-fits to next-frame accuracy and explodes on the
  95..190 segment.
- **Loss mix.** A useful default is
  `loss = α · relative_MSE + β · spectral_MSE`, where the spectral term is
  MSE between `|rfft(pred)|²` and `|rfft(true)|²` along the spatial axis.
  The spectral term protects the long-horizon segment without hurting the
  short one. Tune `α, β` inside the AutoML search.
- **Optimizer.** AdamW with `lr≈1e-3`, `weight_decay≈1e-4`, cosine schedule
  over the budget. Use mixed precision (`torch.cuda.amp`) when training on
  GPU.
- **Gradient clipping.** Always apply `clip_grad_norm_(model.parameters(),
  1.0)`. Autoregressive rollouts diverge silently without it.
- **Seed ensembling.** Training 3–5 models with different seeds and
  averaging their rollouts typically improves the weighted score by a few
  percent at near-zero engineering cost; do this after the AutoML pick, not
  during the search.

### Inference efficiency

The test set is `(1000, 200, 256)`. The reference rollout via
`rollout_fno` finishes the 100-sample val set in ≈10s on a single GPU, so
the 1000-sample test set is expected to fit comfortably in budget. If a
run is slow, in order of typical impact:

1. Increase the rollout batch size until GPU memory is the limit.
2. `torch.compile(model)` after `load_fno_checkpoint`.
3. `model.half()` at inference, after verifying val score is unchanged.

Always run the full 1000-sample rollout end-to-end before submitting —
never extrapolate timing from a subset.

### Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| First 10 frames mismatch test input | Predicted from frame 0 instead of copying | Force `tensor[:, :10] = test_tensor` after rollout |
| Long-horizon (95..190) MSE blows up | 1-step-only training, no spectral term | Rollout curriculum + spectral loss |
| Loss goes NaN mid-training | Exploding autoregressive gradient | Add `clip_grad_norm_(., 1.0)`, lower LR, shorten rollout |
| Val score regresses after a few epochs | Overfit to training distribution | Stronger weight decay, early stop on val |
| `No module named 'ai4sv2_task1'` | Looking for `src/` next to `__file__` while the runner executed your script from a `run/` subdirectory | Resolve `src/` from CWD: `sys.path.insert(0, os.path.join(os.getcwd(), "src"))` |
| Loader error on `1D_Burgers_Sols_Nu0.001_FNO.pt` | Using `neuralop.models.FNO` | Use `load_fno_checkpoint` from `src/ai4sv2_task1/models/fno.py` |
| Output passes shape/finite checks but score is poor (`weighted_score≈29`) | Silent fallback to a non-neural extrapolator when the FNO import failed | Make the FNO load mandatory — raise instead of falling back; rerun after fixing the import path |
| Judge marks the submission `is_buggy=true` with "fine-tuned on validation set" or similar | The code used `data/task1_val.hdf5` (or its tensors) inside a training loop | `task1_val.hdf5` is **eval-only**. Remove it from any loss / optimizer step. If no PDEBench training file is present, do zero-shot rollout instead |
| Stdout says `... not available, using zero-shot weights` and val score stuck at `0.4457` | The code checked `os.path.exists("1D_Burgers_Sols_Nu0.001.hdf5")` (bare path) instead of `data/1D_Burgers_Sols_Nu0.001.hdf5` | The training file is under `data/` in this workspace. Use `train_path = "data/1D_Burgers_Sols_Nu0.001.hdf5"`. After the fix, val score should drop to ~0.000340 |
| Predictions on wrong physical regime | Loaded `Nu=0.01` checkpoint by mistake | Use `1D_Burgers_Sols_Nu0.001_FNO.pt` only |

## AutoML Hyperparameter Search

Treat hyperparameter selection as an AutoML problem rather than hand tuning.
The objective and search space are fixed; the search algorithm is left to
the implementer (random search, Bayesian / TPE via Optuna, Hyperband,
population-based — any of these is acceptable).

### Objective

Minimize the official weighted forecast metric evaluated on
`task1_val.hdf5`:

```text
score = 0.25 * MSE(frames  10..57)
      + 0.25 * MSE(frames  57..105)
      + 0.50 * MSE(frames 105..200)
```

Report `forecast_mse` and `forecast_mae` on val for every trial; pick the
trial with the lowest weighted score, and break ties by long-horizon
(frames 105..200) MSE.

### Search space

Architecture is fixed to the checkpoint shape (`width=20`, `modes=12`,
4 spectral layers, `initial_step=10`) so that warm-start from
`1D_Burgers_Sols_Nu0.001_FNO.pt` stays valid. The search is over training
dynamics only:

| Group | Hyperparameter | Range / choices |
|---|---|---|
| Optim | learning rate | log-uniform `[1e-5, 5e-3]` |
| Optim | weight decay | log-uniform `[1e-8, 1e-3]` |
| Optim | scheduler | {cosine, step, plateau} |
| Optim | batch size | {16, 32, 64} |
| Rollout | teacher-forcing ratio | uniform `[0.0, 1.0]`, optionally annealed |
| Rollout | training horizon | {1, 5, 20, 50} frames per loss |
| Rollout | multi-horizon loss weights | simplex over the horizons above |
| Data | x-grid normalization | {min-max to [0,1], standardize, raw} |

### Search protocol

- Warm-start from the released checkpoint whenever `start from checkpoint =
  True`; this dominates training from scratch within a small budget.
- Use a short proxy budget per trial (e.g. ≤ 5 epochs, subset of training
  data) for the search phase; promote the top-k configs to a full
  fine-tuning run.
- Cap total wall-clock for the search so it stays reproducible inside the
  task runtime.
- Persist every trial's val score so the final pick is auditable.

## Output Files

Your script must create **two separate files in CWD**:

### 1. `task1_pred.hdf5` — prediction tensor only

```text
Required HDF5 dataset:
  tensor   float32-compatible   shape (1000, 200, 256)
```

Hard requirements:

- `tensor[:, :10, :]` must copy `data/task1_test.hdf5["tensor"]` within
  tolerance `1e-3`.
- `tensor[:, 10:200, :]` must contain your 190 predicted future frames.
- All values must be finite.
- **Do not** attach extra metadata, attributes, or auxiliary datasets to
  this file. It holds the prediction tensor and nothing else.

### 2. `task1_inference_time.txt` — rollout wall-clock

A plain text file containing **only the floating-point seconds** value
measured around the test-set rollout call. Examples of valid contents:

```text
55.832
```

or, equivalently:

```text
INFERENCE_TIME=55.832
```

Both formats are accepted by the submission packager. Wrap **only** the
`rollout_fno(...)` call (not data loading, checkpoint loading, or file
writing) with `time.perf_counter()`. Write the value out immediately
after the rollout:

```python
import time
t0 = time.perf_counter()
test_pred = rollout_fno(model, test_initial, test_x, t_full, device, batch_size=100)
inference_time = time.perf_counter() - t0
with open("task1_inference_time.txt", "w") as f:
    f.write(f"{inference_time:.3f}\n")
print(f"INFERENCE_TIME={inference_time:.3f}")
```

Do **not** create `task1_time.csv` yourself — the submission packager
assembles it post-run from `task1_inference_time.txt` plus the LLM call
log timestamps.

## Local Validation Guidance

Use `task1_val.hdf5` to validate model behavior before predicting the test set.
The scoring emphasis is short-term accuracy plus medium/long-term stability.

For the official Task 1 metric, the 190 predicted frames are split into:

| Segment | Forecast range | Weight |
|---|---:|---:|
| 1 | 0-47 | 25% |
| 2 | 47-95 | 25% |
| 3 | 95-190 | 50% |

The first 10 frames are not forecasted; they are only checked for exact copying.

## Constraints

- Do not use numerical solvers to generate predictions.
- Do not use extra data outside the allowed Task 1 resources.
- Do not produce a file with missing `tensor`, wrong shape, non-finite values,
  or mismatched first 10 frames.
