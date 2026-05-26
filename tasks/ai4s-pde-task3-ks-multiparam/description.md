# Task 3: Kuramoto-Sivashinsky Multi-Parameter Prediction

You are solving Task 3, the additional KS-equation task in the AI4S PDE
neural-operator challenge.

Your script must produce **two separate output files** in CWD:

```text
task3_pred.hdf5             # the (100, 400, 256) prediction tensor — and nothing else
task3_inference_time.txt    # the test-set rollout wall-clock, in seconds
```

Keep them strictly separate: the HDF5 file contains only the prediction
tensor (no attributes, no extra datasets), and the txt file contains
only the elapsed-seconds number (one line, e.g. `42.150`).

In addition, also **print** `INFERENCE_TIME=<seconds, float>` to stdout
on its own line. The submission packager prefers the txt file, falls
back to stdout, and re-measures locally as last resort. Do **not**
create `task3_time.csv` yourself.

The 2-minute hard cap is checked against this measured value, so the
agent should `assert inference_time < 120` and, if it fails, rerun with
a faster configuration (larger batch, smaller model, GPU).

## Objective

Predict a full 400-step Kuramoto-Sivashinsky trajectory from the **first
20 observed time steps**.

The KS equation:

```text
u_t + u·u_x + λ₂·u_xx + u_xxxx = 0
```

is one of the canonical chaotic PDEs. Errors grow **exponentially** with
time (the Lyapunov time on this setup is roughly 10–20 model steps), so
exact long-horizon prediction is physically impossible past that point;
the scoring rule reflects this by switching to a Lorentzian / Fréchet
distance for the late segment.

Training and validation data carry a per-sample `lambda2` value. The test
data does **not**. At test time, predict from the first 20 observed
frames only.

The prediction must come from a neural model or neural-operator style
method. Do not use a numerical PDE solver to generate the prediction.

## ❗ Hard Constraints (read before writing any code)

Four zero-tolerance rules. Violating any one of them causes the entire
submission to be scored **0** by the official grader.

1. **Train from scratch — no pretrained checkpoints, no public weights.**
   Do not load `*.pt` files from PDEBench, HuggingFace, Task-1's checkpoint
   directory, or any other source. Initialize every parameter randomly and
   train only on `data/KS_train.hdf5`.
2. **`data/KS_val.hdf5` is for evaluation only.** Do not pass any tensor
   from this file into a training loss, optimizer step, or backward pass.
   Validation `lambda2` may be read for stratified metrics but not for
   gradients.
3. **`data/KS_test.hdf5` first-20-frame copy is mandatory.** Frames
   `[:, :20, :]` of `task3_pred.hdf5` must equal `KS_test.hdf5["tensor"]`
   within `1e-3`. Predict only frames `[:, 20:400, :]`.
4. **Test-time inference is `λ₂`-free.** `KS_test.hdf5` does not contain
   `lambda2`. Your inference path may infer `λ₂` from the 20-frame
   prefix or ignore it entirely, but it must not assume `λ₂` is given.

Time/compute limits enforced by the grader:

- Total training wall-clock (Agent thinking + actual training) ≤ **12 h**.
- Inference time on the full test set ≤ **2 min**. Exceeding it scores 0.

## Physical Setting

| Quantity | Value |
|---|---|
| Spatial grid points `N` | 256 |
| Spatial domain | `x ∈ [0, ~39.84]`, periodic, `dx ≈ 0.156` |
| Stored time step | `dt = 0.5` |
| Total stored steps | 400 (`t ∈ [0, 199.5]`) |
| Diffusion coefficient `λ₂` | `Uniform[1.0, 1.5]` |
| Lyapunov time (rough) | ~ 10–20 model steps |

`λ₂` controls the balance between the energy-injecting `u_xx` term and
the high-wavenumber-damping `u_xxxx` term. Smaller `λ₂` → less
hyperdiffusion → more chaos. The full range `[1.0, 1.5]` covers
distinct dynamical regimes.

## Input Data

```text
data/KS_train.hdf5
data/KS_val.hdf5
data/KS_test.hdf5
```

HDF5 schemas (note **`t-coordinate` / `x-coordinate` with HYPHENS** — Task 2
used underscores; this task is back to hyphens like Task 1):

```text
KS_train.hdf5
  tensor        float32  (2000, 400, 256)    # 2000 trajectories
  lambda2       float32  (2000,)             # per-sample diffusion coefficient
  t-coordinate  float32  (400,)              # 0.0, 0.5, ..., 199.5
  x-coordinate  float32  (256,)

KS_val.hdf5
  tensor        float32  (100, 400, 256)
  lambda2       float32  (100,)
  t-coordinate  float32  (400,)
  x-coordinate  float32  (256,)

KS_test.hdf5
  tensor        float32  (100, 20, 256)      # only first 20 frames
  t-coordinate  float32  (20,)
  x-coordinate  float32  (256,)
  (no `lambda2` field)
```

`KS_test.hdf5` contains only the first 20 observed frames. You must
predict frames 20 through 399.

## Workspace Layout

Your script will be executed with **CWD = task root**, but the script
itself may live in a `run/` subdirectory:

```text
<task root>             ← os.getcwd() points here
├── data/
│   ├── KS_train.hdf5
│   ├── KS_val.hdf5
│   └── KS_test.hdf5
├── config.yaml
└── run/
    └── _sandbox_script_*.py    ← your script lives here, __file__.parent = run/
```

Resolve every path against `os.getcwd()` or use plain relative strings
(`"data/KS_train.hdf5"`), never against `__file__`. There is no `src/`
directory — the model code is inlined below; copy it into your script.

## Domain Knowledge

### Why this task is hard

| | Task 1 / Task 2 (Burgers) | Task 3 (KS) |
|---|---|---|
| PDE order | second-order, dissipative | **fourth-order**, includes `u_xxxx` |
| Long-time behavior | smooth shock + decay | **persistent spatio-temporal chaos** |
| Error growth | linear/sublinear past short times | **exponential** (Lyapunov-bounded predictability) |
| Forecast horizon | 190 steps | **380 steps** |
| Input window | 10 frames | **20 frames** (`initial_step=20`) |
| Test samples | 1000 | **100** |
| `seg3` scoring | MSE | `max(Lorentzian, Fréchet)` — **distribution-matching** |

The `seg3` (steps 200–399, weight 50%) scoring is the critical
difference. After the Lyapunov time, exact pointwise prediction
diverges from the truth, so the grader uses statistical metrics:
- **Lorentzian** rewards RMSE close to a reference scale
- **Fréchet** rewards matching the empirical *distribution* of states
The model wins by producing trajectories that **look statistically like
KS at the right `λ₂`**, even if pointwise off.

### `λ₂` strategy: condition or not?

Same three-option menu as Task 2:

1. **`λ₂`-agnostic FNO.** Ignore `lambda2`. The model learns to identify
   the regime from the 20-frame input. Simplest; works well because the
   20-frame window already contains rich dynamical information.
2. **Inferred-`λ₂` conditioning.** Train a tiny auxiliary head to
   predict `λ₂` from the 20-frame prefix (supervised by the known
   `lambda2` labels). At test time, run this head first, then condition
   the main FNO on the estimate.
3. **`λ₂`-as-extra-channel.** Broadcast `λ₂` (rescaled to `[-1, 1]`) to
   a `(256,)` spatial map and feed as an extra input channel. Most
   powerful but most fragile — bad `λ₂` estimates poison every spatial
   feature.

Start with **option #1** — the 20-frame window is plenty informative,
and the chaotic dynamics mean an inaccurate `λ₂` estimate can hurt more
than no estimate at all. Escalate to #2 / #3 only if `λ₂`-stratified
val metrics show clear per-regime degradation.

### Inline FNO model (paste into your script)

Same FNO1d as Task 1/2 but with `initial_step=20`. You may freely
modify it (deeper, wider, `λ₂` conditioning, etc.) — training from
scratch removes any state-dict constraint.

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
    """Set extra_channels=0 for lambda2-agnostic. Set =1 for lambda2 broadcast."""
    def __init__(self, num_channels=1, modes=32, width=96, initial_step=20, extra_channels=0):
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

Note `modes=32` (vs 12/16 for Burgers) — KS dynamics live at higher
spatial frequencies because of the `u_xxxx` term; the stronger baseline uses
more Fourier modes and a wider hidden state while still fitting the 2-minute
inference cap.

### Rollout helper

```python
@torch.no_grad()
def rollout_ks(model, initial, x_coords, total_steps, device, batch_size,
               extras_per_sample=None):
    """
    initial: (N, 20, 256) np.float32.
    extras_per_sample: optional (N, extra_channels) for lambda2 conditioning;
        broadcast to (N, 256, extra_channels) inside.
    Returns (N, total_steps, 256) np.float32, with the first 20 frames copied
    from `initial`.
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
import time
import math
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load training data.
with h5py.File("data/KS_train.hdf5", "r") as f:
    train_u  = f["tensor"][:].astype(np.float32)                  # (2000, 400, 256)
    train_lambda2 = f["lambda2"][:].astype(np.float32)             # (2000,)
    x_coords = f["x-coordinate"][:].astype(np.float32)             # NOTE: hyphens, like Task 1; Task 2 used underscores
print(f"train: {train_u.shape}, lambda2 range [{train_lambda2.min():.3f}, {train_lambda2.max():.3f}]")

# 2. Normalize (compute stats on train ONLY). KS amplitudes go [-7, 7]ish.
u_mean, u_std = float(train_u.mean()), float(train_u.std())
train_u = (train_u - u_mean) / u_std

# 3. Build model — lambda2-agnostic baseline.
model = FNO1d(num_channels=1, modes=32, width=96, initial_step=20,
              extra_channels=0).to(device)
print(f"params={sum(p.numel() for p in model.parameters()):,}")


def spectral_loss(pred, target):
    """Log-power spectrum loss along x; helps the long-horizon seg3 score."""
    pred_power = torch.abs(torch.fft.rfft(pred, dim=-1)) ** 2
    target_power = torch.abs(torch.fft.rfft(target, dim=-1)) ** 2
    return F.mse_loss(torch.log1p(pred_power), torch.log1p(target_power))


def gradient_loss(pred, target):
    """First-difference loss along x; cheap spatial-structure regularizer."""
    return F.mse_loss(pred[..., 1:] - pred[..., :-1], target[..., 1:] - target[..., :-1])


def init_ema(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def update_ema(model, ema_state, decay=0.995):
    for k, v in model.state_dict().items():
        if torch.is_floating_point(v) or torch.is_complex(v):
            ema_state[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)
        else:
            ema_state[k].copy_(v)

# 4. Train. The 400-frame trajectories are plenty for rollout-window sampling.
#    KS is chaotic — push the rollout horizon up aggressively to expose the
#    model to long-tail dynamics.
EPOCHS  = 40
BATCH   = 16              # smaller than Burgers because 400-frame tensors are bigger
LR      = 5e-4
SPECTRAL_W = 0.20
GRAD_W     = 0.05
# Curriculum: warm up with short rollouts, then deepen.
HORIZON_SCHEDULE = [1]*5 + [5]*10 + [10]*10 + [20]*10 + [40]*5

optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
ema_state = init_ema(model)

train_t = torch.from_numpy(train_u)
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
        batch = train_t[idx].to(device)                  # (B, 400, 256)
        t0w   = int(torch.randint(0, n_time - 20 - H, (1,)).item())
        window = batch[:, t0w:t0w+20, :]
        target = batch[:, t0w+20:t0w+20+H, :]

        xx = window.permute(0, 2, 1).contiguous()        # (B, 256, 20)
        bgrid = grid.expand(xx.shape[0], -1, -1)
        preds = []
        for _ in range(H):
            p = model(xx.reshape(xx.shape[0], xx.shape[1], -1),
                      bgrid, None).squeeze(-1).squeeze(-1)
            preds.append(p)
            xx = torch.cat((xx[:, :, 1:], p.unsqueeze(-1)), dim=2)
        pred_stack = torch.stack(preds, dim=1)
        loss = (
            F.mse_loss(pred_stack, target)
            + SPECTRAL_W * spectral_loss(pred_stack, target)
            + GRAD_W * gradient_loss(pred_stack, target)
        )
        optim.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        update_ema(model, ema_state)
        running += loss.item(); batches += 1
    sched.step()
    print(f"epoch {epoch}: H={H} train_mse={running/batches:.6f} lr={sched.get_last_lr()[0]:.2e}")

# 5. Evaluate on val (READ-ONLY).
def weighted_mse_proxy(pred, true):
    s1 = np.mean((pred[:, 20:50]   - true[:, 20:50])   ** 2)
    s2 = np.mean((pred[:, 50:200]  - true[:, 50:200])  ** 2)
    s3 = np.mean((pred[:, 200:400] - true[:, 200:400]) ** 2)
    return 0.25 * s1 + 0.25 * s2 + 0.50 * s3


def official_like_segment_score(pred, true):
    """Task-3 public formula with Lorentzian-only seg3 fallback.

    Official seg3 is max(Lorentzian, Frechet). If the exact official FD
    implementation is unavailable, use Lorentzian as the local proxy.
    """
    pred = pred.astype(np.float64, copy=False)
    true = true.astype(np.float64, copy=False)

    def rel_mse(a, b):
        return float(np.sum((a - b) ** 2) / (np.sum(b ** 2) + 1e-12))

    rel1 = rel_mse(pred[:, 20:50], true[:, 20:50])
    rel2 = rel_mse(pred[:, 50:200], true[:, 50:200])
    rmse3 = float(np.sqrt(np.mean((pred[:, 200:400] - true[:, 200:400]) ** 2)))
    score1 = 100.0 * math.exp(-20.0 * rel1)
    score2 = 100.0 * math.exp(-10.0 * rel2)
    lorentzian3 = 100.0 / (1.0 + 10.0 * rmse3)
    return 0.25 * score1 + 0.25 * score2 + 0.50 * lorentzian3

with h5py.File("data/KS_val.hdf5", "r") as f:
    val_true_raw = f["tensor"][:].astype(np.float32)               # (100, 400, 256)
    val_x        = f["x-coordinate"][:].astype(np.float32)

def eval_val_candidate(label):
    model.eval()
    val_pred_norm = rollout_ks(
        model, (val_true_raw[:, :20, :] - u_mean) / u_std,
        val_x, total_steps=400, device=device, batch_size=50,
    )
    val_pred = val_pred_norm * u_std + u_mean
    score = official_like_segment_score(val_pred, val_true_raw)
    proxy = weighted_mse_proxy(val_pred, val_true_raw)
    print(f"{label} official_like_segment_score = {score:.6f}")
    print(f"{label} weighted_mse_proxy = {proxy:.6f}")
    return score

raw_state = deepcopy(model.state_dict())
raw_score = eval_val_candidate("raw")
model.load_state_dict(ema_state)
ema_score = eval_val_candidate("ema")
if raw_score >= ema_score:
    model.load_state_dict(raw_state)
    print("keeping raw weights for test rollout")
else:
    print("keeping EMA weights for test rollout")

# 6. Test rollout + persist inference_time. The rollout call is the ONLY
#    thing wrapped by time.perf_counter() — not data loading, not the file
#    write. Write inference_time to its own dedicated txt file (do not
#    embed it in task3_pred.hdf5 as an attribute).
with h5py.File("data/KS_test.hdf5", "r") as f:
    test_initial = f["tensor"][:].astype(np.float32)               # (100, 20, 256)
    test_x       = f["x-coordinate"][:].astype(np.float32)

_t0 = time.perf_counter()
test_pred_norm = rollout_ks(
    model, (test_initial - u_mean) / u_std, test_x,
    total_steps=400, device=device, batch_size=100,
)
inference_time = time.perf_counter() - _t0
print(f"INFERENCE_TIME={inference_time:.3f}")
with open("task3_inference_time.txt", "w") as f:
    f.write(f"{inference_time:.3f}\n")
test_pred = test_pred_norm * u_std + u_mean
test_pred[:, :20, :] = test_initial                                # mandatory copy

assert test_pred.shape == (100, 400, 256)
assert np.all(np.isfinite(test_pred))
assert np.max(np.abs(test_pred[:, :20, :] - test_initial)) < 1e-3
assert inference_time < 120, f"inference_time={inference_time:.1f}s exceeds 2-min cap"

with h5py.File("task3_pred.hdf5", "w") as f:
    f.create_dataset("tensor", data=test_pred.astype(np.float32))
```

### Hyperparameter notes

- **Architecture**: use `width=96, modes=32` as the stronger default
  baseline. KS dynamics contain higher-frequency structure from the
  `u_xxxx` term; going below `modes=16` visibly underfits the chaotic regime.
  If runtime is too high, fall back to `width=64, modes=24` before changing
  the training objective.
- **Epochs**: 40 with cosine annealing is the floor. KS chaos rewards
  longer training — 80–120 epochs gives noticeable gains, but watch val
  to detect overfitting.
- **Horizon curriculum**: pushing to `H=20` is required, and a short late
  phase at `H=40` is the minimal stronger baseline. Models that only ever see
  `H<=5` collapse on `seg3`.
- **Normalization**: KS amplitudes span roughly `[-7, +7]`. Subtract
  train-set mean / divide by std on inputs and targets; denormalize
  predictions before any metric.
- **Spectral and gradient losses**: keep MSE as the main loss, but add
  log-power spectral loss (`SPECTRAL_W≈0.20`) and a cheap spatial
  first-difference loss (`GRAD_W≈0.05`). These are the minimal additions that
  target the long-horizon distributional score without changing the model
  family.
- **EMA + best-by-validation**: maintain EMA weights during training, evaluate
  both raw and EMA weights on a full 400-step validation rollout, and keep the
  higher `official_like_segment_score`. Never let a later run overwrite a
  better validation-selected prediction.
- **Seed ensembling**: 3-model average over different seeds typically
  shaves a few % off val score at near-zero engineering cost. Do it
  after picking the single-model recipe.

## Inference Efficiency (2-minute hard cap)

Test set is `(100, 400, 256)` — 100 samples × 380 forward calls. Even
on CPU this should fit comfortably in 60–90 s with the model above; on
GPU it's under 10 s.

If you find yourself approaching the cap:
1. Run inference on **GPU** (`torch.device("cuda")`).
2. **Batch all 100 samples in one call** — `batch_size=100`.
3. `torch.compile(model)` after the last `eval()` toggle.
4. `model.half()` — risky for KS chaos; verify val score before using.

Always time the full 100-sample rollout end-to-end. The `assert
inference_time < 120` line in the recipe is there to fail loud if you
miss the cap.

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `KeyError: 't_coordinate'` | Copied Task-2's loader; Task-3 uses HYPHENS | Use `f["t-coordinate"]`, `f["x-coordinate"]` |
| `KeyError: 'lambda2'` on test | Tried to read `lambda2` from `KS_test.hdf5`; it isn't there | Test inference must be `λ₂`-free; predict it from the prefix if needed |
| Disqualification: "used pretrained weights" | Loaded `*.pt` from anywhere | Random-init only; train end-to-end on `KS_train.hdf5` |
| First 20 frames mismatch test input | Predicted from frame 0 instead of copying | `test_pred[:, :20, :] = test_initial` after rollout |
| `seg3` MSE explodes (predictions diverge past ~50 steps) | 1-step or 5-step rollout loss only | Curriculum to `H=20`, add spectral loss |
| Predictions blow up to ±100s of u-magnitude | No `clip_grad_norm_`, exploding autoregressive gradient | `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` |
| Inference exceeds 2 min on test | Sequential rollout per sample, no batching | Batch all 100 samples in one rollout call, use GPU |
| Train loss low but val score bad | Overfit to short-trajectory beginnings | Sample `t0w` uniformly in `[0, n_time-20-H)` |
| Score worse than expected | Forgot to denormalize predictions before metric | If you normalized, `pred = pred*std + mean` |
| `seg1`/`seg2` good, `seg3` bad | `seg3` rewards distribution matching, not pointwise | Add spectral loss; consider ensembling seeds |

## Output Files

Your script must create **two separate files in CWD**:

### 1. `task3_pred.hdf5` — prediction tensor only

```text
Required HDF5 dataset:
  tensor   float32-compatible   shape (100, 400, 256)
```

Hard requirements:

- `tensor[:, :20, :]` must copy `data/KS_test.hdf5["tensor"]` within
  tolerance `1e-3`.
- `tensor[:, 20:400, :]` must contain your 380 predicted future frames.
- All values must be finite.
- Test-set `lambda2` is not provided and must not be assumed.
- **Do not** attach extra metadata, attributes, or auxiliary datasets.
  This file holds the prediction tensor and nothing else.

### 2. `task3_inference_time.txt` — rollout wall-clock

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
checked against this value. Wrap **only** the `rollout_ks(...)` call
(not data loading or file writing) with `time.perf_counter()`:

```python
import time
t0 = time.perf_counter()
test_pred = rollout_ks(model, test_initial, test_x, total_steps=400,
                       device=device, batch_size=100)
inference_time = time.perf_counter() - t0
with open("task3_inference_time.txt", "w") as f:
    f.write(f"{inference_time:.3f}\n")
print(f"INFERENCE_TIME={inference_time:.3f}")
assert inference_time < 120, f"exceeds 2-min cap: {inference_time:.1f}s"
```

Do **not** create `task3_time.csv` yourself — the submission packager
assembles it post-run from `task3_inference_time.txt` plus the LLM
call-log timestamps.

## Local Validation Guidance

`KS_val.hdf5` is the only honest signal you have for both short-horizon
accuracy and long-horizon distributional fidelity. Strategy:

- **Always compute the official-like segmented score**, not just plain MSE.
  The canonical formulas are summarized in
  `tasks/AI4S_PDE_Tasks_Scoring_Rules.md`. Use weighted MSE only as a
  secondary debug proxy.
- **Stratify by `λ₂`**. Low-`λ₂` samples are more chaotic and harder.
  If `seg3` errors concentrate at small `λ₂`, push horizon longer or
  add `λ₂` conditioning.
- **Spectrum check**: compute `|rfft(pred)|²` vs `|rfft(true)|²` on
  `seg3` for a few val samples. A model that diverges with the wrong
  spectrum will lose `seg3` even if pointwise MSE looks fine.
- **Never use val tensors in any backward pass.** The `lambda2` array
  may be read for stratified metrics; the `tensor` array is read only
  for computing errors.

The official Task 3 metric scores only the 380 predicted frames:

| Segment | Prediction range (raw frame index) | Physical time `t` | Weight | Formula |
|---|---:|---|---:|---|
| 1 | 20–49 (30 steps) | `[10, 24.5]` | 25% | `100·exp(-20·Rel_MSE_1)` |
| 2 | 50–199 (150 steps) | `[25, 99.5]` | 25% | `100·exp(-10·Rel_MSE_2)` |
| 3 | 200–399 (200 steps) | `[100, 199.5]` | 50% | `max(Lorentzian, Fréchet)` |

`seg3` `Lorentzian = 100 / (1 + 10·RMSE)`, `Fréchet = 50·exp(-FD²)`.
The `max` means a good distributional match alone is enough; pointwise
accuracy past Lyapunov time is not required.

If the exact official Frechet-distance implementation is unavailable locally,
use the Lorentzian branch for a deterministic lower-bound proxy, but keep the
full `max(Lorentzian, Fréchet)` formula in prompts and logs.

## Constraints (summary)

- Train from scratch — no public or pre-existing pretrained weights.
- Do not use numerical PDE solvers in the prediction path.
- Do not use any extra data beyond the three files in `data/`.
- Do not produce a file with missing `tensor`, wrong shape, non-finite
  values, mismatched first 20 frames, or anything dependent on
  test-time `lambda2`.
- Stay within 12 h training wall-clock and 2 min inference.
