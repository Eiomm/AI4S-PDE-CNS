"""Sanity-check the fine-tuned predictions to make sure the loss drop is real
and not the model collapsing to a trivial solution.
"""

import sys
import time
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
TASK_DIR = REPO / "tasks" / "ai4s-pde-task1-burgers-fixed"
sys.path.insert(0, str(REPO / "src"))
from ai4sv2_task1.models.fno import load_fno_checkpoint, rollout_fno  # noqa: E402


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Baseline
m = load_fno_checkpoint(str(TASK_DIR / "burgers_FNO" / "1D_Burgers_Sols_Nu0.001_FNO.pt"), device)

with h5py.File(TASK_DIR / "data" / "task1_val.hdf5", "r") as f:
    val_true = f["tensor"][:].astype(np.float32)
    val_x    = f["x-coordinate"][:].astype(np.float32)
    val_t    = f["t-coordinate"][:].astype(np.float32)

pred_base = rollout_fno(m, val_true[:, :10, :], val_x, val_t, device, batch_size=50)
print(f"baseline:  pred range [{pred_base.min():.4f}, {pred_base.max():.4f}]  true range [{val_true.min():.4f}, {val_true.max():.4f}]")
print(f"baseline:  rollout L2 norm at frame  10 / 100 / 190: "
      f"{np.linalg.norm(pred_base[:, 10]):.2f} / {np.linalg.norm(pred_base[:, 100]):.2f} / {np.linalg.norm(pred_base[:, 199]):.2f}")
print(f"true:     L2 norm at frame  10 / 100 / 190: "
      f"{np.linalg.norm(val_true[:, 10]):.2f} / {np.linalg.norm(val_true[:, 100]):.2f} / {np.linalg.norm(val_true[:, 199]):.2f}")
print()

# Fine-tune (replicate smoketest setup)
with h5py.File(TASK_DIR / "data" / "1D_Burgers_Sols_Nu0.001.hdf5", "r") as f:
    train = f["tensor"][:].astype(np.float32)[:, :200:5, ::4]
    train_x = f["x-coordinate"][:].astype(np.float32)[::4]
train_u = torch.from_numpy(train[:2000])
print(f"train  shape={tuple(train_u.shape)}  range [{train_u.min():.4f}, {train_u.max():.4f}]")
print(f"train range matches val range?  val [{val_true.min():.4f}, {val_true.max():.4f}]")

torch.manual_seed(0)
grid = torch.tensor(
    (train_x - train_x.min()) / (train_x.max() - train_x.min()),
    dtype=torch.float32, device=device,
).view(1, len(train_x), 1)
optim = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-5)
horizon = 5
n_time = train_u.shape[1]
m.train()
t0 = time.perf_counter()
for s in range(0, 2000, 32):
    batch = train_u[s:s+32].to(device)
    t_start = int(torch.randint(0, n_time - 10 - horizon, (1,)).item())
    window = batch[:, t_start:t_start + 10, :]
    target = batch[:, t_start + 10:t_start + 10 + horizon, :]
    xx = window.permute(0, 2, 1).contiguous()
    batch_grid = grid.expand(xx.shape[0], -1, -1)
    preds = []
    for _ in range(horizon):
        p = m(xx.reshape(xx.shape[0], xx.shape[1], -1), batch_grid).squeeze(-1).squeeze(-1)
        preds.append(p)
        xx = torch.cat((xx[:, :, 1:], p.unsqueeze(-1)), dim=2)
    loss = F.mse_loss(torch.stack(preds, dim=1), target)
    optim.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    optim.step()
print(f"fine-tuned 1 epoch in {time.perf_counter()-t0:.1f}s")
m.eval()

pred_ft = rollout_fno(m, val_true[:, :10, :], val_x, val_t, device, batch_size=50)
print(f"\nafter ft:  pred range [{pred_ft.min():.4f}, {pred_ft.max():.4f}]")
print(f"after ft:  rollout L2 norm at frame  10 / 100 / 190: "
      f"{np.linalg.norm(pred_ft[:, 10]):.2f} / {np.linalg.norm(pred_ft[:, 100]):.2f} / {np.linalg.norm(pred_ft[:, 199]):.2f}")

print()
print(f"per-sample frame-wise MSE on val[0] :")
for t in [10, 50, 100, 150, 199]:
    base_err = float(np.mean((pred_base[0, t] - val_true[0, t])**2))
    ft_err   = float(np.mean((pred_ft[0, t]   - val_true[0, t])**2))
    print(f"  frame {t:3d}: baseline mse={base_err:.6f}  ft mse={ft_err:.6f}  truth |u|={float(np.linalg.norm(val_true[0, t])):.3f}")
