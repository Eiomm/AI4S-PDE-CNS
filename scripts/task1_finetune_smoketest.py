"""Empirical fine-tune smoke test for Task-1.

Goal: verify that loading the pretrained checkpoint and continuing training on
the official PDEBench 1D_Burgers_Sols_Nu0.001.hdf5 actually decreases the
weighted forecast score on task1_val.hdf5 (read-only).

Constraints: NEVER train on task1_val.hdf5 or task1_test.hdf5.
"""

import argparse
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
from ai4sv2_task1.models.fno import FNO1d, load_fno_checkpoint, rollout_fno  # noqa: E402


def weighted_val_score(pred: np.ndarray, true: np.ndarray) -> dict:
    seg1 = float(np.mean((pred[:, 10:57]   - true[:, 10:57])   ** 2))
    seg2 = float(np.mean((pred[:, 57:105]  - true[:, 57:105])  ** 2))
    seg3 = float(np.mean((pred[:, 105:200] - true[:, 105:200]) ** 2))
    return {
        "weighted": 0.25 * seg1 + 0.25 * seg2 + 0.50 * seg3,
        "mse": float(np.mean((pred[:, 10:200] - true[:, 10:200]) ** 2)),
        "seg1": seg1, "seg2": seg2, "seg3": seg3,
    }


def evaluate_on_val(model: FNO1d, device: torch.device) -> dict:
    with h5py.File(TASK_DIR / "data" / "task1_val.hdf5", "r") as f:
        val_true = f["tensor"][:].astype(np.float32)
        val_x    = f["x-coordinate"][:].astype(np.float32)
        val_t    = f["t-coordinate"][:].astype(np.float32)
    pred = rollout_fno(model, val_true[:, :10, :], val_x, val_t, device, batch_size=50)
    return weighted_val_score(pred, val_true)


def load_training(reduced_t: int, reduced_x: int) -> tuple[torch.Tensor, np.ndarray]:
    path = TASK_DIR / "data" / "1D_Burgers_Sols_Nu0.001.hdf5"
    with h5py.File(path, "r") as f:
        u = f["tensor"][:]              # (10000, 201, 1024)
        x = f["x-coordinate"][:]        # (1024,)
    u = u[:, :200:reduced_t, ::reduced_x]  # → (10000, 40, 256)
    x = x[::reduced_x]                     # → (256,)
    return torch.from_numpy(u.astype(np.float32)), x.astype(np.float32)


def fine_tune(
    model: FNO1d,
    train_u: torch.Tensor,
    train_x: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    horizon: int,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    grid = torch.tensor(
        (train_x - train_x.min()) / (train_x.max() - train_x.min()),
        dtype=torch.float32, device=device,
    ).view(1, len(train_x), 1)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n_samples = train_u.shape[0]
    n_time    = train_u.shape[1]

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        running, batches = 0.0, 0
        t0 = time.perf_counter()
        for s in range(0, n_samples, batch_size):
            idx = perm[s:s + batch_size]
            batch = train_u[idx].to(device)                           # (B, n_time, 256)
            # Random window start so we cover all temporal regimes
            t_start = int(torch.randint(0, n_time - 10 - horizon, (1,)).item())
            window  = batch[:, t_start:t_start + 10, :]               # (B, 10, 256)
            target  = batch[:, t_start + 10:t_start + 10 + horizon]   # (B, h, 256)

            xx = window.permute(0, 2, 1).contiguous()                 # (B, 256, 10)
            batch_grid = grid.expand(xx.shape[0], -1, -1)
            preds = []
            for _ in range(horizon):
                p = model(
                    xx.reshape(xx.shape[0], xx.shape[1], -1),
                    batch_grid,
                ).squeeze(-1).squeeze(-1)                             # (B, 256)
                preds.append(p)
                xx = torch.cat((xx[:, :, 1:], p.unsqueeze(-1)), dim=2)

            pred_stack = torch.stack(preds, dim=1)                    # (B, h, 256)
            loss = F.mse_loss(pred_stack, target)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += loss.item()
            batches += 1
        dt = time.perf_counter() - t0
        print(f"  epoch {epoch}: train_mse={running/batches:.6f}  ({dt:.1f}s)")
    model.eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--reduced-t", type=int, default=5)
    parser.add_argument("--reduced-x", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=10000,
                        help="cap training samples for fast smoke test")
    parser.add_argument("--save-best", type=Path, default=None,
                        help="save the best weights to this file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    ckpt = TASK_DIR / "burgers_FNO" / "1D_Burgers_Sols_Nu0.001_FNO.pt"
    model = load_fno_checkpoint(str(ckpt), device)
    print(f"loaded checkpoint width={model.fc0.out_features} modes={model.conv0.modes1}")

    # 1) baseline
    baseline = evaluate_on_val(model, device)
    print(f"baseline val: weighted={baseline['weighted']:.6f}  mse={baseline['mse']:.6f} "
          f"seg1={baseline['seg1']:.4f} seg2={baseline['seg2']:.4f} seg3={baseline['seg3']:.4f}")
    best_state = deepcopy(model.state_dict())
    best_score = baseline["weighted"]

    # 2) load training
    print("loading PDEBench training file (~8 GB)…")
    t0 = time.perf_counter()
    train_u, train_x = load_training(args.reduced_t, args.reduced_x)
    if args.n_train < train_u.shape[0]:
        train_u = train_u[:args.n_train]
    print(f"  train tensor shape={tuple(train_u.shape)} ({time.perf_counter()-t0:.1f}s)")

    # 3) fine-tune
    print(f"fine-tuning {args.epochs} epoch(s) lr={args.lr} horizon={args.horizon}")
    fine_tune(model, train_u, train_x, device,
              epochs=args.epochs, batch_size=args.batch_size,
              lr=args.lr, horizon=args.horizon)

    # 4) eval
    after = evaluate_on_val(model, device)
    print(f"after    val: weighted={after['weighted']:.6f}  mse={after['mse']:.6f} "
          f"seg1={after['seg1']:.4f} seg2={after['seg2']:.4f} seg3={after['seg3']:.4f}")
    delta = after["weighted"] - baseline["weighted"]
    print(f"Δweighted = {delta:+.6f}  ({'better' if delta < 0 else 'worse'})")

    if after["weighted"] < best_score:
        best_score = after["weighted"]
        best_state = deepcopy(model.state_dict())
        if args.save_best:
            args.save_best.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": best_state}, args.save_best)
            print(f"saved best weights → {args.save_best}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
