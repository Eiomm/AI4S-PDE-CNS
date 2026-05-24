from __future__ import annotations

import math
import py_compile
import re
from pathlib import Path

import h5py
import numpy as np

from .node_schema import Metrics


CHECKPOINT_PATTERNS = re.compile(
    r"torch\.load|load_state_dict|from_pretrained|checkpoint|\.pt|\.pth|\.ckpt|safetensors"
)

FORBIDDEN_PATTERNS = {
    "reads_test_nu": re.compile(r"task2_test\.h5[^\n]*(?:nu|\['nu'\]|\[\"nu\"\])|(?:\['nu'\]|\[\"nu\"\])[^\n]*task2_test\.h5"),
    "numerical_solver": re.compile(r"solve_ivp|odeint|burgers.*solver|finite.?difference", re.IGNORECASE),
    "writes_data_dir": re.compile(r"data/Task2[^\n]*(?:write|open\([^\n]*['\"]w|h5py\.File\([^\n]*['\"]w)", re.IGNORECASE),
    "dangerous_shell": re.compile(r"rm\s+-rf|mkfs|shutdown|reboot|curl\s+[^|]*\|\s*sh|wget\s+[^|]*\|\s*sh"),
}


class Evaluator:
    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)

    def checkpoint_scan(self, code_dir: Path | str) -> list[str]:
        code_dir = Path(code_dir)
        hits: list[str] = []
        for path in code_dir.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for m in CHECKPOINT_PATTERNS.finditer(text):
                hits.append(f"{path.relative_to(code_dir)}:{m.start()}: {m.group()}")
        return hits

    def static_check(self, code_dir: Path | str) -> tuple[bool, list[str]]:
        code_dir = Path(code_dir)
        train_py = code_dir / "train.py"
        reasons: list[str] = []
        if not train_py.exists():
            return False, ["missing train.py"]
        text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in code_dir.rglob("*.py"))
        for path in code_dir.rglob("*.py"):
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                reasons.append(f"syntax_error:{path.relative_to(code_dir)}:{exc.msg}")
        for name, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                reasons.append(name)
        return not reasons, reasons

    def shape_check(self, pred_path: Path | str) -> Metrics:
        pred_path = Path(pred_path)
        metrics = Metrics()
        if not pred_path.exists():
            return metrics
        with h5py.File(self.data_dir / "task2_test.h5", "r") as test, h5py.File(pred_path, "r") as pred:
            if "tensor" not in pred:
                return metrics
            tensor = pred["tensor"][:]
            metrics.shape_pass = bool(tensor.shape == (1000, 200, 256) and tensor.dtype == np.float32 and np.isfinite(tensor).all())
            if metrics.shape_pass:
                max_abs = float(np.max(np.abs(tensor[:, :10, :] - test["tensor"][:])))
                metrics.first_10_pass = bool(max_abs <= 1e-3)
        metrics.compliance_pass = metrics.shape_pass and metrics.first_10_pass and not metrics.uses_true_nu_at_test
        metrics.reward = self.compute_reward(metrics)
        return metrics

    def validation_metrics(self, val_pred_path: Path | str, runtime_sec: float | None = None) -> Metrics:
        val_pred_path = Path(val_pred_path)
        metrics = Metrics(runtime_sec=runtime_sec)
        if not val_pred_path.exists():
            metrics.reward = self.compute_reward(metrics)
            return metrics
        with h5py.File(self.data_dir / "task2_val.h5", "r") as val, h5py.File(val_pred_path, "r") as pred:
            pred_tensor = pred["tensor"][:].astype(np.float32)
            target = val["tensor"][:, :200, :].astype(np.float32)
            nu = val["nu"][:]
        if pred_tensor.shape != target.shape or not np.isfinite(pred_tensor).all():
            metrics.reward = self.compute_reward(metrics)
            return metrics
        future_pred = pred_tensor[:, 10:200, :]
        future_target = target[:, 10:200, :]
        diff = future_pred - future_target
        metrics.overall_mse = float(np.mean(diff * diff))
        metrics.short_mse = float(np.mean((pred_tensor[:, 10:60, :] - target[:, 10:60, :]) ** 2))
        stats_pred = np.stack([future_pred.mean(axis=(1, 2)), future_pred.std(axis=(1, 2))], axis=1)
        stats_target = np.stack([future_target.mean(axis=(1, 2)), future_target.std(axis=(1, 2))], axis=1)
        metrics.long_stat_error = float(np.mean((stats_pred - stats_target) ** 2))
        segments = [(0, 47), (47, 95), (95, 190)]
        for idx, (start, end) in enumerate(segments, 1):
            seg_pred = future_pred[:, start:end, :]
            seg_target = future_target[:, start:end, :]
            denom = float(np.mean(seg_target * seg_target) + 1e-12)
            setattr(metrics, f"rel_mse_seg{idx}", float(np.mean((seg_pred - seg_target) ** 2) / denom))
        per_nu = {}
        for i, value in enumerate(nu):
            per_nu[f"{float(value):.8g}"] = float(np.mean(diff[i] * diff[i]))
        metrics.per_nu_mse = per_nu
        metrics.worst_nu_mse = max(per_nu.values()) if per_nu else None
        metrics.shape_pass = True
        metrics.first_10_pass = bool(np.max(np.abs(pred_tensor[:, :10, :] - target[:, :10, :])) <= 1e-3)
        metrics.compliance_pass = metrics.shape_pass and metrics.first_10_pass and not metrics.uses_true_nu_at_test
        metrics.official_score_estimate = self._estimate_official_score(future_pred, future_target, metrics)
        metrics.reward = self.compute_reward(metrics)
        return metrics

    def _estimate_official_score(self, future_pred: np.ndarray, future_target: np.ndarray, metrics: Metrics) -> float | None:
        """Official Task2 score (out of 100).

        Segments: 0-47, 47-95, 95-190 (relative to future_190 = steps 10:200)
        - Seg1: weight 25%, raw = 100 * exp(-20 * Rel-MSE)
        - Seg2: weight 25%, raw = 100 * exp(-10 * Rel-MSE)
        - Seg3: weight 50%, raw = max(Lorentzian, Fréchet)
        Total: 100 points
        """
        if metrics.rel_mse_seg1 is None or metrics.rel_mse_seg2 is None or metrics.rel_mse_seg3 is None:
            return None
        seg1_raw = 100.0 * np.exp(-20.0 * metrics.rel_mse_seg1)
        seg2_raw = 100.0 * np.exp(-10.0 * metrics.rel_mse_seg2)
        # Seg3
        seg3_pred = future_pred[:, 95:190, :]
        seg3_target = future_target[:, 95:190, :]
        seg3_mse = float(np.mean((seg3_pred - seg3_target) ** 2))
        seg3_lorentzian = 100.0 / (1.0 + seg3_mse)
        pred_mean = seg3_pred.mean()
        target_mean = seg3_target.mean()
        pred_std = seg3_pred.std()
        target_std = seg3_target.std()
        stat_dist = (pred_mean - target_mean) ** 2 + (pred_std - target_std) ** 2
        seg3_frechet = 100.0 * np.exp(-stat_dist)
        seg3_raw = max(seg3_lorentzian, seg3_frechet)
        return float(seg1_raw * 0.25 + seg2_raw * 0.25 + seg3_raw * 0.50)

    @staticmethod
    def compute_reward(metrics: Metrics) -> float:
        if not metrics.compliance_pass:
            return -1.0
        if metrics.official_score_estimate is not None:
            # Official full score is 100 points; normalize to [-1, 1] range roughly
            return (metrics.official_score_estimate - 50.0) / 50.0
        if metrics.overall_mse is None:
            return -0.5
        reward = -math.log10(metrics.overall_mse + 1e-12)
        if metrics.runtime_sec is not None:
            reward -= 0.1 * math.log1p(metrics.runtime_sec / 600)
        if metrics.worst_nu_mse is not None:
            reward -= 0.2 * math.log1p(metrics.worst_nu_mse / max(metrics.overall_mse, 1e-12))
        return float(reward)
