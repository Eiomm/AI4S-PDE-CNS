from __future__ import annotations

from typing import Any

import numpy as np


def _uniform_spacing(coords: np.ndarray) -> float:
    values = np.asarray(coords, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("coords must be a one-dimensional array with at least two points")
    diffs = np.diff(values)
    return float(np.median(diffs))


def _hermite_bridge(values: np.ndarray, dx: float, continuation: int) -> np.ndarray:
    if continuation <= 0:
        return values
    left_slope = (values[..., 1] - values[..., 0]) / dx
    right_slope = (values[..., -1] - values[..., -2]) / dx
    bridge_length = continuation * dx
    pieces = []
    y0 = values[..., -1]
    y1 = values[..., 0]
    for index in range(1, continuation + 1):
        s = index / continuation
        h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
        h10 = s**3 - 2.0 * s**2 + s
        h01 = -2.0 * s**3 + 3.0 * s**2
        h11 = s**3 - s**2
        bridge = h00 * y0 + h10 * bridge_length * right_slope + h01 * y1 + h11 * bridge_length * left_slope
        pieces.append(bridge[..., None])
    return np.concatenate([values, *pieces], axis=-1)


def _looks_periodic_endpoint_excluded(values: np.ndarray) -> bool:
    if values.shape[-1] < 3:
        return False
    wrapped_next = values[..., -1] + (values[..., -1] - values[..., -2])
    jump = np.mean(np.abs(wrapped_next - values[..., 0]))
    scale = max(float(np.mean(np.ptp(values, axis=-1))), 1.0e-12)
    return jump <= 1.0e-2 * scale


def fc_lite_spatial_derivatives(
    field: np.ndarray,
    x_coords: np.ndarray,
    *,
    continuation: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate spatial derivatives with a lightweight Fourier-continuation bridge."""

    values = np.asarray(field, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("field must have a spatial dimension")
    x = np.asarray(x_coords, dtype=np.float64)
    if values.shape[-1] != len(x):
        raise ValueError(f"field spatial size {values.shape[-1]} does not match x_coords {len(x)}")
    dx = _uniform_spacing(x)
    if continuation > 0 and not _looks_periodic_endpoint_excluded(values):
        extended = _hermite_bridge(values, dx, continuation)
    else:
        extended = values
    length = extended.shape[-1]
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(length, d=dx)
    spectrum = np.fft.fft(extended, axis=-1)
    first = np.fft.ifft(1j * wave_numbers * spectrum, axis=-1).real
    second = np.fft.ifft(-(wave_numbers**2) * spectrum, axis=-1).real
    return first[..., : values.shape[-1]], second[..., : values.shape[-1]]


def _time_derivative(trajectory: np.ndarray, t_coords: np.ndarray) -> np.ndarray:
    t = np.asarray(t_coords, dtype=np.float64)
    values = np.asarray(trajectory, dtype=np.float64)
    if values.shape[1] != len(t):
        raise ValueError(f"trajectory time size {values.shape[1]} does not match t_coords {len(t)}")
    edge_order = 2 if values.shape[1] >= 3 else 1
    return np.gradient(values, t, axis=1, edge_order=edge_order)


def burgers_residual(
    trajectory: np.ndarray,
    x_coords: np.ndarray,
    t_coords: np.ndarray,
    *,
    nu: float | np.ndarray,
    continuation: int = 32,
) -> np.ndarray:
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"trajectory must have shape (N, T, X), got {values.shape}")
    ut = _time_derivative(values, np.asarray(t_coords, dtype=np.float64))
    ux, uxx = fc_lite_spatial_derivatives(values, np.asarray(x_coords, dtype=np.float64), continuation=continuation)
    nu_values = np.asarray(nu, dtype=np.float64)
    if nu_values.ndim == 0:
        nu_values = np.full((values.shape[0],), float(nu_values), dtype=np.float64)
    if nu_values.shape != (values.shape[0],):
        raise ValueError(f"nu must be scalar or shape ({values.shape[0]},), got {nu_values.shape}")
    return ut + values * ux - nu_values[:, None, None] * uxx


def burgers_residual_mse(
    trajectory: np.ndarray,
    x_coords: np.ndarray,
    t_coords: np.ndarray,
    *,
    nu: float | np.ndarray,
    continuation: int = 32,
    time_margin: int = 1,
    spatial_margin: int = 2,
    time_stride: int = 1,
) -> float:
    residual = burgers_residual(
        trajectory,
        x_coords,
        t_coords,
        nu=nu,
        continuation=continuation,
    )
    time_slice = slice(time_margin, residual.shape[1] - time_margin, time_stride)
    if spatial_margin > 0:
        spatial_slice = slice(spatial_margin, residual.shape[2] - spatial_margin)
    else:
        spatial_slice = slice(None)
    trimmed = residual[:, time_slice, spatial_slice]
    return float(np.mean(trimmed**2))


def estimate_viscosity_from_initial(
    trajectory: np.ndarray,
    x_coords: np.ndarray,
    t_coords: np.ndarray,
    *,
    frames: int = 10,
    continuation: int = 32,
    spatial_margin: int = 2,
    min_nu: float = 1.0e-4,
    max_nu: float = 2.0,
) -> np.ndarray:
    values = np.asarray(trajectory, dtype=np.float64)[:, :frames, :]
    times = np.asarray(t_coords, dtype=np.float64)[:frames]
    ut = _time_derivative(values, times)
    ux, uxx = fc_lite_spatial_derivatives(values, np.asarray(x_coords, dtype=np.float64), continuation=continuation)
    lhs = ut + values * ux
    if spatial_margin > 0:
        lhs = lhs[:, :, spatial_margin:-spatial_margin]
        uxx = uxx[:, :, spatial_margin:-spatial_margin]
    numerator = np.mean(lhs * uxx, axis=(1, 2))
    denominator = np.mean(uxx**2, axis=(1, 2))
    estimates = np.divide(numerator, denominator, out=np.full_like(numerator, min_nu), where=denominator > 1.0e-18)
    return np.clip(estimates, min_nu, max_nu)


def select_physics_rerank_candidate(
    candidates: list[dict[str, Any]],
    *,
    mse_tolerance: float,
    mse_key: str = "mse",
    physics_key: str = "physics_mse",
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("candidates must not be empty")
    best_mse = min(float(candidate[mse_key]) for candidate in candidates)
    allowed = [
        candidate
        for candidate in candidates
        if float(candidate[mse_key]) <= best_mse * (1.0 + mse_tolerance)
    ]
    return min(allowed, key=lambda candidate: (float(candidate[physics_key]), float(candidate[mse_key])))
