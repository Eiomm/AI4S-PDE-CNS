import numpy as np

from agent.pde_physics import (
    burgers_residual_mse,
    estimate_viscosity_from_initial,
    fc_lite_spatial_derivatives,
    select_physics_rerank_candidate,
)


def _small_amplitude_heat_solution(*, nu=0.1, samples=1, times=32, points=128):
    x = np.linspace(0.0, 1.0, points, endpoint=False, dtype=np.float64)
    t = np.linspace(0.0, 0.12, times, dtype=np.float64)
    amplitude = 1.0e-4
    wave_number = 2.0 * np.pi
    values = amplitude * np.exp(-nu * wave_number**2 * t[:, None]) * np.sin(wave_number * x[None, :])
    trajectory = np.repeat(values[None, :, :], samples, axis=0).astype(np.float64)
    return trajectory, x, t


def test_fc_lite_spatial_derivatives_match_periodic_sine():
    x = np.linspace(0.0, 1.0, 128, endpoint=False, dtype=np.float64)
    u = np.sin(2.0 * np.pi * x)[None, :]

    ux, uxx = fc_lite_spatial_derivatives(u, x, continuation=16)

    assert np.max(np.abs(ux[0] - 2.0 * np.pi * np.cos(2.0 * np.pi * x))) < 2.0e-1
    assert np.max(np.abs(uxx[0] + (2.0 * np.pi) ** 2 * np.sin(2.0 * np.pi * x))) < 3.0


def test_burgers_residual_is_lower_for_matching_viscosity():
    trajectory, x, t = _small_amplitude_heat_solution(nu=0.1)

    matching = burgers_residual_mse(trajectory, x, t, nu=0.1, continuation=16, spatial_margin=4)
    wrong = burgers_residual_mse(trajectory, x, t, nu=0.4, continuation=16, spatial_margin=4)

    assert matching < wrong * 0.2


def test_estimate_viscosity_from_initial_recovers_synthetic_viscosity():
    trajectory, x, t = _small_amplitude_heat_solution(nu=0.08, samples=2)

    estimates = estimate_viscosity_from_initial(trajectory, x, t, frames=12, continuation=16, spatial_margin=4)

    assert estimates.shape == (2,)
    assert np.all(np.abs(estimates - 0.08) < 0.015)


def test_select_physics_rerank_candidate_uses_physics_inside_mse_window():
    candidates = [
        {"name": "mse-best", "mse": 1.0, "physics_mse": 10.0},
        {"name": "physics-best", "mse": 1.05, "physics_mse": 1.0},
        {"name": "too-far", "mse": 1.5, "physics_mse": 0.1},
    ]

    selected = select_physics_rerank_candidate(candidates, mse_tolerance=0.1)

    assert selected["name"] == "physics-best"
