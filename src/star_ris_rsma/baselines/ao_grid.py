from __future__ import annotations

import numpy as np

from ..action import encode_action
from ..env import StarRisRsmaEnv
from .analytical_ris import analytical_action
from .common import physical_slices, state_from_action, state_from_vector


def _redistribute_simplex(values: np.ndarray, index: int, selected: float, total: float) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    selected = float(np.clip(selected, 0.0, total))
    others = np.arange(result.size) != index
    remainder = total - selected
    previous = float(result[others].sum())
    result[index] = selected
    if not np.any(others):
        return result
    if previous <= 1e-12:
        result[others] = remainder / int(np.sum(others))
    else:
        result[others] *= remainder / previous
    return result


def solve(
    env: StarRisRsmaEnv,
    rounds: int = 2,
    allocation_levels: int = 7,
    beta_levels: int = 5,
    phase_levels: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Deterministic alternating coordinate grid search.

    Every selected coordinate is evaluated on a declared finite codebook. Power
    and common allocations stay on their simplices by redistributing the
    residual mass. STAR-RIS beta and phase variables use fixed discrete grids.
    """
    del seed  # deterministic codebooks and coordinate order
    current = state_from_action(env, analytical_action(env))
    c = env.config
    s = physical_slices(c.n_users, c.n_ris)
    power_grid = np.linspace(0.05, 0.80, allocation_levels) * c.p_max
    common_grid = np.linspace(0.05, 0.80, allocation_levels)
    beta_grid = np.linspace(0.05, 0.95, beta_levels)
    phase_grid = np.linspace(-np.pi, np.pi, phase_levels, endpoint=False)
    history = [current.score]
    evaluations = 1

    for _ in range(rounds):
        previous = current.score

        for local_index, absolute_index in enumerate(range(s["powers"].start, s["powers"].stop)):
            for value in power_grid:
                trial = current.vector.copy()
                trial[s["powers"]] = _redistribute_simplex(
                    current.vector[s["powers"]], local_index, value, c.p_max
                )
                candidate = state_from_vector(env, trial); evaluations += 1
                if candidate.score > current.score + 1e-12:
                    current = candidate

        for local_index, absolute_index in enumerate(range(s["common"].start, s["common"].stop)):
            del absolute_index
            for value in common_grid:
                trial = current.vector.copy()
                trial[s["common"]] = _redistribute_simplex(
                    current.vector[s["common"]], local_index, value, 1.0
                )
                candidate = state_from_vector(env, trial); evaluations += 1
                if candidate.score > current.score + 1e-12:
                    current = candidate

        for block, grid in [("beta", beta_grid), ("theta_t", phase_grid), ("theta_r", phase_grid)]:
            sl = s[block]
            for absolute_index in range(sl.start, sl.stop):
                for value in grid:
                    trial = current.vector.copy()
                    trial[absolute_index] = value
                    candidate = state_from_vector(env, trial); evaluations += 1
                    if candidate.score > current.score + 1e-12:
                        current = candidate

        history.append(current.score)
        if abs(current.score - previous) / max(1.0, abs(previous)) < 1e-4:
            break

    metrics = dict(current.metrics)
    metrics.update({
        "solver": "ao_grid_coordinate_codebook",
        "objective_history": history,
        "iterations": len(history) - 1,
        "evaluations": evaluations,
        "grid": {
            "allocation_levels": allocation_levels,
            "beta_levels": beta_levels,
            "phase_levels": phase_levels,
        },
        "initialization": "analytical_ris_equal_allocation",
    })
    return encode_action(current.action, c.p_max), metrics
