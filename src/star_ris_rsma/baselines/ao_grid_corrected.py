from __future__ import annotations

import numpy as np

from star_ris_rsma.action import encode_action

from .analytical_ris import analytical_action
from .common import physical_slices, state_from_action, state_from_vector

ALGORITHM_VERSION = "corrected_ao_grid_v1"
FROZEN_ROUNDS = 2


def _redistribute(values: np.ndarray, index: int, selected: float, total: float) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    selected = float(np.clip(selected, 0.0, total))
    others = np.arange(result.size) != index
    remainder = total - selected
    previous = float(result[others].sum())
    result[index] = selected
    if np.any(others):
        if previous <= 1e-12:
            result[others] = remainder / int(others.sum())
        else:
            result[others] = result[others] * remainder / previous
    return result


def _run_sweep(env, *, reverse_ris: bool, rounds: int):
    current = state_from_action(env, analytical_action(env))
    cfg = env.config
    slices = physical_slices(cfg.n_users, cfg.n_ris)

    simplex_grid = np.r_[0.0, np.linspace(0.05, 0.80, 7)]
    power_grid = simplex_grid * cfg.p_max
    beta_grid = np.linspace(0.05, 0.95, 5)
    phase_grid = np.linspace(-np.pi, np.pi, 8, endpoint=False)

    history = [float(current.score)]
    evaluations = 1
    accepted_steps = 0

    for _ in range(rounds):
        previous = float(current.score)

        # Best-improvement search inside each simplex block. Including level zero
        # allows a common/private component to be switched off exactly.
        for sl, grid, total, max_steps in (
            (slices["powers"], power_grid, cfg.p_max, cfg.n_users + 1),
            (slices["common"], simplex_grid, 1.0, cfg.n_users),
        ):
            for _ in range(max_steps):
                best = current
                base = current.vector[sl].copy()
                for i in range(sl.stop - sl.start):
                    for value in grid:
                        vector = current.vector.copy()
                        vector[sl] = _redistribute(base, i, float(value), total)
                        candidate = state_from_vector(env, vector)
                        evaluations += 1
                        if candidate.score > best.score + 1e-12:
                            best = candidate
                if best.score <= current.score + 1e-12:
                    break
                current = best
                accepted_steps += 1

        # The legacy coordinate order was biased. Run both RIS orders and let the
        # outer solve choose the better final feasible point.
        for block, grid in (
            ("beta", beta_grid),
            ("theta_t", phase_grid),
            ("theta_r", phase_grid),
        ):
            indices = list(range(slices[block].start, slices[block].stop))
            if reverse_ris:
                indices.reverse()
            for j in indices:
                best = current
                for value in grid:
                    vector = current.vector.copy()
                    vector[j] = float(value)
                    candidate = state_from_vector(env, vector)
                    evaluations += 1
                    if candidate.score > best.score + 1e-12:
                        best = candidate
                if best.score > current.score + 1e-12:
                    current = best
                    accepted_steps += 1

        history.append(float(current.score))
        if abs(current.score - previous) / max(1.0, abs(previous)) < 1e-4:
            break

    return current, history, evaluations, accepted_steps


def solve(env, *, rounds: int = FROZEN_ROUNDS, seed: int = 0):
    """Corrected discrete AO-Grid heuristic.

    Two RIS coordinate orders are evaluated and the feasible solution with the
    higher sum-rate is retained. The simplex codebook explicitly includes zero.
    This remains a restricted discrete heuristic, not a global optimizer.
    """

    del seed  # deterministic by design
    candidates = []
    total_evaluations = 0
    for reverse in (False, True):
        current, history, evaluations, accepted = _run_sweep(
            env, reverse_ris=reverse, rounds=rounds
        )
        total_evaluations += evaluations
        candidates.append((current, history, accepted, reverse))

    current, history, accepted_steps, reverse = max(
        candidates,
        key=lambda item: (bool(item[0].metrics["all_qos"]), float(item[0].metrics["sum_rate"])),
    )

    metrics = dict(current.metrics)
    metrics.update(
        solver=ALGORITHM_VERSION,
        algorithm_version=ALGORITHM_VERSION,
        rounds=int(rounds),
        iterations=len(history) - 1,
        evaluations=int(total_evaluations),
        accepted_steps=int(accepted_steps),
        initialization="analytical_ris",
        selected_ris_sweep="reverse" if reverse else "forward",
        objective_history=history,
        grid={
            "simplex": [0.0, *np.linspace(0.05, 0.80, 7).tolist()],
            "beta": np.linspace(0.05, 0.95, 5).tolist(),
            "phase": np.linspace(-np.pi, np.pi, 8, endpoint=False).tolist(),
        },
    )
    raw = encode_action(
        current.action,
        env.config.p_max,
        env.config.action_parameterization,
        channel=env.channel,
    )
    return raw, metrics
