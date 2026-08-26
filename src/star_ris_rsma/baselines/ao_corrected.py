from __future__ import annotations

import numpy as np

from star_ris_rsma.action import encode_action

from .analytical_ris import analytical_action
from .ao_sca import _block_gradient, _proximal_surrogate_maximizer
from .common import physical_slices, state_from_action, state_from_vector

# Candidate v2 freeze; must pass the locked N16/N32 stationarity pilot.
ALGORITHM_VERSION = "corrected_pairwise_ao_v2"
FROZEN_MAX_ITER = 80
FROZEN_TOL = 1e-4
FROZEN_STATIONARITY_TOL = 1e-6
FROZEN_GRADIENT_EPS = 1e-3
FROZEN_PAIRWISE_PROBE = 1e-3
FROZEN_PAIRWISE_MAX_STEPS = 12
FROZEN_LINE_POINTS = 12


def _pairwise_simplex_ascent(
    env,
    current,
    sl: slice,
    total: float,
    *,
    probe: float = FROZEN_PAIRWISE_PROBE,
    max_steps: int = FROZEN_PAIRWISE_MAX_STEPS,
    line_points: int = FROZEN_LINE_POINTS,
    tol: float = 1e-8,
):
    """Optimize one simplex block using feasible pairwise mass transfers.

    Every candidate direction is e_j - e_i, so feasibility of the simplex is
    preserved without taking a projected finite difference at the boundary.
    """

    evaluations = 0
    accepted = 0
    start, stop = sl.start, sl.stop
    dimension = stop - start

    for _ in range(max_steps):
        values = current.vector[sl].copy()
        best = None
        for i in range(dimension):
            available = float(values[i])
            delta = min(probe * total, available)
            if delta <= 1e-14:
                continue
            for j in range(dimension):
                if i == j:
                    continue
                vector = current.vector.copy()
                vector[start + i] -= delta
                vector[start + j] += delta
                candidate = state_from_vector(env, vector)
                evaluations += 1
                gain = float(candidate.score - current.score)
                slope = gain / delta
                if best is None or slope > best[0]:
                    best = (slope, i, j, delta, gain)

        if best is None or best[0] <= tol:
            break

        _, i, j, probe_delta, probe_gain = best
        maximum = float(current.vector[start + i])
        alphas = {probe_delta, maximum}
        if maximum > probe_delta * (1.0 + 1e-12):
            alphas.update(float(x) for x in np.geomspace(probe_delta, maximum, num=line_points))

        best_candidate = None
        best_gain = -np.inf
        for alpha in sorted(alphas):
            alpha = min(alpha, maximum)
            vector = current.vector.copy()
            vector[start + i] -= alpha
            vector[start + j] += alpha
            candidate = state_from_vector(env, vector)
            evaluations += 1
            gain = float(candidate.score - current.score)
            if gain > best_gain:
                best_candidate = candidate
                best_gain = gain

        if best_candidate is None or best_gain <= tol:
            if probe_gain <= tol:
                break
            vector = current.vector.copy()
            vector[start + i] -= probe_delta
            vector[start + j] += probe_delta
            best_candidate = state_from_vector(env, vector)
            evaluations += 1
            best_gain = float(best_candidate.score - current.score)

        if best_gain <= tol:
            break

        current = best_candidate
        accepted += 1

    return current, evaluations, accepted


def _stationarity_gap(env, current, sl: slice, total: float, eps: float = 1e-4):
    """Largest feasible one-sided pairwise improvement at the returned point."""

    best = 0.0
    evaluations = 0
    start, stop = sl.start, sl.stop
    values = current.vector[sl]
    for i in range(stop - start):
        delta = min(eps * total, float(values[i]))
        if delta <= 1e-14:
            continue
        for j in range(stop - start):
            if i == j:
                continue
            vector = current.vector.copy()
            vector[start + i] -= delta
            vector[start + j] += delta
            candidate = state_from_vector(env, vector)
            evaluations += 1
            best = max(best, float(candidate.score - current.score))
    return best, evaluations


def solve(
    env,
    *,
    max_iter: int = FROZEN_MAX_ITER,
    tol: float = FROZEN_TOL,
    stationarity_tol: float = FROZEN_STATIONARITY_TOL,
    gradient_eps: float = FROZEN_GRADIENT_EPS,
    initial_rho: float = 1.0,
    rho_growth: float = 2.0,
    max_backtracks: int = 16,
    seed: int = 0,
):
    """Corrected continuous alternating baseline.

    The power and common-rate simplex blocks use feasible pairwise transfers.
    The STAR-RIS block intentionally preserves the legacy finite-difference /
    proximal update so the audited change is isolated to the simplex geometry.

    This is a deterministic local baseline, not a global optimum or upper bound.
    """

    del seed  # deterministic by design
    cfg = env.config
    slices = physical_slices(cfg.n_users, cfg.n_ris)
    ris_indices = np.arange(slices["beta"].start, slices["theta_r"].stop)

    current = state_from_action(env, analytical_action(env))
    history = [float(current.score)]
    evaluations = 1
    accepted_steps = 0
    power_gap = np.inf
    common_gap = np.inf
    gaps_match_current = False
    termination_reason = "max_iter"

    for _ in range(max_iter):
        previous = float(current.score)
        gaps_match_current = False

        current, used, accepted = _pairwise_simplex_ascent(
            env, current, slices["powers"], cfg.p_max
        )
        evaluations += used
        accepted_steps += accepted

        current, used, accepted = _pairwise_simplex_ascent(
            env, current, slices["common"], 1.0
        )
        evaluations += used
        accepted_steps += accepted

        gradient, used = _block_gradient(env, current.vector, ris_indices, gradient_eps)
        evaluations += used
        if np.linalg.norm(gradient[ris_indices]) >= 1e-12:
            rho = initial_rho
            for _ in range(max_backtracks):
                proposal_vector = _proximal_surrogate_maximizer(
                    current.vector, gradient, ris_indices, rho, env
                )
                proposal = state_from_vector(env, proposal_vector)
                evaluations += 1
                direction = proposal.vector - current.vector
                true_gain = float(proposal.score - current.score)
                surrogate_gain = float(
                    gradient @ direction - 0.5 * rho * np.dot(direction, direction)
                )
                if true_gain >= -1e-10 and surrogate_gain >= -1e-10:
                    if np.linalg.norm(direction) > 1e-12:
                        accepted_steps += 1
                    current = proposal
                    break
                rho *= rho_growth

        history.append(float(current.score))
        if abs(current.score - previous) / max(1.0, abs(previous)) < tol:
            power_gap, used = _stationarity_gap(
                env, current, slices["powers"], cfg.p_max
            )
            evaluations += used
            common_gap, used = _stationarity_gap(
                env, current, slices["common"], 1.0
            )
            evaluations += used
            gaps_match_current = True
            if power_gap < stationarity_tol and common_gap < stationarity_tol:
                termination_reason = "objective_and_simplex_stationarity"
                break

    if not gaps_match_current:
        power_gap, used = _stationarity_gap(env, current, slices["powers"], cfg.p_max)
        evaluations += used
        common_gap, used = _stationarity_gap(env, current, slices["common"], 1.0)
        evaluations += used

    metrics = dict(current.metrics)
    metrics.update(
        solver=ALGORITHM_VERSION,
        algorithm_version=ALGORITHM_VERSION,
        max_iter=int(max_iter),
        objective_tolerance=float(tol),
        stationarity_tolerance=float(stationarity_tol),
        termination_reason=termination_reason,
        iterations=len(history) - 1,
        evaluations=int(evaluations),
        accepted_steps=int(accepted_steps),
        initialization="analytical_ris",
        objective_history=history,
        power_stationarity_gap=float(power_gap),
        common_stationarity_gap=float(common_gap),
    )
    raw = encode_action(current.action, cfg.p_max, cfg.action_parameterization)
    return raw, metrics
