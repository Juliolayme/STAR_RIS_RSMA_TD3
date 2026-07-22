from __future__ import annotations

import numpy as np

from ..action import encode_action
from ..env import StarRisRsmaEnv
from .analytical_ris import analytical_action
from .common import merit, physical_slices, project_physical, state_from_action, state_from_vector


def _block_gradient(
    env: StarRisRsmaEnv,
    x: np.ndarray,
    indices: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, int]:
    """Central finite-difference gradient of the exact merit in physical variables."""
    gradient = np.zeros_like(x)
    evaluations = 0
    for j in indices:
        xp = x.copy(); xm = x.copy()
        xp[j] += eps; xm[j] -= eps
        xp = project_physical(xp, env.config.n_users, env.config.n_ris, env.config.p_max)
        xm = project_physical(xm, env.config.n_users, env.config.n_ris, env.config.p_max)
        fp = merit(state_from_vector(env, xp).metrics)
        fm = merit(state_from_vector(env, xm).metrics)
        evaluations += 2
        direction = xp - xm
        norm_sq = float(np.dot(direction, direction))
        if norm_sq >= 1e-16:
            gradient += ((fp - fm) / norm_sq) * direction
    return gradient, evaluations


def _proximal_surrogate_maximizer(
    x: np.ndarray,
    gradient: np.ndarray,
    indices: np.ndarray,
    rho: float,
    env: StarRisRsmaEnv,
) -> np.ndarray:
    """Solve max g^T(z-x) - rho/2 ||z-x||^2 over the feasible block."""
    proposal = x.copy()
    proposal[indices] = x[indices] + gradient[indices] / rho
    return project_physical(proposal, env.config.n_users, env.config.n_ris, env.config.p_max)


def solve(
    env: StarRisRsmaEnv,
    max_iter: int = 20,
    tol: float = 1e-4,
    gradient_eps: float = 1e-3,
    initial_rho: float = 1.0,
    rho_growth: float = 2.0,
    max_backtracks: int = 16,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Two-block proximal AO-SCA on feasible physical variables.

    At each block, a concave first-order proximal surrogate is constructed:
      g(x_t)^T (z - x_t) - rho/2 ||z - x_t||^2.
    Its constrained maximizer is obtained by projection onto the power/common
    simplices, beta box and periodic phase domain. Increasing rho implements
    monotone backtracking on the exact merit. This is a local stationary-point
    method and never an upper bound or global optimum.
    """
    del seed  # deterministic by design
    initial = state_from_action(env, analytical_action(env))
    current = initial
    slices = physical_slices(env.config.n_users, env.config.n_ris)
    rsma_indices = np.arange(slices["powers"].start, slices["common"].stop)
    ris_indices = np.arange(slices["beta"].start, slices["theta_r"].stop)
    blocks = [("rsma", rsma_indices), ("star_ris", ris_indices)]
    history = [current.score]
    evaluations = 1
    accepted_steps = 0
    surrogate_records: list[dict[str, float | str]] = []

    for outer in range(max_iter):
        previous = current.score
        for block_name, indices in blocks:
            gradient, grad_evals = _block_gradient(env, current.vector, indices, gradient_eps)
            evaluations += grad_evals
            grad_norm = float(np.linalg.norm(gradient[indices]))
            if grad_norm < 1e-12:
                surrogate_records.append({"block": block_name, "rho": initial_rho, "gain": 0.0})
                continue

            rho = initial_rho
            accepted = False
            for _ in range(max_backtracks):
                proposal_vector = _proximal_surrogate_maximizer(
                    current.vector, gradient, indices, rho, env
                )
                proposal = state_from_vector(env, proposal_vector)
                evaluations += 1
                true_gain = proposal.score - current.score
                displacement = proposal.vector - current.vector
                surrogate_gain = float(
                    gradient @ displacement - 0.5 * rho * np.dot(displacement, displacement)
                )
                if true_gain >= -1e-10 and surrogate_gain >= -1e-10:
                    current = proposal
                    accepted_steps += int(np.linalg.norm(displacement) > 1e-12)
                    surrogate_records.append({
                        "block": block_name,
                        "rho": float(rho),
                        "gain": float(true_gain),
                    })
                    accepted = True
                    break
                rho *= rho_growth
            if not accepted:
                surrogate_records.append({"block": block_name, "rho": float(rho), "gain": 0.0})

        history.append(current.score)
        relative_gain = abs(current.score - previous) / max(1.0, abs(previous))
        if relative_gain < tol:
            break

    metrics = dict(current.metrics)
    metrics.update({
        "solver": "ao_sca_proximal_physical",
        "objective_history": history,
        "iterations": len(history) - 1,
        "evaluations": evaluations,
        "accepted_steps": accepted_steps,
        "surrogate_records": surrogate_records,
        "initialization": "analytical_ris_equal_allocation",
    })
    return encode_action(current.action, env.config.p_max), metrics
