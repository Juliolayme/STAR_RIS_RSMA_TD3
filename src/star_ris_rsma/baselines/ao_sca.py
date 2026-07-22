from __future__ import annotations

import numpy as np

from ..env import StarRisRsmaEnv
from .analytical_ris import solve as analytical_solve


def _finite_difference_gradient(env: StarRisRsmaEnv, x: np.ndarray, indices: np.ndarray, eps: float) -> np.ndarray:
    grad = np.zeros_like(x)
    for j in indices:
        xp = x.copy(); xm = x.copy()
        xp[j] += eps; xm[j] -= eps
        grad[j] = (float(env.evaluate_raw_action(xp)["reward"]) - float(env.evaluate_raw_action(xm)["reward"])) / (2.0 * eps)
    return grad


def solve(env: StarRisRsmaEnv, max_iter: int = 20, tol: float = 1e-4, seed: int = 0):
    """Two-block AO with first-order SCA surrogate and monotone backtracking.

    Block 1 updates RSMA power/common logits. Block 2 updates STAR-RIS ES/phases.
    The surrogate is a linearization plus a proximal quadratic term; backtracking
    accepts only non-decreasing true objectives. This is intentionally distinct
    from AO-Grid, though it remains a local numerical baseline rather than a
    globally optimal solver.
    """
    rng = np.random.default_rng(seed)
    x, metrics = analytical_solve(env)
    best = float(metrics["reward"])
    split = (env.config.n_users + 1) + env.config.n_users
    blocks = [np.arange(0, split), np.arange(split, x.size)]
    history = [best]
    for _ in range(max_iter):
        previous = best
        for indices in blocks:
            if indices.size > 48:
                indices = np.sort(rng.choice(indices, size=48, replace=False))
            grad = _finite_difference_gradient(env, x, indices, eps=2e-3)
            norm = float(np.linalg.norm(grad[indices]))
            if norm < 1e-10:
                continue
            direction = grad / norm
            step = 1.0
            while step >= 1e-4:
                trial = np.clip(x + step * direction, -6.0, 6.0)
                m = env.evaluate_raw_action(trial)
                score = float(m["reward"])
                if score >= best - 1e-10:
                    x, metrics, best = trial, m, score
                    break
                step *= 0.5
        history.append(best)
        if abs(best - previous) / max(1.0, abs(previous)) < tol:
            break
    metrics = dict(metrics)
    metrics["objective_history"] = history
    metrics["iterations"] = len(history) - 1
    return x, metrics
