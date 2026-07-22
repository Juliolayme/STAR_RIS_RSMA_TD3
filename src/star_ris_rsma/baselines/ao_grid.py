from __future__ import annotations

import numpy as np

from ..env import StarRisRsmaEnv
from .analytical_ris import solve as analytical_solve


def solve(env: StarRisRsmaEnv, rounds: int = 3, candidates: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    best, metrics = analytical_solve(env)
    best_score = float(metrics["reward"])
    for _ in range(rounds):
        scale = 1.0
        for _ in range(candidates):
            trial = np.clip(best + rng.choice([-1.0, 0.0, 1.0], size=best.shape) * scale, -6.0, 6.0)
            m = env.evaluate_raw_action(trial)
            if float(m["reward"]) > best_score:
                best, metrics, best_score = trial, m, float(m["reward"])
        scale *= 0.5
    return best, metrics
