from __future__ import annotations

import numpy as np

from ..action import DecodedAction, encode_action
from ..env import StarRisRsmaEnv


def analytical_action(env: StarRisRsmaEnv) -> DecodedAction:
    """Phase-alignment heuristic with equal power/common allocation and beta=0.5."""
    if env.channel is None:
        env.reset()
    c = env.config
    ch = env.channel
    assert ch is not None
    aggregate = np.sum(ch.h_ru.conj() * ch.g_br[None, :], axis=0)
    phase = -np.angle(aggregate)
    return DecodedAction(
        powers=np.full(c.n_users + 1, c.p_max / (c.n_users + 1), dtype=float),
        common_fractions=np.full(c.n_users, 1.0 / c.n_users, dtype=float),
        beta_t=np.full(c.n_ris, 0.5, dtype=float),
        theta_t=phase.copy(),
        theta_r=phase.copy(),
    )


def solve(env: StarRisRsmaEnv) -> tuple[np.ndarray, dict[str, object]]:
    action = analytical_action(env)
    metrics = dict(env.evaluate_decoded_action(action))
    metrics.update({
        "solver": "analytical_ris_equal_allocation",
        "iterations": 0,
        "evaluations": 1,
        "initialization": "closed_form_phase_alignment",
    })
    return encode_action(
        action,
        env.config.p_max,
        env.config.action_parameterization,
    ), metrics
