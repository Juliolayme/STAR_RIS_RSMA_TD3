from __future__ import annotations

import numpy as np

from ..action import action_dim
from ..env import StarRisRsmaEnv


def solve(env: StarRisRsmaEnv) -> tuple[np.ndarray, dict[str, object]]:
    if env.channel is None:
        env.reset()
    c = env.config
    ch = env.channel
    assert ch is not None
    raw = np.zeros(action_dim(c.n_users, c.n_ris), dtype=float)
    i = (c.n_users + 1) + c.n_users
    raw[i:i+c.n_ris] = 0.0  # beta=0.5
    i += c.n_ris
    aggregate = np.sum(ch.h_ru.conj() * ch.g_br[None, :], axis=0)
    phase = -np.angle(aggregate)
    raw[i:i+c.n_ris] = np.arctanh(np.clip(phase / np.pi, -0.999, 0.999))
    i += c.n_ris
    raw[i:i+c.n_ris] = raw[i-c.n_ris:i]
    return raw, env.evaluate_raw_action(raw)
