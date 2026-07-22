from __future__ import annotations

import numpy as np

from ..action import action_dim


def default_action(n_users: int, n_ris: int) -> np.ndarray:
    return np.zeros(action_dim(n_users, n_ris), dtype=np.float64)


def clip_raw_action(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), -6.0, 6.0)
