from __future__ import annotations

import numpy as np

from ..action import DecodedAction, decode_action
from ..env import StarRisRsmaEnv
from ..physics import effective_channels


ABLATION_MODES = ("learned", "no_ris", "fixed_ris", "random_ris", "equal_power")


def evaluate_ablation(
    env: StarRisRsmaEnv,
    raw_action: np.ndarray,
    mode: str,
    seed: int = 0,
) -> dict[str, object]:
    if env.channel is None:
        raise RuntimeError("Call reset() before ablation evaluation")
    if mode not in ABLATION_MODES:
        raise ValueError(mode)

    action = decode_action(raw_action, env.config.n_users, env.config.n_ris, env.config.p_max)
    if mode == "equal_power":
        action = action.copy_with(
            powers=np.full(env.config.n_users + 1, env.config.p_max / (env.config.n_users + 1))
        )
    if mode == "fixed_ris":
        action = action.copy_with(
            beta_t=np.full(env.config.n_ris, 0.5),
            theta_t=np.zeros(env.config.n_ris),
            theta_r=np.zeros(env.config.n_ris),
        )
    elif mode == "random_ris":
        rng = np.random.default_rng(seed)
        action = action.copy_with(
            beta_t=rng.uniform(0.0, 1.0, env.config.n_ris),
            theta_t=rng.uniform(-np.pi, np.pi, env.config.n_ris),
            theta_r=rng.uniform(-np.pi, np.pi, env.config.n_ris),
        )

    if mode == "no_ris":
        h_eff = env.channel.h_direct.astype(np.complex128).copy()
        metrics = env.metrics_from_effective_channel(h_eff, action)
    else:
        h_eff = effective_channels(env.channel, action.beta_t, action.theta_t, action.theta_r)
        metrics = env.metrics_from_effective_channel(h_eff, action)
    return {**metrics, "ablation": mode}
