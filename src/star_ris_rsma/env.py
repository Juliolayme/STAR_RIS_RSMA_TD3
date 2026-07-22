from __future__ import annotations

import numpy as np

from .action import action_dim, decode_action
from .config import ExperimentConfig
from .physics import ChannelSample, effective_channels, generate_channel, rsma_rates


class StarRisRsmaEnv:
    """Minimal Gym-like environment without an external Gym dependency."""

    def __init__(self, config: ExperimentConfig, seed: int = 0):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.channel: ChannelSample | None = None
        self.step_count = 0
        self.observation_dim = 2 * (
            config.n_users + config.n_ris + config.n_users * config.n_ris
        ) + config.n_users
        self.action_dim = action_dim(config.n_users, config.n_ris)

    def _observation(self) -> np.ndarray:
        assert self.channel is not None
        parts = [
            self.channel.h_direct.real,
            self.channel.h_direct.imag,
            self.channel.g_br.real,
            self.channel.g_br.imag,
            self.channel.h_ru.real.reshape(-1),
            self.channel.h_ru.imag.reshape(-1),
            self.channel.user_side.astype(float),
        ]
        obs = np.concatenate(parts).astype(np.float32)
        scale = max(float(np.linalg.norm(obs)), 1.0)
        return obs / scale

    def reset(self, *, channel: ChannelSample | None = None) -> np.ndarray:
        self.channel = channel or generate_channel(self.rng, self.config.n_users, self.config.n_ris)
        self.step_count = 0
        return self._observation()

    def evaluate_raw_action(self, raw_action: np.ndarray) -> dict[str, np.ndarray | float | bool]:
        if self.channel is None:
            raise RuntimeError("Call reset() before evaluate_raw_action().")
        a = decode_action(raw_action, self.config.n_users, self.config.n_ris, self.config.p_max)
        h_eff = effective_channels(self.channel, a.beta_t, a.theta_t, a.theta_r)
        metrics = rsma_rates(h_eff, a.powers, a.common_fractions, self.config.noise_power)
        user_rates = np.asarray(metrics["user_rates"])
        qos_fraction = float(np.mean(user_rates >= self.config.qos_min))
        all_qos = bool(np.all(user_rates >= self.config.qos_min))
        violation = float(np.maximum(self.config.qos_min - user_rates, 0.0).sum())
        reward = float(metrics["sum_rate"]) - 2.0 * violation
        return {**metrics, "qos_fraction": qos_fraction, "all_qos": all_qos, "violation": violation, "reward": reward}

    def step(self, raw_action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        metrics = self.evaluate_raw_action(raw_action)
        self.step_count += 1
        done = self.step_count >= self.config.episode_length
        if not done:
            self.channel = generate_channel(self.rng, self.config.n_users, self.config.n_ris)
        return self._observation(), float(metrics["reward"]), done, metrics
