from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .action import DecodedAction, action_dim, decode_action
from .config import ExperimentConfig
from .physics import ChannelSample, effective_channels, generate_channel, rsma_rates


class StarRisRsmaEnv:
    """Minimal Gym-like environment with an optional locked channel sampler."""

    def __init__(self, config: ExperimentConfig, seed: int = 0):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.channel: ChannelSample | None = None
        self.step_count = 0
        self._channel_sampler: Callable[[], ChannelSample] | None = None
        self.observation_dim = 2 * (
            config.n_users + config.n_ris + config.n_users * config.n_ris
        ) + config.n_users
        self.action_dim = action_dim(config.n_users, config.n_ris)

    def set_channel_sampler(self, sampler: Callable[[], ChannelSample] | None) -> None:
        self._channel_sampler = sampler

    def _sample_channel(self) -> ChannelSample:
        if self._channel_sampler is not None:
            return self._channel_sampler()
        return generate_channel(self.rng, self.config.n_users, self.config.n_ris)

    def _observation(self) -> np.ndarray:
        assert self.channel is not None
        if self.config.observation_normalization == "blockwise_v2":
            # Match the scales used by physics.generate_channel. Unlike one global
            # L2 norm, this keeps each channel coefficient O(1) as N grows.
            parts = [
                self.channel.h_direct.real / 0.35,
                self.channel.h_direct.imag / 0.35,
                self.channel.g_br.real / 1.0,
                self.channel.g_br.imag / 1.0,
                self.channel.h_ru.real.reshape(-1) / 0.75,
                self.channel.h_ru.imag.reshape(-1) / 0.75,
                2.0 * self.channel.user_side.astype(float) - 1.0,
            ]
            return np.clip(np.concatenate(parts), -8.0, 8.0).astype(np.float32)

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
        self.channel = channel if channel is not None else self._sample_channel()
        self.step_count = 0
        return self._observation()

    def metrics_from_effective_channel(self, h_eff: np.ndarray, action: DecodedAction) -> dict[str, object]:
        metrics = rsma_rates(h_eff, action.powers, action.common_fractions, self.config.noise_power)
        user_rates = np.asarray(metrics["user_rates"])
        qos_fraction = float(np.mean(user_rates >= self.config.qos_min))
        all_qos = bool(np.all(user_rates >= self.config.qos_min))
        deficits = np.maximum(self.config.qos_min - user_rates, 0.0)
        violation = float(deficits.sum())
        violation_squared = float(np.square(deficits).sum())
        reward = (
            float(metrics["sum_rate"])
            - self.config.qos_penalty_linear * violation
            - self.config.qos_penalty_quadratic * violation_squared
        )
        return {
            **metrics,
            "qos_fraction": qos_fraction,
            "all_qos": all_qos,
            "violation": violation,
            "violation_squared": violation_squared,
            "reward": reward,
        }

    def evaluate_decoded_action(self, action: DecodedAction) -> dict[str, object]:
        if self.channel is None:
            raise RuntimeError("Call reset() before evaluate_decoded_action().")
        h_eff = effective_channels(self.channel, action.beta_t, action.theta_t, action.theta_r)
        return self.metrics_from_effective_channel(h_eff, action)

    def evaluate_raw_action(self, raw_action: np.ndarray) -> dict[str, object]:
        action = decode_action(raw_action, self.config.n_users, self.config.n_ris, self.config.p_max)
        return self.evaluate_decoded_action(action)

    def step(self, raw_action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        metrics = self.evaluate_raw_action(raw_action)
        self.step_count += 1
        done = self.step_count >= self.config.episode_length
        if not done:
            self.channel = self._sample_channel()
        return self._observation(), float(metrics["reward"]), done, metrics
