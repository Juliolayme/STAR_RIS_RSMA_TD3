from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


_LEGACY_V1_FIELDS = (
    "n_ris",
    "n_users",
    "p_max",
    "noise_power",
    "qos_min",
    "episode_length",
    "gamma",
    "tau",
    "hidden_dim",
    "batch_size",
    "replay_size",
    "warmup_steps",
    "train_steps",
    "eval_scenarios",
    "validation_interval",
    "validation_scenarios",
    "exploration_noise",
    "ppo_horizon",
    "gae_lambda",
    "train_bank_path",
    "validation_bank_path",
    "test_bank_path",
)

# Fields introduced after the frozen v2 checkpoint format. They must be removed
# when reconstructing the historical v2 hash so retained checkpoints remain
# evaluable after adding the constrained physical-action protocol.
_POST_V2_FIELDS = (
    "action_parameterization",
    "qos_dual_enabled",
    "qos_dual_initial",
    "qos_dual_learning_rate",
    "qos_dual_target_violation",
    "qos_dual_update_interval",
    "qos_dual_ema_beta",
    "qos_dual_min",
    "qos_dual_max",
    "validate_at_initialization",
    "actor_small_final_init",
    # Matched-implementation baseline controls added for the V6 fair
    # comparison. Excluded here so retained v2 checkpoints stay evaluable.
    "ddpg_actor_lr",
    "ddpg_critic_lr",
    "ddpg_gradient_clip_norm",
    "ddpg_critic_loss",
    "ddpg_layer_norm",
    "ppo_lr",
    "ppo_gradient_clip_norm",
    "ppo_layer_norm",
    "ppo_epochs",
    "ppo_minibatch_size",
    "ppo_clip_ratio",
    "ppo_entropy_coef",
    "ppo_value_coef",
    "retained_candidate_checkpoints",
)


@dataclass(slots=True)
class ExperimentConfig:
    n_ris: int = 32
    n_users: int = 4
    p_max: float = 1.0
    noise_power: float = 1e-3
    qos_min: float = 0.5
    episode_length: int = 32
    gamma: float = 0.99
    tau: float = 0.005
    hidden_dim: int = 256
    batch_size: int = 256
    replay_size: int = 200_000
    warmup_steps: int = 2_000
    train_steps: int = 100_000
    eval_scenarios: int = 1_000
    validation_interval: int = 5_000
    validation_scenarios: int = 128
    exploration_noise: float = 0.15
    ppo_horizon: int = 2_048
    gae_lambda: float = 0.95
    train_bank_path: str | None = None
    validation_bank_path: str | None = None
    test_bank_path: str | None = None

    # Backward-compatible environment controls. Existing YAML files reproduce
    # the original experiment because these defaults match the old behaviour.
    observation_normalization: str = "global_l2"
    action_parameterization: str = "legacy_v1"
    validate_at_initialization: bool = False
    actor_small_final_init: bool = False
    qos_penalty_linear: float = 2.0
    qos_penalty_quadratic: float = 0.0

    # Optional projected dual-ascent controller for constrained TD3 training.
    # Disabled by default so all historical v1/v2 configs remain unchanged.
    qos_dual_enabled: bool = False
    qos_dual_initial: float = 8.0
    qos_dual_learning_rate: float = 20.0
    qos_dual_target_violation: float = 1e-3
    qos_dual_update_interval: int = 1_000
    qos_dual_ema_beta: float = 0.95
    qos_dual_min: float = 4.0
    qos_dual_max: float = 64.0

    # QoS-first validation checkpoint selection used by experiment_v2.
    validation_qos_fraction_target: float = 0.95
    validation_all_qos_target: float = 0.80
    validation_violation_tolerance: float = 0.01

    # Dimension-aware exploration schedule used by experiment_v2.
    exploration_noise_final: float = 0.15
    exploration_decay_steps: int = 100_000

    # TD3 stability controls. Defaults reproduce the original TD3 settings.
    td3_actor_lr: float = 3e-4
    td3_critic_lr: float = 3e-4
    td3_policy_delay: int = 2
    td3_target_noise: float = 0.2
    td3_noise_clip: float = 0.5
    td3_gradient_clip_norm: float = 0.0
    td3_noise_reference_dim: int = 0
    td3_critic_loss: str = "mse"
    td3_layer_norm: bool = False

    # DDPG stability controls. Defaults reproduce the original vanilla DDPG,
    # so historical configs are untouched; the V6 configs opt into the same
    # settings TD3 uses so the comparison is not confounded by implementation.
    ddpg_actor_lr: float = 3e-4
    ddpg_critic_lr: float = 3e-4
    ddpg_gradient_clip_norm: float = 0.0
    ddpg_critic_loss: str = "mse"
    ddpg_layer_norm: bool = False

    # PPO controls. Defaults reproduce the previously hard-coded values.
    ppo_lr: float = 3e-4
    ppo_gradient_clip_norm: float = 1.0
    ppo_layer_norm: bool = False
    ppo_epochs: int = 10
    ppo_minibatch_size: int = 0
    ppo_clip_ratio: float = 0.2
    ppo_entropy_coef: float = 1e-3
    ppo_value_coef: float = 0.5

    # Extra validation checkpoints kept so a later change to the selection
    # criterion can be applied by re-selection instead of retraining.
    retained_candidate_checkpoints: int = 0

    def __post_init__(self) -> None:
        if self.observation_normalization not in {"global_l2", "blockwise_v2"}:
            raise ValueError(
                "observation_normalization must be 'global_l2' or 'blockwise_v2'"
            )
        if self.action_parameterization not in {
            "legacy_v1", "physical_v3", "physical_v5_hard", "physical_v5_soft",
            "physical_v6_soft_anchor",
        }:
            raise ValueError(
                "unsupported action_parameterization"
            )
        if self.td3_critic_loss not in {"mse", "huber"}:
            raise ValueError("td3_critic_loss must be 'mse' or 'huber'")
        if self.ddpg_critic_loss not in {"mse", "huber"}:
            raise ValueError("ddpg_critic_loss must be 'mse' or 'huber'")
        if self.ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be positive")
        if self.ppo_minibatch_size < 0:
            raise ValueError("ppo_minibatch_size must be non-negative")
        if self.ppo_clip_ratio <= 0:
            raise ValueError("ppo_clip_ratio must be positive")
        if self.retained_candidate_checkpoints < 0:
            raise ValueError("retained_candidate_checkpoints must be non-negative")
        if self.qos_penalty_linear < 0 or self.qos_penalty_quadratic < 0:
            raise ValueError("QoS penalties must be non-negative")
        if self.qos_dual_initial < 0 or self.qos_dual_min < 0:
            raise ValueError("QoS dual penalties must be non-negative")
        if self.qos_dual_max < self.qos_dual_min:
            raise ValueError("qos_dual_max must be >= qos_dual_min")
        if not self.qos_dual_min <= self.qos_dual_initial <= self.qos_dual_max:
            raise ValueError("qos_dual_initial must lie within [qos_dual_min, qos_dual_max]")
        if self.qos_dual_learning_rate < 0:
            raise ValueError("qos_dual_learning_rate must be non-negative")
        if self.qos_dual_target_violation < 0:
            raise ValueError("qos_dual_target_violation must be non-negative")
        if self.qos_dual_update_interval <= 0:
            raise ValueError("qos_dual_update_interval must be positive")
        if not 0 <= self.qos_dual_ema_beta < 1:
            raise ValueError("qos_dual_ema_beta must be in [0, 1)")
        if not 0 <= self.validation_qos_fraction_target <= 1:
            raise ValueError("validation_qos_fraction_target must be in [0, 1]")
        if not 0 <= self.validation_all_qos_target <= 1:
            raise ValueError("validation_all_qos_target must be in [0, 1]")
        if self.validation_violation_tolerance < 0:
            raise ValueError("validation_violation_tolerance must be non-negative")
        if self.exploration_decay_steps <= 0:
            raise ValueError("exploration_decay_steps must be positive")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def _hash_payload(payload_dict: dict[str, Any]) -> str:
        payload = json.dumps(payload_dict, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def config_hash(self) -> str:
        return self._hash_payload(self.to_dict())

    def legacy_config_hash_v2(self) -> str:
        data = self.to_dict()
        for field in _POST_V2_FIELDS:
            data.pop(field, None)
        return self._hash_payload(data)

    def legacy_config_hash_v1(self) -> str:
        data = self.to_dict()
        legacy = {field: data[field] for field in _LEGACY_V1_FIELDS}
        return self._hash_payload(legacy)
