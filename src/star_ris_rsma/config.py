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
    qos_penalty_linear: float = 2.0
    qos_penalty_quadratic: float = 0.0

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

    def __post_init__(self) -> None:
        if self.observation_normalization not in {"global_l2", "blockwise_v2"}:
            raise ValueError(
                "observation_normalization must be 'global_l2' or 'blockwise_v2'"
            )
        if self.action_parameterization not in {"legacy_v1", "physical_v3"}:
            raise ValueError(
                "action_parameterization must be 'legacy_v1' or 'physical_v3'"
            )
        if self.td3_critic_loss not in {"mse", "huber"}:
            raise ValueError("td3_critic_loss must be 'mse' or 'huber'")
        if self.qos_penalty_linear < 0 or self.qos_penalty_quadratic < 0:
            raise ValueError("QoS penalties must be non-negative")
        if not 0 <= self.validation_qos_fraction_target <= 1:
            raise ValueError("validation_qos_fraction_target must be in [0, 1]")
        if not 0 <= self.validation_all_qos_target <= 1:
            raise ValueError("validation_all_qos_target must be in [0, 1]")
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

    def legacy_config_hash_v1(self) -> str:
        data = self.to_dict()
        legacy = {field: data[field] for field in _LEGACY_V1_FIELDS}
        return self._hash_payload(legacy)
