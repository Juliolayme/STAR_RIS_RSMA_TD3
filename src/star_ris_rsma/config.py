from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
