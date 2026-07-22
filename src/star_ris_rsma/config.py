from __future__ import annotations

from dataclasses import asdict, dataclass
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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
