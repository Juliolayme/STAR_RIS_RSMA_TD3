from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .physics import ChannelSample, generate_channel


@dataclass(slots=True)
class ScenarioBank:
    h_direct: np.ndarray
    g_br: np.ndarray
    h_ru: np.ndarray
    user_side: np.ndarray
    metadata: dict[str, object]

    def __len__(self) -> int:
        return int(self.h_direct.shape[0])

    @property
    def n_users(self) -> int:
        return int(self.h_direct.shape[1])

    @property
    def n_ris(self) -> int:
        return int(self.g_br.shape[1])

    def channel(self, index: int) -> ChannelSample:
        if not 0 <= index < len(self):
            raise IndexError(index)
        return ChannelSample(
            h_direct=self.h_direct[index].copy(),
            g_br=self.g_br[index].copy(),
            h_ru=self.h_ru[index].copy(),
            user_side=self.user_side[index].copy(),
        )

    def validate(self, cfg: ExperimentConfig | None = None) -> None:
        count = len(self)
        if self.g_br.shape[0] != count or self.h_ru.shape[0] != count or self.user_side.shape[0] != count:
            raise ValueError("ScenarioBank arrays have inconsistent scenario counts")
        if self.h_ru.shape[1:] != (self.n_users, self.n_ris):
            raise ValueError("Invalid h_ru shape")
        if self.user_side.shape[1:] != (self.n_users,):
            raise ValueError("Invalid user_side shape")
        if cfg is not None and (cfg.n_users != self.n_users or cfg.n_ris != self.n_ris):
            raise ValueError(
                f"Bank is K={self.n_users}, N={self.n_ris}; config is K={cfg.n_users}, N={cfg.n_ris}"
            )
        for array in (self.h_direct, self.g_br, self.h_ru):
            if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
                raise ValueError("ScenarioBank contains non-finite channels")

    def save(self, path: str | Path) -> None:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            h_direct=self.h_direct,
            g_br=self.g_br,
            h_ru=self.h_ru,
            user_side=self.user_side,
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path, cfg: ExperimentConfig | None = None) -> "ScenarioBank":
        source = Path(path)
        with np.load(source, allow_pickle=False) as data:
            bank = cls(
                h_direct=data["h_direct"],
                g_br=data["g_br"],
                h_ru=data["h_ru"],
                user_side=data["user_side"],
                metadata=json.loads(str(data["metadata"].item())),
            )
        bank.validate(cfg)
        return bank

    def checksum(self) -> str:
        digest = hashlib.sha256()
        for array in (self.h_direct, self.g_br, self.h_ru, self.user_side):
            digest.update(np.ascontiguousarray(array).view(np.uint8))
        digest.update(json.dumps(self.metadata, sort_keys=True).encode())
        return digest.hexdigest()


def generate_bank(cfg: ExperimentConfig, count: int, seed: int, split: str) -> ScenarioBank:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    samples = [generate_channel(rng, cfg.n_users, cfg.n_ris) for _ in range(count)]
    bank = ScenarioBank(
        h_direct=np.stack([s.h_direct for s in samples]),
        g_br=np.stack([s.g_br for s in samples]),
        h_ru=np.stack([s.h_ru for s in samples]),
        user_side=np.stack([s.user_side for s in samples]),
        metadata={
            "split": split,
            "seed": int(seed),
            "count": int(count),
            "n_users": int(cfg.n_users),
            "n_ris": int(cfg.n_ris),
            "generator": "star_ris_rsma.physics.generate_channel/v1",
        },
    )
    bank.validate(cfg)
    return bank


def assert_disjoint(*banks: ScenarioBank) -> None:
    seen: set[bytes] = set()
    for bank in banks:
        for idx in range(len(bank)):
            fingerprint = hashlib.sha256(
                np.ascontiguousarray(bank.h_direct[idx]).view(np.uint8).tobytes()
                + np.ascontiguousarray(bank.g_br[idx]).view(np.uint8).tobytes()
                + np.ascontiguousarray(bank.h_ru[idx]).view(np.uint8).tobytes()
            ).digest()
            if fingerprint in seen:
                raise ValueError("Scenario banks are not disjoint")
            seen.add(fingerprint)
