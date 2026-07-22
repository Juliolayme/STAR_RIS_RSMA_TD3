from pathlib import Path

import numpy as np
import pytest

from star_ris_rsma.checkpoints import build_agent, load_checkpoint, save_checkpoint
from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.env import StarRisRsmaEnv


def test_checkpoint_roundtrip_deterministic(tmp_path: Path):
    cfg = ExperimentConfig(n_users=2, n_ris=3, hidden_dim=16)
    env = StarRisRsmaEnv(cfg, 1)
    obs = env.reset()
    for method in ["td3", "ddpg", "ppo"]:
        agent = build_agent(method, env.observation_dim, env.action_dim, cfg, "cpu")
        path = tmp_path / f"{method}.pt"
        save_checkpoint(path, method, agent, 10, 1.5, cfg)
        loaded, payload = load_checkpoint(path, method, env.observation_dim, env.action_dim, cfg, "cpu")
        if method == "ppo":
            a1, _, _ = agent.act(obs, deterministic=True); a2, _, _ = loaded.act(obs, deterministic=True)
        else:
            a1 = agent.act(obs, 0.0); a2 = loaded.act(obs, 0.0)
        assert np.allclose(a1, a2)
        assert payload["step"] == 10


def test_checkpoint_rejects_config_mismatch(tmp_path: Path):
    cfg = ExperimentConfig(n_users=2, n_ris=3, hidden_dim=16)
    env = StarRisRsmaEnv(cfg, 1)
    agent = build_agent("td3", env.observation_dim, env.action_dim, cfg, "cpu")
    path = tmp_path / "td3.pt"
    save_checkpoint(path, "td3", agent, 1, 0.0, cfg)
    wrong = ExperimentConfig(n_users=2, n_ris=4, hidden_dim=16)
    with pytest.raises(ValueError):
        load_checkpoint(path, "td3", env.observation_dim, env.action_dim, wrong, "cpu")
