from __future__ import annotations

from pathlib import Path

import torch

from .agents import DDPGAgent, PPOAgent, TD3Agent
from .config import ExperimentConfig


def build_agent(method: str, obs_dim: int, action_dim: int, cfg: ExperimentConfig, device: str):
    if method == "td3":
        return TD3Agent(obs_dim, action_dim, cfg.hidden_dim, cfg.gamma, cfg.tau, device)
    if method == "ddpg":
        return DDPGAgent(obs_dim, action_dim, cfg.hidden_dim, cfg.gamma, cfg.tau, device)
    if method == "ppo":
        return PPOAgent(obs_dim, action_dim, cfg.hidden_dim, device)
    raise ValueError(method)


def save_checkpoint(path: str | Path, method: str, agent, step: int, score: float, cfg: ExperimentConfig) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "method": method,
        "step": int(step),
        "validation_score": float(score),
        "config": cfg.to_dict(),
        "config_hash": cfg.config_hash(),
        "agent": agent.checkpoint_state(),
    }, target)


def load_checkpoint(path: str | Path, method: str, obs_dim: int, action_dim: int, cfg: ExperimentConfig, device: str = "cpu"):
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("method") != method:
        raise ValueError(f"Checkpoint method {payload.get('method')} does not match {method}")
    if payload.get("config_hash") != cfg.config_hash():
        raise ValueError("Checkpoint configuration hash does not match evaluation config")
    agent = build_agent(method, obs_dim, action_dim, cfg, device)
    agent.load_checkpoint_state(payload["agent"], inference_only=True)
    return agent, payload
