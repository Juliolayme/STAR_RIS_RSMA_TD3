from __future__ import annotations

from pathlib import Path

import torch

from .agents import DDPGAgent, PPOAgent, TD3Agent
from .config import ExperimentConfig


def build_agent(method: str, obs_dim: int, action_dim: int, cfg: ExperimentConfig, device: str):
    if method == "td3":
        return TD3Agent(
            obs_dim,
            action_dim,
            cfg.hidden_dim,
            cfg.gamma,
            cfg.tau,
            device,
            actor_lr=cfg.td3_actor_lr,
            critic_lr=cfg.td3_critic_lr,
            policy_delay=cfg.td3_policy_delay,
            target_noise=cfg.td3_target_noise,
            noise_clip=cfg.td3_noise_clip,
            gradient_clip_norm=cfg.td3_gradient_clip_norm,
            noise_reference_dim=cfg.td3_noise_reference_dim,
            critic_loss=cfg.td3_critic_loss,
            layer_norm=cfg.td3_layer_norm,
            small_final_init=cfg.actor_small_final_init,
        )
    if method == "ddpg":
        return DDPGAgent(
            obs_dim,
            action_dim,
            cfg.hidden_dim,
            cfg.gamma,
            cfg.tau,
            device,
            actor_lr=cfg.ddpg_actor_lr,
            critic_lr=cfg.ddpg_critic_lr,
            gradient_clip_norm=cfg.ddpg_gradient_clip_norm,
            critic_loss=cfg.ddpg_critic_loss,
            layer_norm=cfg.ddpg_layer_norm,
            small_final_init=cfg.actor_small_final_init,
        )
    if method == "ppo":
        return PPOAgent(
            obs_dim,
            action_dim,
            cfg.hidden_dim,
            device,
            lr=cfg.ppo_lr,
            gradient_clip_norm=cfg.ppo_gradient_clip_norm,
            layer_norm=cfg.ppo_layer_norm,
            epochs=cfg.ppo_epochs,
            minibatch_size=cfg.ppo_minibatch_size,
            clip_ratio=cfg.ppo_clip_ratio,
            entropy_coef=cfg.ppo_entropy_coef,
            value_coef=cfg.ppo_value_coef,
        )
    raise ValueError(method)


OPTIMIZER_STATE_KEYS = ("actor_opt", "q_opt", "optimizer")


def save_checkpoint(
    path: str | Path,
    method: str,
    agent,
    step: int,
    score: float,
    cfg: ExperimentConfig,
    include_optimizer: bool = True,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = agent.checkpoint_state()
    if not include_optimizer:
        # load_checkpoint always restores with inference_only=True, so dropping
        # optimizer state halves the file while staying fully evaluable.
        state = {k: v for k, v in state.items() if k not in OPTIMIZER_STATE_KEYS}
    torch.save({
        "method": method,
        "step": int(step),
        "validation_score": float(score),
        "config": cfg.to_dict(),
        "config_hash": cfg.config_hash(),
        "agent": state,
    }, target)


def load_checkpoint(path: str | Path, method: str, obs_dim: int, action_dim: int, cfg: ExperimentConfig, device: str = "cpu"):
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("method") != method:
        raise ValueError(f"Checkpoint method {payload.get('method')} does not match {method}")
    accepted_hashes = {
        cfg.config_hash(),
        cfg.legacy_config_hash_v2(),
        cfg.legacy_config_hash_v1(),
    }
    if payload.get("config_hash") not in accepted_hashes:
        raise ValueError("Checkpoint configuration hash does not match evaluation config")
    agent = build_agent(method, obs_dim, action_dim, cfg, device)
    agent.load_checkpoint_state(payload["agent"], inference_only=True)
    return agent, payload
