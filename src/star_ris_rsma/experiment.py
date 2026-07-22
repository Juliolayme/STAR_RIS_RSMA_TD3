from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .agents import DDPGAgent, PPOAgent, TD3Agent
from .baselines import analytical_ris, ao_grid, ao_sca
from .config import ExperimentConfig
from .env import StarRisRsmaEnv
from .replay import ReplayBuffer


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_off_policy(method: str, cfg: ExperimentConfig, seed: int, output: Path) -> None:
    np.random.seed(seed); torch.manual_seed(seed)
    env = StarRisRsmaEnv(cfg, seed)
    agent_cls = TD3Agent if method == "td3" else DDPGAgent
    agent = agent_cls(env.observation_dim, env.action_dim, cfg.hidden_dim, cfg.gamma, cfg.tau, _device())
    replay = ReplayBuffer(env.observation_dim, env.action_dim, cfg.replay_size, seed)
    obs = env.reset()
    rows = []
    for step in range(cfg.train_steps):
        action = np.random.uniform(-1, 1, env.action_dim).astype(np.float32) if step < cfg.warmup_steps else agent.act(obs, noise_std=0.15)
        next_obs, reward, done, info = env.step(action)
        replay.add(obs, action, reward, next_obs, done)
        obs = env.reset() if done else next_obs
        if replay.size >= cfg.batch_size:
            losses = agent.update(replay.sample(cfg.batch_size))
        else:
            losses = {}
        if step % 1000 == 0:
            rows.append({"step": step, "reward": reward, "sum_rate": info["sum_rate"], **losses})
    output.mkdir(parents=True, exist_ok=True)
    torch.save(agent.actor.state_dict(), output / "actor.pt")
    pd.DataFrame(rows).to_csv(output / "training.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({"method": method, "seed": seed, "device": _device(), "config": cfg.to_dict()}, indent=2))


def train_ppo(cfg: ExperimentConfig, seed: int, output: Path) -> None:
    np.random.seed(seed); torch.manual_seed(seed)
    env = StarRisRsmaEnv(cfg, seed)
    agent = PPOAgent(env.observation_dim, env.action_dim, cfg.hidden_dim, _device())
    obs = env.reset(); global_step = 0; logs = []
    horizon = min(2048, max(128, cfg.episode_length * 4))
    while global_step < cfg.train_steps:
        ob, ac, lp, rw, vl, dn = [], [], [], [], [], []
        for _ in range(min(horizon, cfg.train_steps - global_step)):
            action, logp, value = agent.act(obs, deterministic=False)
            nxt, reward, done, info = env.step(action)
            ob.append(obs); ac.append(action); lp.append(logp); rw.append(reward); vl.append(value); dn.append(done)
            obs = env.reset() if done else nxt
            global_step += 1
        returns = np.zeros(len(rw), dtype=np.float32); advantages = np.zeros(len(rw), dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(rw))):
            running = rw[t] + cfg.gamma * running * (1.0 - float(dn[t]))
            returns[t] = running
            advantages[t] = returns[t] - vl[t]
        losses = agent.update(np.asarray(ob), np.asarray(ac), np.asarray(lp), returns, advantages)
        logs.append({"step": global_step, "mean_reward": float(np.mean(rw)), **losses})
    output.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), output / "ppo.pt")
    pd.DataFrame(logs).to_csv(output / "training.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({"method": "ppo", "seed": seed, "device": _device(), "config": cfg.to_dict()}, indent=2))


def evaluate_solver(method: str, cfg: ExperimentConfig, seed: int, start: int, count: int, output: Path) -> None:
    rows = []
    for scenario in range(start, start + count):
        env = StarRisRsmaEnv(cfg, seed + scenario)
        env.reset()
        if method == "ao_sca": _, metrics = ao_sca(env, seed=seed + scenario)
        elif method == "ao_grid": _, metrics = ao_grid(env, seed=seed + scenario)
        elif method == "analytical_ris": _, metrics = analytical_ris(env)
        else: raise ValueError(method)
        rows.append({
            "method": method, "seed": seed, "scenario": scenario,
            "sum_rate": metrics["sum_rate"], "reward": metrics["reward"],
            "qos_fraction": metrics["qos_fraction"], "all_qos": metrics["all_qos"],
            "violation": metrics["violation"],
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
