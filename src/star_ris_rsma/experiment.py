from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from .agents import DDPGAgent, PPOAgent, TD3Agent
from .baselines import analytical_ris, ao_grid, ao_sca
from .checkpoints import build_agent, load_checkpoint, save_checkpoint
from .config import ExperimentConfig
from .env import StarRisRsmaEnv
from .replay import ReplayBuffer
from .scenario_bank import ScenarioBank, generate_bank


def _device(force_cpu: bool = False) -> str:
    return "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"


def _git_commit() -> str:
    explicit = os.environ.get("GIT_COMMIT") or os.environ.get("KAGGLE_KERNEL_RUN_ID")
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _bank_from_path_or_generate(
    path: str | None,
    cfg: ExperimentConfig,
    count: int,
    seed: int,
    split: str,
) -> ScenarioBank:
    if path:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(
                f"Locked {split} ScenarioBank not found: {source}. "
                "Run scripts/create_scenario_banks.py first."
            )
        return ScenarioBank.load(source, cfg)
    return generate_bank(cfg, count, seed, split)


def _attach_training_bank(env: StarRisRsmaEnv, bank: ScenarioBank, seed: int) -> None:
    rng = np.random.default_rng(seed)
    env.set_channel_sampler(lambda: bank.channel(int(rng.integers(0, len(bank)))))


def _policy_action(agent: Any, method: str, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
    if method == "ppo":
        action, _, _ = agent.act(obs, deterministic=deterministic)
        return action
    return agent.act(obs, noise_std=0.0 if deterministic else 0.15)


def evaluate_policy_on_bank(
    agent: Any,
    method: str,
    cfg: ExperimentConfig,
    bank: ScenarioBank,
    seed: int,
    max_scenarios: int | None = None,
) -> pd.DataFrame:
    count = len(bank) if max_scenarios is None else min(len(bank), max_scenarios)
    env = StarRisRsmaEnv(cfg, seed)
    rows: list[dict[str, object]] = []
    for scenario in range(count):
        obs = env.reset(channel=bank.channel(scenario))
        action = _policy_action(agent, method, obs, deterministic=True)
        metrics = env.evaluate_raw_action(action)
        rows.append({
            "method": method,
            "seed": seed,
            "scenario": scenario,
            "split": bank.metadata.get("split", "unknown"),
            "sum_rate": metrics["sum_rate"],
            "reward": metrics["reward"],
            "qos_fraction": metrics["qos_fraction"],
            "all_qos": metrics["all_qos"],
            "violation": metrics["violation"],
        })
    return pd.DataFrame(rows)


def _manifest(
    method: str,
    seed: int,
    cfg: ExperimentConfig,
    device: str,
    banks: dict[str, ScenarioBank],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "method": method,
        "seed": seed,
        "device": device,
        "git_commit": _git_commit(),
        "config": cfg.to_dict(),
        "config_hash": cfg.config_hash(),
        "scenario_banks": {
            name: {"checksum": bank.checksum(), "metadata": bank.metadata}
            for name, bank in banks.items()
        },
    }
    if extra:
        payload.update(extra)
    return payload


def _validation_step(
    agent: Any,
    method: str,
    cfg: ExperimentConfig,
    bank: ScenarioBank,
    seed: int,
    step: int,
    output: Path,
    best_score: float,
) -> float:
    raw = evaluate_policy_on_bank(
        agent, method, cfg, bank, seed, max_scenarios=cfg.validation_scenarios
    )
    raw.insert(3, "eval_step", step)
    validation_file = output / "validation_raw.csv"
    raw.to_csv(validation_file, mode="a", header=not validation_file.exists(), index=False)
    score = float(raw["reward"].mean())
    if score > best_score:
        save_checkpoint(output / "best.pt", method, agent, step, score, cfg)
        return score
    return best_score


def train_off_policy(method: str, cfg: ExperimentConfig, seed: int, output: Path) -> None:
    if method not in {"td3", "ddpg"}:
        raise ValueError(method)
    _seed_everything(seed)
    device = _device()
    train_bank = _bank_from_path_or_generate(
        cfg.train_bank_path, cfg, max(cfg.eval_scenarios, 256), 11001, "train"
    )
    validation_bank = _bank_from_path_or_generate(
        cfg.validation_bank_path, cfg, cfg.validation_scenarios, 22001, "validation"
    )
    env = StarRisRsmaEnv(cfg, seed)
    _attach_training_bank(env, train_bank, seed + 30001)
    agent = build_agent(method, env.observation_dim, env.action_dim, cfg, device)
    replay = ReplayBuffer(env.observation_dim, env.action_dim, cfg.replay_size, seed)
    obs = env.reset()
    rows: list[dict[str, object]] = []
    output.mkdir(parents=True, exist_ok=True)
    best_score = -np.inf

    for step in range(1, cfg.train_steps + 1):
        if step <= cfg.warmup_steps:
            action = np.random.uniform(-1, 1, env.action_dim).astype(np.float32)
        else:
            action = agent.act(obs, noise_std=cfg.exploration_noise)
        next_obs, reward, done, info = env.step(action)
        replay.add(obs, action, reward, next_obs, done)
        obs = env.reset() if done else next_obs
        losses = agent.update(replay.sample(cfg.batch_size)) if replay.size >= cfg.batch_size else {}
        if step == 1 or step % 1000 == 0:
            rows.append({"step": step, "reward": reward, "sum_rate": info["sum_rate"], **losses})
        if step % cfg.validation_interval == 0 or step == cfg.train_steps:
            best_score = _validation_step(
                agent, method, cfg, validation_bank, seed, step, output, best_score
            )

    save_checkpoint(output / "latest.pt", method, agent, cfg.train_steps, best_score, cfg)
    pd.DataFrame(rows).to_csv(output / "training.csv", index=False)
    (output / "manifest.json").write_text(json.dumps(_manifest(
        method, seed, cfg, device, {"train": train_bank, "validation": validation_bank},
        {"best_validation_reward": best_score},
    ), indent=2))


def train_ppo(cfg: ExperimentConfig, seed: int, output: Path) -> None:
    _seed_everything(seed)
    device = _device()
    train_bank = _bank_from_path_or_generate(
        cfg.train_bank_path, cfg, max(cfg.eval_scenarios, 256), 11001, "train"
    )
    validation_bank = _bank_from_path_or_generate(
        cfg.validation_bank_path, cfg, cfg.validation_scenarios, 22001, "validation"
    )
    env = StarRisRsmaEnv(cfg, seed)
    _attach_training_bank(env, train_bank, seed + 30001)
    agent = build_agent("ppo", env.observation_dim, env.action_dim, cfg, device)
    obs = env.reset(); global_step = 0; logs: list[dict[str, object]] = []
    horizon = min(cfg.ppo_horizon, max(128, cfg.episode_length * 4))
    output.mkdir(parents=True, exist_ok=True)
    best_score = -np.inf

    while global_step < cfg.train_steps:
        ob: list[np.ndarray] = []; ac: list[np.ndarray] = []
        lp: list[float] = []; rw: list[float] = []; vl: list[float] = []; dn: list[bool] = []
        rollout = min(horizon, cfg.train_steps - global_step)
        for _ in range(rollout):
            action, logp, value = agent.act(obs, deterministic=False)
            nxt, reward, done, _ = env.step(action)
            ob.append(obs); ac.append(action); lp.append(logp); rw.append(reward); vl.append(value); dn.append(done)
            obs = env.reset() if done else nxt
            global_step += 1
        returns = np.zeros(len(rw), dtype=np.float32)
        advantages = np.zeros(len(rw), dtype=np.float32)
        # Generalized Advantage Estimation with a bootstrap value for the tail state.
        _, _, last_value = agent.act(obs, deterministic=True)
        gae = 0.0
        for t in reversed(range(len(rw))):
            nonterminal = 1.0 - float(dn[t])
            next_value = last_value if t == len(rw) - 1 else vl[t + 1]
            delta = rw[t] + cfg.gamma * next_value * nonterminal - vl[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + vl[t]
        losses = agent.update(np.asarray(ob), np.asarray(ac), np.asarray(lp), returns, advantages)
        logs.append({"step": global_step, "mean_reward": float(np.mean(rw)), **losses})
        if global_step % cfg.validation_interval < rollout or global_step == cfg.train_steps:
            best_score = _validation_step(
                agent, "ppo", cfg, validation_bank, seed, global_step, output, best_score
            )

    save_checkpoint(output / "latest.pt", "ppo", agent, cfg.train_steps, best_score, cfg)
    pd.DataFrame(logs).to_csv(output / "training.csv", index=False)
    (output / "manifest.json").write_text(json.dumps(_manifest(
        "ppo", seed, cfg, device, {"train": train_bank, "validation": validation_bank},
        {"best_validation_reward": best_score},
    ), indent=2))


def evaluate_checkpoint(
    method: str,
    cfg: ExperimentConfig,
    checkpoint: Path,
    bank: ScenarioBank,
    seed: int,
    output: Path,
) -> None:
    env = StarRisRsmaEnv(cfg, seed)
    agent, payload = load_checkpoint(
        checkpoint, method, env.observation_dim, env.action_dim, cfg, _device()
    )
    rows = evaluate_policy_on_bank(agent, method, cfg, bank, seed)
    rows["checkpoint_step"] = int(payload["step"])
    rows["config_hash"] = cfg.config_hash()
    rows["git_commit"] = _git_commit()
    rows["bank_checksum"] = bank.checksum()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output, index=False)


def evaluate_solver(
    method: str,
    cfg: ExperimentConfig,
    seed: int,
    start: int,
    count: int,
    output: Path,
    bank: ScenarioBank | None = None,
) -> None:
    if bank is None:
        # Fallback for non-locked (smoke) runs only. Use a fixed test seed distinct
        # from the train/validation fallbacks (11001/22001) so the generated test
        # scenarios stay disjoint from training and are identical across methods and
        # solver seeds. Locked experiments must pass cfg.test_bank_path instead.
        bank = _bank_from_path_or_generate(cfg.test_bank_path, cfg, cfg.eval_scenarios, 33001, "test")
    end = min(start + count, len(bank))
    rows: list[dict[str, object]] = []
    for scenario in range(start, end):
        env = StarRisRsmaEnv(cfg, seed)
        env.reset(channel=bank.channel(scenario))
        started = time.perf_counter_ns()
        if method == "ao_sca": _, metrics = ao_sca(env, seed=seed + scenario)
        elif method == "ao_grid": _, metrics = ao_grid(env, seed=seed + scenario)
        elif method == "analytical_ris": _, metrics = analytical_ris(env)
        else: raise ValueError(method)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        rows.append({
            "method": method,
            "seed": seed,
            "scenario": scenario,
            "split": bank.metadata.get("split", "test"),
            "sum_rate": metrics["sum_rate"],
            "reward": metrics["reward"],
            "qos_fraction": metrics["qos_fraction"],
            "all_qos": metrics["all_qos"],
            "violation": metrics["violation"],
            "iterations": metrics.get("iterations", 0),
            "evaluations": metrics.get("evaluations", 1),
            "solve_ms": elapsed_ms,
            "config_hash": cfg.config_hash(),
            "git_commit": _git_commit(),
            "bank_checksum": bank.checksum(),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
