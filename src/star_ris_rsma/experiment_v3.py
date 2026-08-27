from __future__ import annotations

"""Fair six-method DRL training protocol.

TD3, DDPG and PPO share the same locked ScenarioBanks, environment-interaction
budget, QoS dual reward shaping and feasibility-first checkpoint selection.
The implementation intentionally reuses the frozen physical environment and
rate equations; only the learning algorithm changes.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .checkpoints import build_agent, save_checkpoint
from .config import ExperimentConfig
from .env import StarRisRsmaEnv
from .experiment import (
    _attach_training_bank,
    _bank_from_path_or_generate,
    _device,
    _manifest,
    _seed_everything,
    evaluate_policy_on_bank,
)
from .experiment_v2 import (
    QosDualController,
    constrained_validation_summary,
    exploration_noise_at_step,
)
from .replay import ReplayBuffer


SUPPORTED_METHODS = {"td3", "ddpg", "ppo"}


def _is_better(candidate: dict[str, object], best: dict[str, object] | None) -> bool:
    if best is None:
        return True
    return tuple(candidate["selection_key"]) > tuple(best["selection_key"])


def _validation_step(
    agent: Any,
    method: str,
    cfg: ExperimentConfig,
    bank,
    seed: int,
    step: int,
    output: Path,
    best: dict[str, object] | None,
) -> dict[str, object]:
    raw = evaluate_policy_on_bank(
        agent,
        method,
        cfg,
        bank,
        seed,
        max_scenarios=cfg.validation_scenarios,
    )
    raw.insert(3, "eval_step", step)
    validation_file = output / "validation_raw.csv"
    raw.to_csv(
        validation_file,
        mode="a",
        header=not validation_file.exists(),
        index=False,
    )

    summary = constrained_validation_summary(raw, cfg, step)
    summary_row = {k: v for k, v in summary.items() if k != "selection_key"}
    summary_row["selection_key"] = json.dumps(summary["selection_key"])
    summary_file = output / "validation_summary.csv"
    pd.DataFrame([summary_row]).to_csv(
        summary_file,
        mode="a",
        header=not summary_file.exists(),
        index=False,
    )

    if _is_better(summary, best):
        save_checkpoint(
            output / "best.pt",
            method,
            agent,
            step,
            float(summary["mean_reward"]),
            cfg,
        )
        (output / "best_validation.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary
    assert best is not None
    return best


def _common_banks(cfg: ExperimentConfig):
    train_bank = _bank_from_path_or_generate(
        cfg.train_bank_path,
        cfg,
        max(cfg.eval_scenarios, 256),
        11001,
        "train",
    )
    validation_bank = _bank_from_path_or_generate(
        cfg.validation_bank_path,
        cfg,
        cfg.validation_scenarios,
        22001,
        "validation",
    )
    return train_bank, validation_bank


def train_off_policy_v3(
    method: str,
    cfg: ExperimentConfig,
    seed: int,
    output: Path,
) -> None:
    if method not in {"td3", "ddpg"}:
        raise ValueError(method)

    _seed_everything(seed)
    device = _device()
    train_bank, validation_bank = _common_banks(cfg)
    env = StarRisRsmaEnv(cfg, seed)
    _attach_training_bank(env, train_bank, seed + 30001)
    agent = build_agent(method, env.observation_dim, env.action_dim, cfg, device)
    replay = ReplayBuffer(env.observation_dim, env.action_dim, cfg.replay_size, seed)
    dual = QosDualController.from_config(cfg)
    obs = env.reset()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    if cfg.validate_at_initialization:
        best = _validation_step(
            agent, method, cfg, validation_bank, seed, 0, output, best
        )
        save_checkpoint(
            output / "initial.pt",
            method,
            agent,
            0,
            float(best["mean_reward"]),
            cfg,
        )
    for step in range(1, cfg.train_steps + 1):
        if step <= cfg.warmup_steps:
            action = np.random.uniform(-1.0, 1.0, env.action_dim).astype(np.float32)
            exploration_noise = 0.0
        else:
            exploration_noise = exploration_noise_at_step(cfg, step)
            action = agent.act(obs, noise_std=exploration_noise)

        next_obs, environment_reward, done, info = env.step(action)
        training_reward = dual.shaped_reward(info, cfg.qos_penalty_quadratic)
        replay.add(obs, action, training_reward, next_obs, done)
        dual.observe(float(info["violation"]))
        dual_updated = dual.maybe_update(step, cfg.warmup_steps)
        obs = env.reset() if done else next_obs
        losses = (
            agent.update(replay.sample(cfg.batch_size))
            if replay.size >= cfg.batch_size
            else {}
        )

        if step == 1 or step % 1000 == 0:
            rows.append(
                {
                    "step": step,
                    "reward": training_reward,
                    "environment_reward": environment_reward,
                    "sum_rate": info["sum_rate"],
                    "qos_fraction": info["qos_fraction"],
                    "all_qos": info["all_qos"],
                    "violation": info["violation"],
                    "qos_dual": dual.value,
                    "qos_violation_ema": dual.violation_ema,
                    "qos_dual_updated": dual_updated,
                    "exploration_noise": exploration_noise,
                    **losses,
                }
            )

        if step % cfg.validation_interval == 0 or step == cfg.train_steps:
            best = _validation_step(
                agent,
                method,
                cfg,
                validation_bank,
                seed,
                step,
                output,
                best,
            )

    if best is None:
        raise RuntimeError("No validation checkpoint was produced")

    save_checkpoint(
        output / "latest.pt",
        method,
        agent,
        cfg.train_steps,
        float(best["mean_reward"]),
        cfg,
    )
    pd.DataFrame(rows).to_csv(output / "training.csv", index=False)
    manifest = _manifest(
        method,
        seed,
        cfg,
        device,
        {"train": train_bank, "validation": validation_bank},
        {
            "training_protocol": "drl_v3_qos_constrained_fair",
            "best_validation": best,
            "checkpoint_selection": "feasibility_first_normalized_gap_then_sum_rate",
            "qos_dual": dual.state_dict(),
            "environment_interactions": cfg.train_steps,
            "exploration_schedule": {
                "start": cfg.exploration_noise,
                "final": cfg.exploration_noise_final,
                "decay_steps": cfg.exploration_decay_steps,
            },
        },
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def train_ppo_v3(cfg: ExperimentConfig, seed: int, output: Path) -> None:
    _seed_everything(seed)
    device = _device()
    train_bank, validation_bank = _common_banks(cfg)
    env = StarRisRsmaEnv(cfg, seed)
    _attach_training_bank(env, train_bank, seed + 30001)
    agent = build_agent("ppo", env.observation_dim, env.action_dim, cfg, device)
    dual = QosDualController.from_config(cfg)
    obs = env.reset()
    global_step = 0
    output.mkdir(parents=True, exist_ok=True)
    logs: list[dict[str, object]] = []
    best: dict[str, object] | None = None

    if cfg.validate_at_initialization:
        best = _validation_step(
            agent, "ppo", cfg, validation_bank, seed, 0, output, best
        )
        save_checkpoint(
            output / "initial.pt",
            "ppo",
            agent,
            0,
            float(best["mean_reward"]),
            cfg,
        )

    while global_step < cfg.train_steps:
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        log_probs: list[float] = []
        rewards: list[float] = []
        values: list[float] = []
        dones: list[bool] = []
        infos: list[dict[str, object]] = []
        rollout = min(cfg.ppo_horizon, cfg.train_steps - global_step)

        for _ in range(rollout):
            action, log_prob, value = agent.act(obs, deterministic=False)
            next_obs, environment_reward, done, info = env.step(action)
            shaped_reward = dual.shaped_reward(info, cfg.qos_penalty_quadratic)
            dual.observe(float(info["violation"]))
            global_step += 1
            dual.maybe_update(global_step, cfg.warmup_steps)

            observations.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(shaped_reward)
            values.append(value)
            dones.append(done)
            infos.append({**info, "environment_reward": environment_reward})
            obs = env.reset() if done else next_obs

        returns = np.zeros(len(rewards), dtype=np.float32)
        advantages = np.zeros(len(rewards), dtype=np.float32)
        _, _, last_value = agent.act(obs, deterministic=True)
        gae = 0.0
        for index in reversed(range(len(rewards))):
            nonterminal = 1.0 - float(dones[index])
            next_value = last_value if index == len(rewards) - 1 else values[index + 1]
            delta = (
                rewards[index]
                + cfg.gamma * next_value * nonterminal
                - values[index]
            )
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
            advantages[index] = gae
            returns[index] = gae + values[index]

        losses = agent.update(
            np.asarray(observations),
            np.asarray(actions),
            np.asarray(log_probs),
            returns,
            advantages,
        )
        logs.append(
            {
                "step": global_step,
                "reward": float(np.mean(rewards)),
                "environment_reward": float(
                    np.mean([float(item["environment_reward"]) for item in infos])
                ),
                "sum_rate": float(np.mean([float(item["sum_rate"]) for item in infos])),
                "qos_fraction": float(
                    np.mean([float(item["qos_fraction"]) for item in infos])
                ),
                "all_qos": float(np.mean([float(item["all_qos"]) for item in infos])),
                "violation": float(
                    np.mean([float(item["violation"]) for item in infos])
                ),
                "qos_dual": dual.value,
                "qos_violation_ema": dual.violation_ema,
                **losses,
            }
        )

        if global_step % cfg.validation_interval < rollout or global_step == cfg.train_steps:
            best = _validation_step(
                agent,
                "ppo",
                cfg,
                validation_bank,
                seed,
                global_step,
                output,
                best,
            )

    if best is None:
        raise RuntimeError("No validation checkpoint was produced")

    save_checkpoint(
        output / "latest.pt",
        "ppo",
        agent,
        cfg.train_steps,
        float(best["mean_reward"]),
        cfg,
    )
    pd.DataFrame(logs).to_csv(output / "training.csv", index=False)
    manifest = _manifest(
        "ppo",
        seed,
        cfg,
        device,
        {"train": train_bank, "validation": validation_bank},
        {
            "training_protocol": "drl_v3_qos_constrained_fair",
            "best_validation": best,
            "checkpoint_selection": "feasibility_first_normalized_gap_then_sum_rate",
            "qos_dual": dual.state_dict(),
            "environment_interactions": cfg.train_steps,
            "ppo_horizon": cfg.ppo_horizon,
        },
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def train_drl_v3(
    method: str,
    cfg: ExperimentConfig,
    seed: int,
    output: Path,
) -> None:
    normalized = method.lower()
    if normalized not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}")
    if normalized == "ppo":
        train_ppo_v3(cfg, seed, output)
    else:
        train_off_policy_v3(normalized, cfg, seed, output)
