from __future__ import annotations

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
from .replay import ReplayBuffer


def exploration_noise_at_step(cfg: ExperimentConfig, step: int) -> float:
    progress = min(max(step, 0) / cfg.exploration_decay_steps, 1.0)
    return float(
        cfg.exploration_noise
        + progress * (cfg.exploration_noise_final - cfg.exploration_noise)
    )


def constrained_validation_summary(
    raw: pd.DataFrame,
    cfg: ExperimentConfig,
    step: int,
) -> dict[str, object]:
    mean_reward = float(raw["reward"].mean())
    mean_sum_rate = float(raw["sum_rate"].mean())
    mean_qos_fraction = float(raw["qos_fraction"].mean())
    mean_all_qos = float(raw["all_qos"].astype(float).mean())
    mean_violation = float(raw["violation"].mean())
    feasible = bool(
        mean_qos_fraction >= cfg.validation_qos_fraction_target
        and mean_all_qos >= cfg.validation_all_qos_target
        and mean_violation <= cfg.validation_violation_tolerance
    )

    # Lexicographic QoS-first selection. Once feasible, maximize sum-rate.
    # Before feasibility, reduce violation first and then prefer broader QoS.
    selection_key = (
        int(feasible),
        mean_sum_rate if feasible else -mean_violation,
        mean_all_qos,
        mean_qos_fraction,
        -mean_violation,
        mean_reward,
    )
    return {
        "eval_step": int(step),
        "mean_reward": mean_reward,
        "mean_sum_rate": mean_sum_rate,
        "mean_qos_fraction": mean_qos_fraction,
        "mean_all_qos": mean_all_qos,
        "mean_violation": mean_violation,
        "feasible": feasible,
        "selection_key": list(selection_key),
    }


def _is_better(candidate: dict[str, object], best: dict[str, object] | None) -> bool:
    if best is None:
        return True
    return tuple(candidate["selection_key"]) > tuple(best["selection_key"])


def _validation_step_v2(
    agent: Any,
    cfg: ExperimentConfig,
    bank,
    seed: int,
    step: int,
    output: Path,
    best: dict[str, object] | None,
) -> dict[str, object]:
    raw = evaluate_policy_on_bank(
        agent,
        "td3",
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
            "td3",
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


def train_td3_v2(cfg: ExperimentConfig, seed: int, output: Path) -> None:
    """TD3 v2 pilot with N-stable inputs and QoS-first checkpoint selection."""
    _seed_everything(seed)
    device = _device()
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

    env = StarRisRsmaEnv(cfg, seed)
    _attach_training_bank(env, train_bank, seed + 30001)
    agent = build_agent("td3", env.observation_dim, env.action_dim, cfg, device)
    replay = ReplayBuffer(env.observation_dim, env.action_dim, cfg.replay_size, seed)
    obs = env.reset()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for step in range(1, cfg.train_steps + 1):
        if step <= cfg.warmup_steps:
            action = np.random.uniform(-1.0, 1.0, env.action_dim).astype(np.float32)
            exploration_noise = 0.0
        else:
            exploration_noise = exploration_noise_at_step(cfg, step)
            action = agent.act(obs, noise_std=exploration_noise)

        next_obs, reward, done, info = env.step(action)
        replay.add(obs, action, reward, next_obs, done)
        obs = env.reset() if done else next_obs
        losses = (
            agent.update(replay.sample(cfg.batch_size))
            if replay.size >= cfg.batch_size
            else {}
        )

        if step == 1 or step % 1000 == 0:
            rows.append({
                "step": step,
                "reward": reward,
                "sum_rate": info["sum_rate"],
                "qos_fraction": info["qos_fraction"],
                "all_qos": info["all_qos"],
                "violation": info["violation"],
                "exploration_noise": exploration_noise,
                **losses,
            })

        if step % cfg.validation_interval == 0 or step == cfg.train_steps:
            best = _validation_step_v2(
                agent,
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
        "td3",
        agent,
        cfg.train_steps,
        float(best["mean_reward"]),
        cfg,
    )
    pd.DataFrame(rows).to_csv(output / "training.csv", index=False)
    manifest = _manifest(
        "td3",
        seed,
        cfg,
        device,
        {"train": train_bank, "validation": validation_bank},
        {
            "training_protocol": "td3_qos_scalability_v2",
            "best_validation": best,
            "checkpoint_selection": "qos_first_lexicographic",
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
