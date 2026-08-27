from __future__ import annotations

"""Fair six-method DRL training protocol.

TD3, DDPG and PPO share the same locked ScenarioBanks, environment-interaction
budget, QoS dual reward shaping and feasibility-first checkpoint selection.
The implementation intentionally reuses the frozen physical environment and
rate equations; only the learning algorithm changes.
"""

import json
import platform
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

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


def _reset_peak_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _timing_metadata(
    device: str, started: float, interactions: int
) -> dict[str, object]:
    elapsed = time.perf_counter() - started
    cuda = device == "cuda"
    return {
        "training_seconds": elapsed,
        "environment_interactions": interactions,
        "interactions_per_second": interactions / max(elapsed, 1e-12),
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(0)
            if cuda
            else platform.processor() or "CPU"
        ),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if cuda else 0.0
        ),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "platform": platform.platform(),
    }


def _is_better(candidate: dict[str, object], best: dict[str, object] | None) -> bool:
    if best is None:
        return True
    return tuple(candidate["selection_key"]) > tuple(best["selection_key"])


# Every axis a validation threshold can move along, and the direction that
# makes a checkpoint more selectable.
CANDIDATE_AXES = (
    ("mean_qos_fraction", 1.0),
    ("mean_all_qos", 1.0),
    ("mean_violation", -1.0),
    ("mean_sum_rate", 1.0),
)


def _dominates(a: dict[str, object], b: dict[str, object]) -> bool:
    """True when `a` is at least as selectable as `b` on every axis, and better on one."""
    better = False
    for key, sign in CANDIDATE_AXES:
        left, right = sign * float(a[key]), sign * float(b[key])
        if left < right:
            return False
        if left > right:
            better = True
    return better


def _pareto_front(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    """Entries some threshold setting could still select.

    Selection admits a checkpoint when qos_fraction and all_qos clear their
    targets and violation clears its tolerance, then takes the highest
    sum-rate among those. A checkpoint that another one matches or beats on
    all four axes can therefore never win, whatever the three thresholds are
    set to, so dropping it costs nothing. Everything else is kept, which is
    what makes a later threshold change a re-selection rather than a retrain.
    """
    return [
        item
        for item in entries
        if not any(_dominates(other, item) for other in entries if other is not item)
    ]


def _relaxation_cost(entry: dict[str, object], cfg: ExperimentConfig) -> float:
    """How far the thresholds must move before this entry becomes selectable.

    Zero for anything already feasible. Same normalisation the validation
    summary uses, so the ordering here agrees with the selection rule.
    """
    qos_gap = max(
        cfg.validation_qos_fraction_target - float(entry["mean_qos_fraction"]), 0.0
    ) / max(1.0 - cfg.validation_qos_fraction_target, 1e-12)
    all_qos_gap = max(
        cfg.validation_all_qos_target - float(entry["mean_all_qos"]), 0.0
    ) / max(1.0 - cfg.validation_all_qos_target, 1e-12)
    violation_gap = max(
        float(entry["mean_violation"]) - cfg.validation_violation_tolerance, 0.0
    ) / max(cfg.validation_violation_tolerance, 1e-12)
    return qos_gap + all_qos_gap + violation_gap


def _retain_candidate(
    agent: Any,
    method: str,
    cfg: ExperimentConfig,
    summary: dict[str, object],
    step: int,
    output: Path,
    candidates: list[dict[str, object]],
) -> None:
    """Keep the checkpoints a later re-selection could need.

    Only best.pt used to survive a run, so revisiting any validation threshold
    meant retraining all 25 jobs for that method.
    """
    limit = int(cfg.retained_candidate_checkpoints)
    if limit <= 0:
        return
    entry = {
        "eval_step": int(step),
        "checkpoint": f"candidate_step{int(step)}.pt",
        "mean_sum_rate": float(summary["mean_sum_rate"]),
        "mean_violation": float(summary["mean_violation"]),
        "mean_qos_fraction": float(summary["mean_qos_fraction"]),
        "mean_all_qos": float(summary["mean_all_qos"]),
        "mean_reward": float(summary["mean_reward"]),
    }
    front = _pareto_front([*candidates, entry])
    if len(front) > limit:
        # Over budget. Order by how much the thresholds would have to move to
        # reach each entry, so the currently selected checkpoint and its
        # nearest alternatives survive. Ranking by raw sum-rate instead would
        # spend the budget on diverged policies, which score highest precisely
        # because they stopped respecting QoS.
        front = sorted(
            front,
            key=lambda item: (_relaxation_cost(item, cfg), -float(item["mean_sum_rate"])),
        )[:limit]
    kept = {int(item["eval_step"]) for item in front}
    for stale in candidates:
        if int(stale["eval_step"]) not in kept:
            path = output / str(stale["checkpoint"])
            if path.exists():
                path.unlink()
    candidates[:] = sorted(front, key=lambda item: int(item["eval_step"]))
    if step in kept:
        save_checkpoint(
            output / str(entry["checkpoint"]),
            method,
            agent,
            step,
            float(summary["mean_reward"]),
            cfg,
            state_scope="policy",
        )


def _write_candidate_index(output: Path, candidates: list[dict[str, object]]) -> None:
    (output / "candidate_checkpoints.json").write_text(
        json.dumps(
            {
                "purpose": (
                    "re-select the reported checkpoint under a different "
                    "validation_violation_tolerance without retraining"
                ),
                "retained_set": (
                    "pareto front over high mean_qos_fraction, high mean_all_qos, "
                    "low mean_violation, high mean_sum_rate"
                ),
                "covers": [
                    "validation_qos_fraction_target",
                    "validation_all_qos_target",
                    "validation_violation_tolerance",
                ],
                "state_scope": "policy",
                "count": len(candidates),
                "candidates": candidates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _validation_step(
    agent: Any,
    method: str,
    cfg: ExperimentConfig,
    bank,
    seed: int,
    step: int,
    output: Path,
    best: dict[str, object] | None,
    candidates: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    validation_started = time.perf_counter()
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
    summary["validation_seconds"] = time.perf_counter() - validation_started
    summary_row = {k: v for k, v in summary.items() if k != "selection_key"}
    summary_row["selection_key"] = json.dumps(summary["selection_key"])
    summary_file = output / "validation_summary.csv"
    pd.DataFrame([summary_row]).to_csv(
        summary_file,
        mode="a",
        header=not summary_file.exists(),
        index=False,
    )

    if candidates is not None:
        _retain_candidate(agent, method, cfg, summary, step, output, candidates)

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
    candidates: list[dict[str, object]] = []
    if cfg.validate_at_initialization:
        best = _validation_step(
            agent, method, cfg, validation_bank, seed, 0, output, best, candidates
        )
        save_checkpoint(
            output / "initial.pt",
            method,
            agent,
            0,
            float(best["mean_reward"]),
            cfg,
        )
    _reset_peak_memory(device)
    training_started = time.perf_counter()
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
            elapsed_seconds = time.perf_counter() - training_started
            rows.append(
                {
                    "step": step,
                    "elapsed_seconds": elapsed_seconds,
                    "interactions_per_second": step / max(elapsed_seconds, 1e-12),
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
                candidates,
            )

    timing = _timing_metadata(device, training_started, cfg.train_steps)

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
    _write_candidate_index(output, candidates)
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
            "timing": timing,
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
    candidates: list[dict[str, object]] = []

    if cfg.validate_at_initialization:
        best = _validation_step(
            agent, "ppo", cfg, validation_bank, seed, 0, output, best, candidates
        )
        save_checkpoint(
            output / "initial.pt",
            "ppo",
            agent,
            0,
            float(best["mean_reward"]),
            cfg,
        )

    _reset_peak_memory(device)
    training_started = time.perf_counter()
    while global_step < cfg.train_steps:
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        pre_actions: list[np.ndarray] = []
        log_probs: list[float] = []
        rewards: list[float] = []
        values: list[float] = []
        dones: list[bool] = []
        infos: list[dict[str, object]] = []
        rollout = min(cfg.ppo_horizon, cfg.train_steps - global_step)

        for _ in range(rollout):
            action, pre_action, log_prob, value = agent.act(
                obs, deterministic=False, return_pre=True
            )
            next_obs, environment_reward, done, info = env.step(action)
            shaped_reward = dual.shaped_reward(info, cfg.qos_penalty_quadratic)
            dual.observe(float(info["violation"]))
            global_step += 1
            dual.maybe_update(global_step, cfg.warmup_steps)

            observations.append(obs)
            actions.append(action)
            pre_actions.append(pre_action)
            log_probs.append(log_prob)
            rewards.append(shaped_reward)
            values.append(value)
            dones.append(done)
            infos.append({**info, "environment_reward": environment_reward})
            obs = env.reset() if done else next_obs

        returns = np.zeros(len(rewards), dtype=np.float32)
        advantages = np.zeros(len(rewards), dtype=np.float32)
        _, _, last_value = agent.act(obs, deterministic=True)  # noqa: E501 - 3-tuple form
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
            pre_actions=np.asarray(pre_actions),
        )
        elapsed_seconds = time.perf_counter() - training_started
        logs.append(
            {
                "step": global_step,
                "elapsed_seconds": elapsed_seconds,
                "interactions_per_second": global_step / max(elapsed_seconds, 1e-12),
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
                candidates,
            )

    timing = _timing_metadata(device, training_started, cfg.train_steps)

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
    _write_candidate_index(output, candidates)
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
            "timing": timing,
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
