from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class QosDualController:
    """Projected dual-ascent controller for the mean QoS violation constraint.

    The TD3 critic receives a shaped reward using the current dual multiplier.
    The multiplier increases only when the exponentially smoothed violation is
    above the predeclared target and is projected onto a bounded interval. This
    avoids choosing one excessively large fixed penalty for every RIS size.
    """

    enabled: bool
    value: float
    learning_rate: float
    target_violation: float
    update_interval: int
    ema_beta: float
    minimum: float
    maximum: float
    violation_ema: float | None = None
    updates: int = 0

    @classmethod
    def from_config(cls, cfg: ExperimentConfig) -> "QosDualController":
        return cls(
            enabled=bool(cfg.qos_dual_enabled),
            value=float(cfg.qos_dual_initial if cfg.qos_dual_enabled else cfg.qos_penalty_linear),
            learning_rate=float(cfg.qos_dual_learning_rate),
            target_violation=float(cfg.qos_dual_target_violation),
            update_interval=int(cfg.qos_dual_update_interval),
            ema_beta=float(cfg.qos_dual_ema_beta),
            minimum=float(cfg.qos_dual_min),
            maximum=float(cfg.qos_dual_max),
        )

    def shaped_reward(self, info: dict[str, object], quadratic_penalty: float) -> float:
        sum_rate = float(info["sum_rate"])
        violation = float(info["violation"])
        violation_squared = float(info.get("violation_squared", violation * violation))
        return float(
            sum_rate
            - self.value * violation
            - float(quadratic_penalty) * violation_squared
        )

    def observe(self, violation: float) -> None:
        value = float(violation)
        if self.violation_ema is None:
            self.violation_ema = value
        else:
            self.violation_ema = (
                self.ema_beta * self.violation_ema
                + (1.0 - self.ema_beta) * value
            )

    def maybe_update(self, step: int, warmup_steps: int) -> bool:
        if (
            not self.enabled
            or self.violation_ema is None
            or step <= warmup_steps
            or step % self.update_interval != 0
        ):
            return False
        proposed = self.value + self.learning_rate * (
            self.violation_ema - self.target_violation
        )
        self.value = float(np.clip(proposed, self.minimum, self.maximum))
        self.updates += 1
        return True

    def state_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "value": self.value,
            "learning_rate": self.learning_rate,
            "target_violation": self.target_violation,
            "update_interval": self.update_interval,
            "ema_beta": self.ema_beta,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "violation_ema": self.violation_ema,
            "updates": self.updates,
        }


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

    qos_gap = max(cfg.validation_qos_fraction_target - mean_qos_fraction, 0.0)
    all_qos_gap = max(cfg.validation_all_qos_target - mean_all_qos, 0.0)
    violation_gap = max(
        mean_violation - cfg.validation_violation_tolerance,
        0.0,
    ) / max(cfg.validation_violation_tolerance, 1e-12)
    constraint_gap = float(qos_gap + all_qos_gap + violation_gap)
    feasible = bool(constraint_gap <= 1e-12)

    # A checkpoint must first satisfy all predeclared QoS constraints. Among
    # feasible checkpoints maximize sum-rate; before feasibility minimize the
    # normalized aggregate constraint gap and then the raw violation.
    selection_key = (
        int(feasible),
        mean_sum_rate if feasible else -constraint_gap,
        -mean_violation,
        mean_all_qos,
        mean_qos_fraction,
        mean_reward,
    )
    return {
        "eval_step": int(step),
        "mean_reward": mean_reward,
        "mean_sum_rate": mean_sum_rate,
        "mean_qos_fraction": mean_qos_fraction,
        "mean_all_qos": mean_all_qos,
        "mean_violation": mean_violation,
        "qos_fraction_gap": qos_gap,
        "all_qos_gap": all_qos_gap,
        "normalized_violation_gap": violation_gap,
        "constraint_gap": constraint_gap,
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
    """TD3 with N-stable inputs and constrained QoS checkpoint selection."""
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
    dual = QosDualController.from_config(cfg)
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
            rows.append({
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
            "training_protocol": "td3_qos_scalability_v3_constrained",
            "best_validation": best,
            "checkpoint_selection": "feasibility_first_normalized_gap_then_sum_rate",
            "qos_dual": dual.state_dict(),
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
