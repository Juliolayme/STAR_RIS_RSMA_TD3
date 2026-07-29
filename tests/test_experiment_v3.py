from __future__ import annotations

import json

import pandas as pd
import pytest

from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.experiment_v3 import train_drl_v3


@pytest.mark.parametrize("method", ["td3", "ddpg", "ppo"])
def test_drl_v3_smoke_produces_constrained_checkpoint(method, tmp_path):
    cfg = ExperimentConfig(
        n_ris=4,
        n_users=2,
        hidden_dim=32,
        batch_size=16,
        replay_size=256,
        warmup_steps=8,
        train_steps=64,
        eval_scenarios=16,
        validation_interval=32,
        validation_scenarios=16,
        episode_length=4,
        exploration_noise=0.10,
        exploration_noise_final=0.02,
        exploration_decay_steps=64,
        observation_normalization="blockwise_v2",
        action_parameterization="physical_v3",
        qos_penalty_linear=4.0,
        qos_penalty_quadratic=4.0,
        qos_dual_enabled=True,
        qos_dual_initial=4.0,
        qos_dual_learning_rate=1.0,
        qos_dual_target_violation=0.01,
        qos_dual_update_interval=8,
        qos_dual_min=1.0,
        qos_dual_max=16.0,
        validation_qos_fraction_target=0.0,
        validation_all_qos_target=0.0,
        validation_violation_tolerance=10.0,
        ppo_horizon=32,
        td3_layer_norm=True,
        td3_critic_loss="huber",
        td3_gradient_clip_norm=10.0,
    )
    output = tmp_path / method
    train_drl_v3(method, cfg, seed=0, output=output)

    for name in (
        "best.pt",
        "latest.pt",
        "best_validation.json",
        "training.csv",
        "validation_raw.csv",
        "validation_summary.csv",
        "manifest.json",
    ):
        assert (output / name).exists(), name

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["method"] == method
    assert manifest["training_protocol"] == "drl_v3_qos_constrained_fair"
    assert manifest["checkpoint_selection"] == "feasibility_first_normalized_gap_then_sum_rate"
    assert manifest["environment_interactions"] == 64

    training = pd.read_csv(output / "training.csv")
    validation = pd.read_csv(output / "validation_summary.csv")
    assert not training.empty
    assert not validation.empty
    assert {
        "mean_sum_rate",
        "mean_qos_fraction",
        "mean_all_qos",
        "mean_violation",
        "feasible",
    }.issubset(validation.columns)
