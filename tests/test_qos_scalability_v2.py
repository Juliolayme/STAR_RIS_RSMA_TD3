import numpy as np
import pandas as pd

from star_ris_rsma.agents.td3 import TD3Agent
from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.env import StarRisRsmaEnv
from star_ris_rsma.experiment_v2 import (
    QosDualController,
    constrained_validation_summary,
)


def test_blockwise_observation_scale_does_not_vanish_with_n():
    cfg16 = ExperimentConfig(n_ris=16, observation_normalization="blockwise_v2")
    cfg128 = ExperimentConfig(n_ris=128, observation_normalization="blockwise_v2")
    obs16 = StarRisRsmaEnv(cfg16, 7).reset()
    obs128 = StarRisRsmaEnv(cfg128, 7).reset()
    rms16 = float(np.sqrt(np.mean(np.square(obs16))))
    rms128 = float(np.sqrt(np.mean(np.square(obs128))))
    assert 0.4 <= rms16 <= 2.0
    assert 0.4 <= rms128 <= 2.0
    assert 0.5 <= rms128 / rms16 <= 2.0


def test_configurable_qos_penalty_is_stronger_for_same_action():
    base = ExperimentConfig(n_users=2, n_ris=4, qos_min=10.0)
    strong = ExperimentConfig(
        n_users=2,
        n_ris=4,
        qos_min=10.0,
        qos_penalty_linear=8.0,
        qos_penalty_quadratic=4.0,
    )
    env_base = StarRisRsmaEnv(base, 3)
    env_strong = StarRisRsmaEnv(strong, 3)
    channel = env_base._sample_channel()
    env_base.reset(channel=channel)
    env_strong.reset(channel=channel)
    raw = np.zeros(env_base.action_dim)
    weak_metrics = env_base.evaluate_raw_action(raw)
    strong_metrics = env_strong.evaluate_raw_action(raw)
    assert weak_metrics["violation"] > 0
    assert strong_metrics["reward"] < weak_metrics["reward"]


def test_qos_first_checkpoint_selection_beats_higher_reward_infeasible_policy():
    cfg = ExperimentConfig(
        validation_qos_fraction_target=0.95,
        validation_all_qos_target=0.80,
        validation_violation_tolerance=0.01,
    )
    feasible = pd.DataFrame({
        "reward": [2.0, 2.1],
        "sum_rate": [2.0, 2.1],
        "qos_fraction": [1.0, 1.0],
        "all_qos": [True, True],
        "violation": [0.0, 0.0],
    })
    infeasible = pd.DataFrame({
        "reward": [4.0, 4.1],
        "sum_rate": [4.5, 4.6],
        "qos_fraction": [0.75, 0.75],
        "all_qos": [False, False],
        "violation": [0.5, 0.5],
    })
    good = constrained_validation_summary(feasible, cfg, 10)
    bad = constrained_validation_summary(infeasible, cfg, 20)
    assert good["feasible"] is True
    assert bad["feasible"] is False
    assert tuple(good["selection_key"]) > tuple(bad["selection_key"])


def test_infeasible_checkpoint_uses_normalized_total_constraint_gap():
    cfg = ExperimentConfig(
        validation_qos_fraction_target=0.99,
        validation_all_qos_target=0.95,
        validation_violation_tolerance=0.001,
    )
    small_gap = pd.DataFrame({
        "reward": [8.0, 8.0],
        "sum_rate": [10.0, 10.0],
        "qos_fraction": [0.99, 0.99],
        "all_qos": [0.94, 0.94],
        "violation": [0.0005, 0.0005],
    })
    large_gap = pd.DataFrame({
        "reward": [12.0, 12.0],
        "sum_rate": [14.0, 14.0],
        "qos_fraction": [0.99, 0.99],
        "all_qos": [0.96, 0.96],
        "violation": [0.002, 0.002],
    })
    preferred = constrained_validation_summary(small_gap, cfg, 10)
    rejected = constrained_validation_summary(large_gap, cfg, 20)
    assert preferred["feasible"] is False
    assert rejected["feasible"] is False
    assert preferred["constraint_gap"] < rejected["constraint_gap"]
    assert tuple(preferred["selection_key"]) > tuple(rejected["selection_key"])


def test_adaptive_qos_dual_increases_and_projects_at_maximum():
    cfg = ExperimentConfig(
        qos_dual_enabled=True,
        qos_dual_initial=8.0,
        qos_dual_learning_rate=20.0,
        qos_dual_target_violation=0.001,
        qos_dual_update_interval=10,
        qos_dual_ema_beta=0.0,
        qos_dual_min=4.0,
        qos_dual_max=9.0,
    )
    dual = QosDualController.from_config(cfg)
    dual.observe(0.101)
    assert dual.maybe_update(step=10, warmup_steps=0) is True
    assert dual.value == 9.0
    assert dual.updates == 1


def test_adaptive_qos_dual_is_disabled_for_legacy_configs():
    cfg = ExperimentConfig(qos_penalty_linear=2.0)
    dual = QosDualController.from_config(cfg)
    dual.observe(1.0)
    assert dual.maybe_update(step=1000, warmup_steps=0) is False
    assert dual.value == 2.0


def test_new_dual_fields_do_not_change_legacy_v2_hash():
    base = ExperimentConfig(action_parameterization="physical_v3")
    constrained = ExperimentConfig(
        action_parameterization="physical_v3",
        qos_dual_enabled=True,
        qos_dual_initial=16.0,
        qos_dual_learning_rate=40.0,
        qos_dual_target_violation=0.0005,
    )
    assert base.legacy_config_hash_v2() == constrained.legacy_config_hash_v2()


def test_td3_noise_is_dimension_normalized():
    small = TD3Agent(8, 64, noise_reference_dim=64)
    large = TD3Agent(8, 393, noise_reference_dim=64)
    assert np.isclose(small._dimension_noise_scale(), 1.0)
    assert large._dimension_noise_scale() < 0.5
