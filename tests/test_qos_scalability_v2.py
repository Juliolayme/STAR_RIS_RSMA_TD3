import numpy as np
import pandas as pd

from star_ris_rsma.agents.td3 import TD3Agent
from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.env import StarRisRsmaEnv
from star_ris_rsma.experiment_v2 import constrained_validation_summary


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


def test_td3_noise_is_dimension_normalized():
    small = TD3Agent(8, 64, noise_reference_dim=64)
    large = TD3Agent(8, 393, noise_reference_dim=64)
    assert np.isclose(small._dimension_noise_scale(), 1.0)
    assert large._dimension_noise_scale() < 0.5
