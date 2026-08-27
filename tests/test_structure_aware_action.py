from __future__ import annotations

import numpy as np
import pytest

from star_ris_rsma.action import (
    action_dim,
    decode_action,
    reference_phase,
    weighted_reference_phase,
    wrap_phase,
)
from star_ris_rsma.baselines.ablations import evaluate_ablation
from star_ris_rsma.checkpoints import build_agent
from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.env import StarRisRsmaEnv


def make_env(parameterization: str) -> StarRisRsmaEnv:
    cfg = ExperimentConfig(
        n_ris=8,
        n_users=4,
        noise_power=1e-3,
        qos_min=0.5,
        observation_normalization="blockwise_v2",
        action_parameterization=parameterization,
        actor_small_final_init=True,
    )
    env = StarRisRsmaEnv(cfg, 7)
    env.reset()
    return env


@pytest.mark.parametrize("parameterization", ["physical_v5_hard", "physical_v5_soft"])
def test_zero_action_uses_analytical_phase_anchor(parameterization: str) -> None:
    env = make_env(parameterization)
    action = decode_action(
        np.zeros(env.action_dim), 4, 8, 1.0, parameterization, channel=env.channel
    )
    assert np.allclose(action.theta_t, reference_phase(env.channel))
    assert np.allclose(action.theta_r, reference_phase(env.channel))
    assert np.isclose(action.powers.sum(), 1.0)


def test_hard_selector_has_exactly_one_private_stream() -> None:
    env = make_env("physical_v5_hard")
    raw = np.linspace(-1.0, 1.0, action_dim(4, 8))
    action = decode_action(raw, 4, 8, 1.0, "physical_v5_hard", channel=env.channel)
    assert np.count_nonzero(action.powers[1:]) == 1


def test_soft_selector_is_continuous_and_can_be_nearly_one_hot() -> None:
    env = make_env("physical_v5_soft")
    raw = np.zeros(env.action_dim)
    raw[1:5] = [-1.0, -1.0, -1.0, 1.0]
    first = decode_action(raw, 4, 8, 1.0, "physical_v5_soft", channel=env.channel)
    perturbed = raw.copy()
    perturbed[4] -= 1e-5
    second = decode_action(perturbed, 4, 8, 1.0, "physical_v5_soft", channel=env.channel)
    private_share = first.powers[1:] / first.powers[1:].sum()
    assert private_share.max() > 0.999999
    assert np.linalg.norm(first.powers - second.powers) < 1e-4


def test_structure_aware_ablation_receives_channel() -> None:
    env = make_env("physical_v5_hard")
    metrics = evaluate_ablation(env, np.zeros(env.action_dim), "learned")
    assert np.isfinite(metrics["sum_rate"])


def test_small_final_init_starts_near_anchor() -> None:
    env = make_env("physical_v5_hard")
    agent = build_agent("td3", env.observation_dim, env.action_dim, env.config, "cpu")
    raw = agent.act(env.reset(), noise_std=0.0)
    assert np.abs(raw).mean() < 0.05


def test_historical_v2_hash_is_unchanged() -> None:
    cfg = ExperimentConfig.from_yaml("configs/v3/constrained_action_n32.yaml")
    assert cfg.legacy_config_hash_v2() == (
        "3f532d9493a43a1111f0f7d170f507e55bb594a932a17a244a922c8e2910d3cd"
    )


def test_phase_residual_is_bounded() -> None:
    env = make_env("physical_v5_soft")
    raw = np.ones(env.action_dim)
    action = decode_action(raw, 4, 8, 1.0, "physical_v5_soft", channel=env.channel)
    offset = np.abs(wrap_phase(action.theta_t - reference_phase(env.channel)))
    assert offset.max() <= 0.25 * np.pi + 1e-12


def test_v6_anchor_follows_soft_private_user_selection() -> None:
    env = make_env("physical_v6_soft_anchor")
    raw = np.zeros(env.action_dim)
    raw[1:5] = [-1.0, -1.0, 1.0, -1.0]
    action = decode_action(
        raw, 4, 8, 1.0, "physical_v6_soft_anchor", channel=env.channel
    )
    private_weights = action.powers[1:] / action.powers[1:].sum()
    expected = weighted_reference_phase(env.channel, private_weights)
    assert np.allclose(action.theta_t, expected)
    assert np.allclose(action.theta_r, expected)


def test_v6_anchor_and_decoded_action_are_continuous() -> None:
    env = make_env("physical_v6_soft_anchor")
    raw = np.linspace(-0.8, 0.8, env.action_dim)
    first = decode_action(
        raw, 4, 8, 1.0, "physical_v6_soft_anchor", channel=env.channel
    )
    perturbed = raw.copy()
    perturbed[2] += 1e-6
    second = decode_action(
        perturbed, 4, 8, 1.0, "physical_v6_soft_anchor", channel=env.channel
    )
    assert np.linalg.norm(first.powers - second.powers) < 1e-4
    assert np.linalg.norm(wrap_phase(first.theta_t - second.theta_t)) < 1e-4
