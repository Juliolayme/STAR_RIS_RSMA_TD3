import numpy as np

from star_ris_rsma.action import action_dim, decode_action
from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.env import StarRisRsmaEnv
from star_ris_rsma.physics import star_coefficients


def test_action_constraints():
    cfg = ExperimentConfig(n_ris=8, n_users=4)
    a = decode_action(np.random.default_rng(0).normal(size=action_dim(4, 8)), 4, 8, cfg.p_max)
    assert np.isclose(a.powers.sum(), cfg.p_max)
    assert np.isclose(a.common_fractions.sum(), 1.0)
    assert np.all((a.beta_t >= 0) & (a.beta_t <= 1))


def test_star_energy_split():
    beta = np.linspace(0, 1, 5)
    pt, pr = star_coefficients(beta, np.zeros(5), np.zeros(5))
    assert np.allclose(np.abs(pt) ** 2 + np.abs(pr) ** 2, 1.0)


def test_environment_finite_metrics():
    cfg = ExperimentConfig(n_ris=4, n_users=2, episode_length=2)
    env = StarRisRsmaEnv(cfg, 1); env.reset()
    metrics = env.evaluate_raw_action(np.zeros(env.action_dim))
    assert np.isfinite(metrics["sum_rate"])
    assert 0 <= metrics["qos_fraction"] <= 1
