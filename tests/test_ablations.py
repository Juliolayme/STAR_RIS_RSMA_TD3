import numpy as np

from star_ris_rsma.action import decode_action
from star_ris_rsma.baselines.ablations import ABLATION_MODES, evaluate_ablation
from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.env import StarRisRsmaEnv


def test_all_ablation_modes_are_finite_and_random_is_reproducible():
    env = StarRisRsmaEnv(ExperimentConfig(n_users=2, n_ris=4), 3)
    env.reset()
    raw = np.zeros(env.action_dim)
    for mode in ABLATION_MODES:
        metrics = evaluate_ablation(env, raw, mode, seed=9)
        assert np.isfinite(metrics["sum_rate"])
    a = evaluate_ablation(env, raw, "random_ris", seed=7)
    b = evaluate_ablation(env, raw, "random_ris", seed=7)
    assert np.isclose(a["sum_rate"], b["sum_rate"])


def test_no_ris_removes_all_ris_dependence():
    env = StarRisRsmaEnv(ExperimentConfig(n_users=2, n_ris=4), 3)
    env.reset()
    raw_a = np.zeros(env.action_dim)
    raw_b = raw_a.copy()
    split = (env.config.n_users + 1) + env.config.n_users
    raw_b[split:] = np.linspace(-4.0, 4.0, raw_b.size - split)
    a = evaluate_ablation(env, raw_a, "no_ris", seed=1)
    b = evaluate_ablation(env, raw_b, "no_ris", seed=1)
    assert np.isclose(a["sum_rate"], b["sum_rate"])


def test_ablation_uses_configured_physical_action_parameterization(monkeypatch):
    cfg = ExperimentConfig(n_users=2, n_ris=4, action_parameterization="physical_v3")
    env = StarRisRsmaEnv(cfg, 3)
    env.reset()
    raw = np.linspace(-1.0, 1.0, env.action_dim)
    expected = decode_action(raw, cfg.n_users, cfg.n_ris, cfg.p_max, "physical_v3")

    captured = {}
    original = env.metrics_from_effective_channel

    def record_action(channel, action):
        captured["action"] = action
        return original(channel, action)

    monkeypatch.setattr(env, "metrics_from_effective_channel", record_action)
    evaluate_ablation(env, raw, "learned")

    assert np.allclose(captured["action"].powers, expected.powers)
    assert np.allclose(captured["action"].beta_t, expected.beta_t)
