import numpy as np

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
