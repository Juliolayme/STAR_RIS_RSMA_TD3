import numpy as np

from star_ris_rsma.baselines import analytical_ris, ao_grid, ao_sca
from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.env import StarRisRsmaEnv


def make_env():
    env = StarRisRsmaEnv(ExperimentConfig(n_ris=4, n_users=2), 4)
    env.reset(); return env


def test_all_solvers_return_feasible_action():
    for solver in [analytical_ris, ao_grid, ao_sca]:
        env = make_env(); x, m = solver(env)
        assert x.shape == (env.action_dim,)
        assert np.isfinite(m["sum_rate"])


def test_ao_sca_monotone_history():
    _, m = ao_sca(make_env(), max_iter=3)
    h = np.asarray(m["objective_history"])
    assert np.all(np.diff(h) >= -1e-8)
