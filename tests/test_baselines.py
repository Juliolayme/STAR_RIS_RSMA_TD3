import numpy as np
import pandas as pd

from star_ris_rsma.action import decode_action
from star_ris_rsma.baselines import analytical_ris, ao_grid, ao_sca
from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.env import StarRisRsmaEnv
from star_ris_rsma.experiment import evaluate_solver
from star_ris_rsma.scenario_bank import generate_bank


def make_env():
    env = StarRisRsmaEnv(ExperimentConfig(n_ris=3, n_users=2), 4)
    env.reset()
    return env


def test_all_solvers_return_feasible_action():
    for solver in [analytical_ris, ao_grid, ao_sca]:
        env = make_env()
        kwargs = {"rounds": 1, "allocation_levels": 3, "beta_levels": 3, "phase_levels": 4} if solver is ao_grid else {}
        if solver is ao_sca:
            kwargs = {"max_iter": 2}
        x, metrics = solver(env, **kwargs)
        action = decode_action(x, env.config.n_users, env.config.n_ris, env.config.p_max)
        assert x.shape == (env.action_dim,)
        assert np.isfinite(metrics["sum_rate"])
        assert np.isclose(action.powers.sum(), env.config.p_max)
        assert np.isclose(action.common_fractions.sum(), 1.0)
        assert np.all((action.beta_t >= 0) & (action.beta_t <= 1))


def test_ao_sca_monotone_history_and_surrogate_metadata():
    _, metrics = ao_sca(make_env(), max_iter=3)
    history = np.asarray(metrics["objective_history"])
    assert np.all(np.diff(history) >= -1e-8)
    assert metrics["solver"] == "ao_sca_proximal_physical"
    assert metrics["evaluations"] >= 1
    assert len(metrics["surrogate_records"]) >= 1


def test_ao_grid_is_deterministic_codebook_search():
    env1 = make_env(); x1, m1 = ao_grid(env1, rounds=1, allocation_levels=3, beta_levels=3, phase_levels=4, seed=1)
    env2 = make_env(); x2, m2 = ao_grid(env2, rounds=1, allocation_levels=3, beta_levels=3, phase_levels=4, seed=999)
    assert np.allclose(x1, x2)
    assert np.isclose(m1["reward"], m2["reward"])
    assert m1["solver"] == "ao_grid_coordinate_codebook"
    assert m1["evaluations"] > 1


def test_solver_csv_preserves_method_metadata(tmp_path):
    cfg = ExperimentConfig(n_ris=3, n_users=2)
    bank = generate_bank(cfg, 1, seed=123, split="test")

    ao_sca_path = tmp_path / "ao_sca.csv"
    evaluate_solver("ao_sca", cfg, seed=0, start=0, count=1, output=ao_sca_path, bank=bank)
    ao_sca_row = pd.read_csv(ao_sca_path).iloc[0]
    assert ao_sca_row["solver"] == "ao_sca_proximal_physical"
    assert ao_sca_row["objective_history"].startswith("[")
    assert ao_sca_row["surrogate_records"].startswith("[")

    ao_grid_path = tmp_path / "ao_grid.csv"
    evaluate_solver("ao_grid", cfg, seed=0, start=0, count=1, output=ao_grid_path, bank=bank)
    ao_grid_row = pd.read_csv(ao_grid_path).iloc[0]
    assert ao_grid_row["solver"] == "ao_grid_coordinate_codebook"
    assert "allocation_levels" in ao_grid_row["grid"]
