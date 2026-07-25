from __future__ import annotations

import numpy as np

from star_ris_rsma.action import DecodedAction, action_dim, decode_action, encode_action
from star_ris_rsma.config import ExperimentConfig


def test_legacy_parameterization_remains_restricted_and_unchanged() -> None:
    n_users, n_ris = 4, 2
    raw = np.zeros(action_dim(n_users, n_ris), dtype=float)
    raw[0] = 1.0
    raw[1:n_users + 1] = -1.0
    legacy = decode_action(raw, n_users, n_ris, 1.0, "legacy_v1")
    assert np.isclose(legacy.powers.sum(), 1.0)
    assert 0.64 < legacy.powers[0] < 0.66
    assert np.all((legacy.beta_t >= 0.268) & (legacy.beta_t <= 0.732))


def test_physical_v3_reaches_simplex_vertices_beta_bounds_and_full_phase() -> None:
    n_users, n_ris = 4, 2
    raw = np.full(action_dim(n_users, n_ris), -1.0, dtype=float)
    raw[0] = 1.0
    i = n_users + 1
    raw[i] = 1.0
    i += n_users
    raw[i:i + n_ris] = np.array([-1.0, 1.0])
    i += n_ris
    raw[i:i + n_ris] = np.array([-1.0, 1.0])
    i += n_ris
    raw[i:i + n_ris] = np.array([1.0, -1.0])

    action = decode_action(raw, n_users, n_ris, 1.0, "physical_v3")
    assert np.allclose(action.powers, [1.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(action.common_fractions, [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(action.beta_t, [0.0, 1.0])
    assert np.allclose(np.exp(1j * action.theta_t), [-1.0, -1.0])
    assert np.allclose(np.exp(1j * action.theta_r), [-1.0, -1.0])


def test_physical_v3_round_trip_preserves_feasible_action() -> None:
    action = DecodedAction(
        powers=np.array([0.55, 0.20, 0.15, 0.10, 0.0]),
        common_fractions=np.array([0.60, 0.20, 0.15, 0.05]),
        beta_t=np.array([0.0, 0.25, 0.75, 1.0]),
        theta_t=np.array([-np.pi, -0.7, 0.3, 2.4]),
        theta_r=np.array([2.8, -2.2, 0.0, 1.1]),
    )
    raw = encode_action(action, 1.0, "physical_v3")
    decoded = decode_action(raw, 4, 4, 1.0, "physical_v3")
    assert np.all(raw >= -1.0) and np.all(raw <= 1.0)
    assert np.allclose(decoded.powers, action.powers, atol=1e-12)
    assert np.allclose(decoded.common_fractions, action.common_fractions, atol=1e-12)
    assert np.allclose(decoded.beta_t, action.beta_t, atol=1e-12)
    assert np.allclose(np.exp(1j * decoded.theta_t), np.exp(1j * action.theta_t), atol=1e-12)
    assert np.allclose(np.exp(1j * decoded.theta_r), np.exp(1j * action.theta_r), atol=1e-12)


def test_config_defaults_preserve_legacy_and_accept_physical_v3() -> None:
    assert ExperimentConfig().action_parameterization == "legacy_v1"
    assert ExperimentConfig(action_parameterization="physical_v3").action_parameterization == "physical_v3"
