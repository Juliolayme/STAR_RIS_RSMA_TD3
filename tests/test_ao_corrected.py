from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from star_ris_rsma.baselines import ao_corrected


def test_optimizer_probe_covers_stationarity_probe() -> None:
    assert ao_corrected.FROZEN_PAIRWISE_PROBE <= (
        ao_corrected.FROZEN_STATIONARITY_PROBE
    )


def test_small_objective_change_does_not_stop_with_open_simplex_gap(monkeypatch) -> None:
    """A RIS-reopened simplex direction is polished before outer stopping."""

    state = SimpleNamespace(
        score=1.0,
        vector=np.zeros(4, dtype=float),
        action=object(),
        metrics={},
    )
    env = SimpleNamespace(
        config=SimpleNamespace(
            n_users=1, n_ris=1, p_max=1.0, action_parameterization="legacy"
        )
    )
    slices = {
        "powers": slice(0, 1),
        "common": slice(1, 2),
        "beta": slice(2, 3),
        "theta_r": slice(3, 4),
    }
    gap_values = iter((1e-3, 0.0, 0.0, 0.0))

    monkeypatch.setattr(ao_corrected, "physical_slices", lambda *_: slices)
    monkeypatch.setattr(ao_corrected, "analytical_action", lambda _: object())
    monkeypatch.setattr(ao_corrected, "state_from_action", lambda *_: state)
    monkeypatch.setattr(
        ao_corrected,
        "_pairwise_simplex_ascent",
        lambda _env, current, *_args, **_kwargs: (current, 0, 0),
    )
    monkeypatch.setattr(
        ao_corrected,
        "_block_gradient",
        lambda *_args, **_kwargs: (np.zeros(4, dtype=float), 0),
    )
    monkeypatch.setattr(
        ao_corrected,
        "_stationarity_gap",
        lambda *_args, **_kwargs: (next(gap_values), 1),
    )
    monkeypatch.setattr(
        ao_corrected,
        "encode_action",
        lambda *_args, **_kwargs: np.zeros(1, dtype=float),
    )

    _, metrics = ao_corrected.solve(env, max_iter=5)

    assert metrics["iterations"] == 1
    assert metrics["simplex_polish_sweeps"] == 2
    assert metrics["termination_reason"] == "objective_and_simplex_stationarity"
    assert metrics["power_stationarity_gap"] == 0.0
    assert metrics["common_stationarity_gap"] == 0.0
    assert metrics["stationarity_tolerance"] == 1e-6


def test_corrected_ao_v2_reports_max_iter_when_not_stationary(monkeypatch) -> None:
    state = SimpleNamespace(
        score=1.0,
        vector=np.zeros(4, dtype=float),
        action=object(),
        metrics={},
    )
    env = SimpleNamespace(
        config=SimpleNamespace(
            n_users=1, n_ris=1, p_max=1.0, action_parameterization="legacy"
        )
    )
    slices = {
        "powers": slice(0, 1),
        "common": slice(1, 2),
        "beta": slice(2, 3),
        "theta_r": slice(3, 4),
    }

    monkeypatch.setattr(ao_corrected, "physical_slices", lambda *_: slices)
    monkeypatch.setattr(ao_corrected, "analytical_action", lambda _: object())
    monkeypatch.setattr(ao_corrected, "state_from_action", lambda *_: state)
    monkeypatch.setattr(
        ao_corrected,
        "_pairwise_simplex_ascent",
        lambda _env, current, *_args, **_kwargs: (current, 0, 0),
    )
    monkeypatch.setattr(
        ao_corrected,
        "_block_gradient",
        lambda *_args, **_kwargs: (np.zeros(4, dtype=float), 0),
    )
    monkeypatch.setattr(
        ao_corrected,
        "_stationarity_gap",
        lambda *_args, **_kwargs: (1e-3, 1),
    )
    monkeypatch.setattr(
        ao_corrected,
        "encode_action",
        lambda *_args, **_kwargs: np.zeros(1, dtype=float),
    )

    _, metrics = ao_corrected.solve(
        env, max_iter=2, simplex_polish_max_sweeps=2
    )

    assert metrics["iterations"] == 2
    assert metrics["termination_reason"] == "max_iter"
    assert metrics["algorithm_version"] == "corrected_pairwise_ao_v2"
    assert metrics["simplex_polish_sweeps"] == 4
