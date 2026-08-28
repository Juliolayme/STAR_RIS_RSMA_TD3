"""The two inference units the paired tests report.

The scenario-level test pairs over 1,000 locked scenarios that are all scored
by the same five policies, so it resolves hundredths and marked 73 of 75
method pairs significant. A method-level claim has to survive retraining, so
the seed-level test pairs over the five seeds instead.
"""

from pathlib import Path
import runpy

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
REPORT = runpy.run_path(ROOT / "scripts" / "build_physical_v6_full_report.py")
seed_level_test = REPORT["seed_level_test"]
holm_adjust = REPORT["holm_adjust"]


def test_two_learned_methods_are_paired_over_their_seeds() -> None:
    means = {"td3": [1.0, 1.1, 0.9, 1.2, 1.0], "ddpg": [1.5, 1.6, 1.4, 1.7, 1.5]}
    result = seed_level_test("td3", "ddpg", means)
    assert result["seed_level_n"] == 5
    assert "paired over 5 training seeds" == result["seed_level_unit"]
    assert result["seed_mean_difference_a_minus_b"] == pytest.approx(-0.5)
    # A constant offset across every seed is unambiguous.
    assert result["seed_t_p"] < 1e-6


def test_a_deterministic_baseline_is_a_fixed_value_not_a_paired_sample() -> None:
    means = {"td3": [1.0, 1.1, 0.9, 1.2, 1.0], "ao_grid": [2.0]}
    result = seed_level_test("td3", "ao_grid", means)
    assert result["seed_level_n"] == 5
    assert "deterministic baseline" in result["seed_level_unit"]
    assert result["seed_mean_difference_a_minus_b"] == pytest.approx(1.04 - 2.0)
    # Five seeds against a fixed value: clearly significant, but n=5 bounds
    # how small the p-value can get.
    assert result["seed_t_p"] < 1e-3


def test_the_sign_follows_the_argument_order_either_way_round() -> None:
    means = {"td3": [1.0, 1.1, 0.9, 1.2, 1.0], "ao_grid": [2.0]}
    forward = seed_level_test("td3", "ao_grid", means)
    reverse = seed_level_test("ao_grid", "td3", means)
    assert forward["seed_mean_difference_a_minus_b"] == pytest.approx(
        -reverse["seed_mean_difference_a_minus_b"]
    )
    assert forward["seed_t_statistic"] == pytest.approx(-reverse["seed_t_statistic"])
    assert forward["seed_t_p"] == pytest.approx(reverse["seed_t_p"])


def test_two_deterministic_baselines_get_no_fabricated_p_value() -> None:
    means = {"ao_grid": [2.0], "ao_sca": [1.7]}
    result = seed_level_test("ao_grid", "ao_sca", means)
    assert result["seed_level_n"] == 0
    assert "not applicable" in result["seed_level_unit"]
    assert np.isnan(result["seed_t_p"])
    assert result["seed_mean_difference_a_minus_b"] == pytest.approx(0.3)


def test_seed_noise_can_outweigh_a_difference_scenarios_call_certain() -> None:
    """The case the change exists for: a real but seed-unstable difference."""
    means = {
        "td3": [20.975, 20.947, 20.830, 20.848, 20.997],
        "ddpg": [21.024, 21.094, 20.921, 20.992, 20.915],
    }
    result = seed_level_test("td3", "ddpg", means)
    assert result["seed_mean_difference_a_minus_b"] == pytest.approx(-0.0695, abs=5e-4)
    # Scenario-level Holm put this at 5e-35.
    assert result["seed_t_p"] > 0.05


def test_holm_is_monotone_and_bounded() -> None:
    adjusted = holm_adjust([0.001, 0.02, 0.5])
    assert adjusted == sorted(adjusted)
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert adjusted[0] >= 0.001
