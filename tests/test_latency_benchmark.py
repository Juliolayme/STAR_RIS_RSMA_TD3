import numpy as np
import pandas as pd
import pytest

from star_ris_rsma.latency import summarize_latency, validate_latency_frame
from star_ris_rsma.result_validation import CORE_METRICS


def _frame() -> pd.DataFrame:
    rows = []
    for scenario in range(3):
        row = {
            "n_ris": 16,
            "seed": 0,
            "scenario": scenario,
            "bank_checksum": "locked-test-bank",
            "td3_actor_ms": 0.10 + 0.01 * scenario,
            "td3_decode_ms": 0.05,
            "td3_decision_ms": 0.15 + 0.01 * scenario,
            "td3_end_to_end_ms": 0.20 + 0.01 * scenario,
            "ao_sca_end_to_end_ms": 20.0 + scenario,
            "ao_grid_end_to_end_ms": 5.0 + scenario,
            "analytical_ris_end_to_end_ms": 0.05 + 0.01 * scenario,
        }
        for method in ("td3", "ao_sca", "ao_grid", "analytical_ris"):
            row[f"{method}_sum_rate"] = 1.0 + scenario
            row[f"{method}_qos_fraction"] = 1.0
            row[f"{method}_all_qos"] = "true" if scenario % 2 == 0 else False
            row[f"{method}_violation"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_latency_validation_accepts_mixed_bool_metrics_and_finite_times():
    frame = _frame()
    validate_latency_frame(frame, expected_n=16, expected_seed=0, expected_rows=3)


def test_latency_summary_reports_solver_over_td3_ratios():
    summary = summarize_latency(_frame()).iloc[0]
    assert summary["ao_sca_over_td3_ratio_of_means"] > 90.0
    assert summary["ao_grid_over_td3_ratio_of_means"] > 20.0
    assert summary["analytical_ris_over_td3_ratio_of_means"] < 1.0
    assert summary["scenarios"] == 3


def test_latency_validation_rejects_duplicate_scenario():
    frame = _frame()
    frame.loc[2, "scenario"] = 1
    with pytest.raises(ValueError, match="unique scenario"):
        validate_latency_frame(frame)


def test_latency_validation_rejects_nonfinite_or_nonpositive_time():
    frame = _frame()
    frame.loc[1, "ao_grid_end_to_end_ms"] = np.nan
    with pytest.raises(ValueError, match="finite and strictly positive"):
        validate_latency_frame(frame)
