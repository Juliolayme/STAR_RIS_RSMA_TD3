import pandas as pd

from star_ris_rsma.latency_aggregation import summarize_latency_by_n
from star_ris_rsma.result_validation import CORE_METRICS


def _frame(n_ris: int, checksum: str) -> pd.DataFrame:
    rows = []
    for scenario in range(3):
        row = {
            "n_ris": n_ris,
            "seed": 0,
            "scenario": scenario,
            "bank_checksum": checksum,
            "td3_actor_ms": 0.10 + 0.01 * scenario,
            "td3_decode_ms": 0.05,
            "td3_decision_ms": 0.15 + 0.01 * scenario,
            "td3_end_to_end_ms": 0.20 + 0.01 * scenario,
            "ao_sca_end_to_end_ms": 20.0 + scenario,
            "ao_grid_end_to_end_ms": 5.0 + scenario,
            "analytical_ris_end_to_end_ms": 0.05 + 0.01 * scenario,
        }
        for method in ("td3", "ao_sca", "ao_grid", "analytical_ris"):
            values = {
                "sum_rate": 1.0 + scenario,
                "qos_fraction": 1.0,
                "all_qos": True,
                "violation": 0.0,
            }
            for metric in CORE_METRICS:
                row[f"{method}_{metric}"] = values[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def test_multi_n_aggregation_validates_each_locked_bank_separately():
    merged = pd.concat(
        [
            _frame(16, "bank-N16"),
            _frame(32, "bank-N32"),
        ],
        ignore_index=True,
    )

    summary = summarize_latency_by_n(merged).sort_values("n_ris").reset_index(drop=True)

    assert summary["n_ris"].tolist() == [16, 32]
    assert summary["scenarios"].tolist() == [3, 3]
    assert summary["bank_checksum"].tolist() == ["bank-N16", "bank-N32"]
