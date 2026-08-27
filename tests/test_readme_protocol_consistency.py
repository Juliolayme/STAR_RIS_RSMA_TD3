from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_root_readme_uses_constrained_v3_examples() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "configs/v3/pilot_v6_soft_anchor_n32.yaml" in text
    assert "configs/v3/pilot_v6_soft_anchor_n16.yaml" in text
    assert "configs/v3/pilot_v6_soft_anchor_n128.yaml" in text
    assert "configs/v3/constrained_action_n32.yaml" in text


def test_root_readme_matches_frozen_latency_count() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "benchmark_latency_v2.py --warmup 10 --count 100" in text
    assert "exactly 100 latency samples" in text
    assert "every method/N pair" in text
    assert "six_method_v1" not in text


def test_frozen_latency_table_is_exactly_100_samples_per_method_n() -> None:
    path = ROOT / "results" / "six_method_v2" / "tables" / "TABLE_SIX_METHOD_CPU_LATENCY.csv"
    table = pd.read_csv(path)
    assert not table.empty
    assert set(table["count"].astype(int)) == {100}
    assert len(table) == 6 * 5


def test_v6_readme_records_self_contained_baseline_evidence() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "results/six_method_v2/" in text
    assert "results/physical_v6_full/" in text
