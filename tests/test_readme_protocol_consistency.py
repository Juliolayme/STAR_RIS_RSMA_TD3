from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_root_readme_uses_constrained_v3_examples() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "--config configs/v3/constrained_action_n32.yaml" in text
    assert "configs/v3/constrained_action_n16.yaml" in text
    assert "configs/v3/constrained_action_n128.yaml" in text
    assert "--config configs/siso_n" not in text


def test_root_readme_matches_frozen_latency_count() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "--warmup 20 --count 100" in text
    assert "--warmup 20 --count 500" not in text
    assert "exactly 100 latency samples for every method/N pair" in text


def test_frozen_latency_table_is_exactly_100_samples_per_method_n() -> None:
    path = ROOT / "results" / "six_method_v1" / "tables" / "TABLE_SIX_METHOD_CPU_LATENCY.csv"
    table = pd.read_csv(path)
    assert not table.empty
    assert set(table["count"].astype(int)) == {100}
    assert len(table) == 6 * 5


def test_six_method_readme_distinguishes_gate_from_frozen_count() -> None:
    text = (ROOT / "experiments" / "six_method" / "README.md").read_text(encoding="utf-8")
    assert "configs/v3/constrained_action_n*.yaml" in text
    assert "exactly 100 single-thread CPU latency samples" in text
    assert "at least 100 single-thread CPU latency samples" in text
    assert "`--count 500`" in text
    assert "optional extended latency experiment" in text
