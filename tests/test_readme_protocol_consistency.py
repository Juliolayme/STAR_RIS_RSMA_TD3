from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "results" / "physical_v6_full_r2"


def test_root_readme_documents_v6_protocol() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "configs/v3/pilot_v6_soft_anchor_n32.yaml" in text
    assert "physical_v6_soft_anchor" in text
    assert "results/physical_v6_full_r2/" in text
    assert "thesis-v6" in text
    assert "--warmup 10 --count 100" in text
    assert "--config configs/siso_n" not in text
    for n in (16, 32, 64, 96, 128):
        assert (ROOT / "configs" / "v3" / f"pilot_v6_soft_anchor_n{n}.yaml").exists()


def test_v6_latency_table_is_exactly_100_samples_per_method_n() -> None:
    table = pd.read_csv(V6 / "tables" / "TABLE_V6_SIX_METHOD_CPU_LATENCY.csv")
    assert set(table["count"].astype(int)) == {100}
    assert len(table) == 6 * 5


def test_v6_performance_table_covers_six_methods_five_n() -> None:
    table = pd.read_csv(V6 / "tables" / "TABLE_V6_SIX_METHOD_PERFORMANCE.csv")
    assert set(table["method"]) == {"td3", "ddpg", "ppo", "ao_sca", "ao_grid", "analytical_ris"}
    assert set(table["n_ris"].astype(int)) == {16, 32, 64, 96, 128}
    assert len(table) == 30


def test_traditional_raw_matches_published_ao_means() -> None:
    raw = pd.read_csv(ROOT / "results" / "six_method_v2" / "raw" / "TRADITIONAL_TEST_RAW_ALL.csv")
    perf = pd.read_csv(V6 / "tables" / "TABLE_V6_SIX_METHOD_PERFORMANCE.csv").set_index(["method", "n_ris"])
    means = raw.groupby(["method", "n_ris"]).sum_rate.mean()
    for (method, n), value in means.items():
        assert abs(value - perf.loc[(method, n), "sum_rate_mean"]) < 1e-6


def test_frozen_v1_latency_table_kept_for_history() -> None:
    table = pd.read_csv(ROOT / "results" / "six_method_v1" / "tables" / "TABLE_SIX_METHOD_CPU_LATENCY.csv")
    assert set(table["count"].astype(int)) == {100}
    assert len(table) == 6 * 5


def test_six_method_readme_distinguishes_gate_from_frozen_count() -> None:
    text = (ROOT / "experiments" / "six_method" / "README.md").read_text(encoding="utf-8")
    assert "configs/v3/constrained_action_n*.yaml" in text
    assert "exactly 100 single-thread CPU latency samples" in text
    assert "at least 100 single-thread CPU latency samples" in text
    assert "`--count 500`" in text
    assert "optional extended latency experiment" in text
