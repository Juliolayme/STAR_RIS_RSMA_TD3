from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_six_method_report import (
    BASELINE_METHODS,
    CORE_METRICS,
    DRL_METHODS,
    METHODS,
    N_VALUES,
    SEEDS,
    build_performance_table,
    finite_numeric,
    load_latency,
    paired_tests,
    plot_latency,
    plot_metric,
    plot_quality_latency,
)

AO_FREEZE = "corrected_pairwise_ao_v2:max_iter=80:stationarity_tol=1e-6"
GRID_FREEZE = "corrected_ao_grid_v1:rounds=2:zero_level:bidirectional_ris"


def load_frozen_drl(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {
        "method",
        "n_ris",
        "seed",
        "scenario",
        "bank_checksum",
        *CORE_METRICS,
    }
    if not required.issubset(raw.columns):
        raise RuntimeError(f"Frozen DRL raw missing columns: {sorted(required - set(raw.columns))}")
    raw["method"] = raw.method.astype(str).str.lower()
    raw = raw[raw.method.isin(DRL_METHODS)].copy()
    finite_numeric(raw, ("n_ris", "seed", "scenario", *CORE_METRICS), "frozen DRL raw")

    if raw.duplicated(["method", "n_ris", "seed", "scenario"]).any():
        raise RuntimeError("Duplicate frozen DRL method/N/seed/scenario keys")

    for method in DRL_METHODS:
        for n_ris in N_VALUES:
            group = raw[(raw.method == method) & (raw.n_ris.astype(int) == n_ris)]
            if len(group) != 8000:
                raise RuntimeError(f"{method} N={n_ris}: expected 8000 rows, got {len(group)}")
            if set(group.seed.astype(int)) != set(SEEDS):
                raise RuntimeError(f"{method} N={n_ris}: seed coverage mismatch")
            for seed in SEEDS:
                scenarios = group[group.seed.astype(int) == seed].scenario.astype(int)
                if len(scenarios) != 1000 or set(scenarios) != set(range(1000)):
                    raise RuntimeError(f"{method} N={n_ris} seed={seed}: scenario coverage mismatch")
            if group.bank_checksum.astype(str).nunique() != 1:
                raise RuntimeError(f"{method} N={n_ris}: multiple bank checksums")

    for n_ris in N_VALUES:
        checksums = {
            raw[(raw.method == method) & (raw.n_ris.astype(int) == n_ris)]
            .bank_checksum.astype(str)
            .iloc[0]
            for method in DRL_METHODS
        }
        if len(checksums) != 1:
            raise RuntimeError(f"Frozen DRL methods do not share the test bank at N={n_ris}")
    return raw


def load_corrected_baselines(root: Path, drl: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.rglob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        required = {"method", "n_ris", "scenario", "bank_checksum", *CORE_METRICS}
        if required.issubset(frame.columns):
            methods = set(frame.method.astype(str).str.lower())
            if methods.intersection(BASELINE_METHODS):
                frame["source_file"] = str(path)
                frames.append(frame)
    if not frames:
        raise RuntimeError(f"No corrected baseline CSVs found under {root}")

    raw = pd.concat(frames, ignore_index=True)
    raw["method"] = raw.method.astype(str).str.lower()
    raw = raw[raw.method.isin(BASELINE_METHODS)].copy()
    if raw.duplicated(["method", "n_ris", "scenario"]).any():
        duplicated = raw[raw.duplicated(["method", "n_ris", "scenario"], keep=False)]
        raise RuntimeError(f"Duplicate corrected baseline keys:\n{duplicated.head(20)}")
    if "seed" not in raw.columns:
        raw["seed"] = 0
    finite_numeric(raw, ("n_ris", "seed", "scenario", *CORE_METRICS), "corrected baselines")

    for method in BASELINE_METHODS:
        for n_ris in N_VALUES:
            group = raw[(raw.method == method) & (raw.n_ris.astype(int) == n_ris)]
            if len(group) != 1000:
                raise RuntimeError(f"{method} N={n_ris}: expected 1000 rows, got {len(group)}")
            if set(group.scenario.astype(int)) != set(range(1000)):
                raise RuntimeError(f"{method} N={n_ris}: scenario coverage mismatch")
            if group.bank_checksum.astype(str).nunique() != 1:
                raise RuntimeError(f"{method} N={n_ris}: multiple bank checksums")
            expected = drl[drl.n_ris.astype(int) == n_ris].bank_checksum.astype(str).iloc[0]
            observed = group.bank_checksum.astype(str).iloc[0]
            if observed != expected:
                raise RuntimeError(
                    f"ScenarioBank mismatch N={n_ris}: {method}={observed}, frozen_drl={expected}"
                )

    ao_versions = set(raw[raw.method == "ao_sca"].algorithm_version.astype(str))
    grid_versions = set(raw[raw.method == "ao_grid"].algorithm_version.astype(str))
    if ao_versions != {"corrected_pairwise_ao_v2"}:
        raise RuntimeError(f"Unexpected corrected AO versions: {ao_versions}")
    if grid_versions != {"corrected_ao_grid_v1"}:
        raise RuntimeError(f"Unexpected corrected AO-Grid versions: {grid_versions}")
    if set(raw[raw.method == "ao_sca"].max_iter.astype(int)) != {80}:
        raise RuntimeError("Corrected AO is not uniformly frozen at max_iter=80")
    return raw


def write_review(path: Path, performance: pd.DataFrame, latency: pd.DataFrame) -> None:
    best_quality = performance.loc[
        performance.groupby("n_ris")["sum_rate_mean"].idxmax(),
        ["n_ris", "method", "sum_rate_mean"],
    ]
    fastest = latency.loc[
        latency.groupby("n_ris")["solve_ms_median"].idxmin(),
        ["n_ris", "method", "solve_ms_median"],
    ]
    lines = [
        "# Six-method STAR-RIS–RSMA benchmark v2",
        "",
        "## Frozen protocol",
        "",
        "- Learned methods (unchanged canonical evidence): TD3, DDPG, PPO; 8 seeds × 1,000 locked test scenarios per N.",
        "- Continuous traditional baseline: corrected pairwise AO, frozen at max_iter=80.",
        "- Discrete traditional baseline: corrected AO-Grid with zero simplex level and forward/reverse RIS sweeps.",
        "- AnalyticalRIS is unchanged.",
        "- N = 16, 32, 64, 96, 128.",
        "- All six methods use identical locked ScenarioBank checksums at each N.",
        "- Holm correction is applied separately within each N over all 15 pairwise method comparisons.",
        "- DRL seeds are averaged per locked scenario before paired tests to avoid pseudo-replication.",
        "- CPU decision latency uses one thread, warmup=10, count=100, and all six methods are measured on the same GitHub Actions runner.",
        "",
        "## Best observed sum-rate by N",
        "",
        best_quality.to_markdown(index=False),
        "",
        "## Fastest observed decision method by N",
        "",
        fastest.to_markdown(index=False),
        "",
        "## Interpretation guardrails",
        "",
        "- Corrected pairwise AO is a deterministic local continuous baseline, not a global optimum or an upper bound.",
        "- Corrected AO-Grid is a restricted discrete heuristic, not a global optimizer.",
        "- Exact slide values must be copied from the v2 tables generated in this artifact.",
        "",
        f"AO freeze: `{AO_FREEZE}`.",
        f"AO-Grid freeze: `{GRID_FREEZE}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drl-raw", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--latency-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    tables = args.output / "tables"
    figures = args.output / "figures"
    raw_dir = args.output / "raw"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)

    drl = load_frozen_drl(args.drl_raw)
    baselines = load_corrected_baselines(args.baseline_root, drl)
    performance = build_performance_table(drl, baselines)
    tests = paired_tests(drl, baselines)
    latency_raw, latency_summary = load_latency(args.latency_root)

    # Validate the latency banks against the same canonical frozen DRL checksums.
    for n_ris in N_VALUES:
        expected = drl[drl.n_ris.astype(int) == n_ris].bank_checksum.astype(str).iloc[0]
        for method in METHODS:
            group = latency_raw[
                (latency_raw.method == method) & (latency_raw.n_ris.astype(int) == n_ris)
            ]
            observed = set(group.bank_checksum.astype(str))
            if observed != {expected}:
                raise RuntimeError(
                    f"Latency ScenarioBank mismatch N={n_ris} {method}: {observed} != {expected}"
                )

    drl.to_csv(raw_dir / "DRL_TEST_RAW_ALL.csv", index=False)
    baselines.sort_values(["method", "n_ris", "scenario"]).to_csv(
        raw_dir / "TRADITIONAL_TEST_RAW_ALL.csv", index=False
    )
    latency_raw.sort_values(["method", "n_ris", "scenario"]).to_csv(
        raw_dir / "CPU_LATENCY_RAW_ALL.csv", index=False
    )
    performance.to_csv(tables / "TABLE_SIX_METHOD_PERFORMANCE.csv", index=False)
    tests.to_csv(tables / "TABLE_SIX_METHOD_PAIRED_TESTS_HOLM.csv", index=False)
    latency_summary.to_csv(tables / "TABLE_SIX_METHOD_CPU_LATENCY.csv", index=False)

    td3_tests = tests[tests.method_a == "td3"].copy()
    td3_tests.to_csv(tables / "TABLE_TD3_VS_OTHERS_HOLM.csv", index=False)

    slide_metrics = performance.merge(latency_summary, on=["method", "n_ris"], how="left")
    slide_metrics[
        [
            "method",
            "n_ris",
            "sum_rate_mean",
            "sum_rate_ci95_low",
            "sum_rate_ci95_high",
            "qos_fraction_mean",
            "all_qos_mean",
            "violation_mean",
            "solve_ms_median",
            "solve_ms_mean",
        ]
    ].to_csv(tables / "TABLE_SIX_METHOD_SLIDE_METRICS.csv", index=False)

    plot_metric(performance, "sum_rate", "Mean sum-rate", figures / "fig01_six_method_sum_rate")
    plot_metric(performance, "qos_fraction", "Mean QoS fraction", figures / "fig02_six_method_qos_fraction")
    plot_metric(performance, "all_qos", "Probability all users satisfy QoS", figures / "fig03_six_method_all_qos")
    plot_metric(
        performance,
        "violation",
        "Mean QoS violation",
        figures / "fig04_six_method_violation",
        log_y=True,
    )
    plot_latency(latency_summary, figures / "fig05_six_method_cpu_latency")
    plot_quality_latency(performance, latency_summary, figures / "fig06_six_method_quality_latency")
    write_review(args.output / "SIX_METHOD_V2_REVIEW.md", performance, latency_summary)

    checksums = {
        str(n): drl[drl.n_ris.astype(int) == n].bank_checksum.astype(str).iloc[0]
        for n in N_VALUES
    }
    audit = {
        "verdict": "PASS",
        "methods": list(METHODS),
        "n_values": list(N_VALUES),
        "drl_source": "frozen results/six_method_v1/raw/DRL_TEST_RAW_ALL.csv",
        "drl_seeds": list(SEEDS),
        "test_scenarios_per_n": 1000,
        "ao_freeze": AO_FREEZE,
        "ao_grid_freeze": GRID_FREEZE,
        "shared_scenario_banks": True,
        "scenario_bank_checksums": checksums,
        "paired_test_family": "15 pairwise comparisons per N; Holm adjusted separately for paired t and Wilcoxon",
        "latency_protocol": "single-thread same GitHub Actions runner; warmup=10; count=100",
        "published_tables": sorted(path.name for path in tables.glob("*.csv")),
        "published_figures": sorted(path.name for path in figures.glob("*.png")),
    }
    (args.output / "SIX_METHOD_V2_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
