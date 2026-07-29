from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DRL_METHODS = ("td3", "ddpg", "ppo")
BASELINE_METHODS = ("ao_sca", "ao_grid", "analytical_ris")
METHODS = DRL_METHODS + BASELINE_METHODS
N_VALUES = (16, 32, 64, 96, 128)
SEEDS = tuple(range(8))
CORE_METRICS = ("sum_rate", "qos_fraction", "all_qos", "violation")
REQUIRED_STAGES = {
    "td3_low_n",
    "td3_high_n",
    "ddpg_low_n",
    "ddpg_high_n",
    "ppo_low_n",
    "ppo_high_n",
}


def finite_numeric(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    numeric = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError(f"NaN/Inf or nonnumeric values in {context}")


def discover_drl(root: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    manifests: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    for manifest_path in sorted(root.rglob("STAGE_MANIFEST.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stage_id = str(manifest.get("stage_id", ""))
        if stage_id not in REQUIRED_STAGES:
            continue
        manifests.append(manifest)
        csvs = sorted(manifest_path.parent.glob("*_TEST_RAW_ALL.csv"))
        if len(csvs) != 1:
            raise RuntimeError(f"Stage {stage_id} must contain one TEST_RAW_ALL CSV: {csvs}")
        frame = pd.read_csv(csvs[0])
        frame["source_stage"] = stage_id
        frames.append(frame)

    stage_ids = {str(item.get("stage_id")) for item in manifests}
    if stage_ids != REQUIRED_STAGES:
        raise RuntimeError(f"Missing or extra DRL stages: expected={REQUIRED_STAGES}, got={stage_ids}")
    commits = {str(item.get("repository_commit")) for item in manifests}
    protocols = {str(item.get("training_protocol")) for item in manifests}
    if len(commits) != 1:
        raise RuntimeError(f"DRL stages used different commits: {commits}")
    if protocols != {"drl_v3_qos_constrained_fair"}:
        raise RuntimeError(f"Unexpected DRL protocols: {protocols}")

    raw = pd.concat(frames, ignore_index=True)
    required = {
        "method",
        "n_ris",
        "seed",
        "scenario",
        "bank_checksum",
        *CORE_METRICS,
    }
    if not required.issubset(raw.columns):
        raise RuntimeError(f"DRL output missing columns: {sorted(required - set(raw.columns))}")
    raw["method"] = raw["method"].astype(str).str.lower()
    finite_numeric(raw, ("n_ris", "seed", "scenario", *CORE_METRICS), "DRL raw results")

    duplicates = raw.duplicated(["method", "n_ris", "seed", "scenario"], keep=False)
    if duplicates.any():
        raise RuntimeError("Duplicate DRL method/N/seed/scenario keys")
    for method in DRL_METHODS:
        for n_ris in N_VALUES:
            group = raw[(raw.method == method) & (raw.n_ris == n_ris)]
            if len(group) != 8000:
                raise RuntimeError(f"{method} N={n_ris}: expected 8000 rows, got {len(group)}")
            if set(group.seed.astype(int)) != set(SEEDS):
                raise RuntimeError(f"{method} N={n_ris}: seed coverage mismatch")
            for seed in SEEDS:
                scenarios = group[group.seed == seed].scenario.astype(int)
                if len(scenarios) != 1000 or set(scenarios) != set(range(1000)):
                    raise RuntimeError(f"{method} N={n_ris} seed={seed}: scenario coverage mismatch")
            if group.bank_checksum.nunique() != 1:
                raise RuntimeError(f"{method} N={n_ris}: multiple test-bank checksums")

    for n_ris in N_VALUES:
        checksums = {
            raw[(raw.method == method) & (raw.n_ris == n_ris)].bank_checksum.iloc[0]
            for method in DRL_METHODS
        }
        if len(checksums) != 1:
            raise RuntimeError(f"DRL methods do not share the test bank at N={n_ris}")
    return raw, manifests


def discover_baselines(root: Path) -> pd.DataFrame:
    candidates: list[pd.DataFrame] = []
    for path in sorted(root.rglob("*.csv")):
        try:
            header = pd.read_csv(path, nrows=3)
        except Exception:
            continue
        required = {"method", "n_ris", "scenario", "bank_checksum", *CORE_METRICS}
        if not required.issubset(header.columns):
            continue
        methods = set(header["method"].astype(str).str.lower())
        if not methods.intersection(BASELINE_METHODS):
            continue
        frame = pd.read_csv(path)
        frame["method"] = frame["method"].astype(str).str.lower()
        frame = frame[frame.method.isin(BASELINE_METHODS)].copy()
        if not frame.empty:
            frame["source_file"] = str(path)
            candidates.append(frame)
    if not candidates:
        raise RuntimeError(f"No baseline raw CSVs found under {root}")

    raw = pd.concat(candidates, ignore_index=True)
    raw = raw.drop_duplicates(["method", "n_ris", "scenario"], keep="first")
    if "seed" not in raw.columns:
        raw["seed"] = 0
    finite_numeric(raw, ("n_ris", "seed", "scenario", *CORE_METRICS), "baseline raw results")
    for method in BASELINE_METHODS:
        for n_ris in N_VALUES:
            group = raw[(raw.method == method) & (raw.n_ris == n_ris)]
            if len(group) != 1000:
                raise RuntimeError(f"{method} N={n_ris}: expected 1000 rows, got {len(group)}")
            scenarios = group.scenario.astype(int)
            if set(scenarios) != set(range(1000)):
                raise RuntimeError(f"{method} N={n_ris}: scenario coverage mismatch")
            if group.bank_checksum.nunique() != 1:
                raise RuntimeError(f"{method} N={n_ris}: multiple bank checksums")
    return raw


def validate_shared_banks(drl: pd.DataFrame, baselines: pd.DataFrame) -> None:
    for n_ris in N_VALUES:
        expected = drl[drl.n_ris == n_ris].bank_checksum.iloc[0]
        for method in BASELINE_METHODS:
            observed = baselines[
                (baselines.method == method) & (baselines.n_ris == n_ris)
            ].bank_checksum.iloc[0]
            if observed != expected:
                raise RuntimeError(
                    f"ScenarioBank mismatch at N={n_ris}: {method}={observed}, DRL={expected}"
                )


def t_interval(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if len(values) <= 1 or std == 0.0:
        return mean, std, mean, mean
    half = float(stats.t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values)))
    return mean, std, mean - half, mean + half


def build_performance_table(drl: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        source = drl if method in DRL_METHODS else baselines
        for n_ris in N_VALUES:
            group = source[(source.method == method) & (source.n_ris == n_ris)]
            unit = "seed_mean" if method in DRL_METHODS else "scenario"
            if method in DRL_METHODS:
                samples = group.groupby("seed")[list(CORE_METRICS)].mean()
            else:
                samples = group[list(CORE_METRICS)]
            row: dict[str, object] = {
                "method": method,
                "n_ris": n_ris,
                "seeds": int(group.seed.nunique()),
                "test_scenarios": int(group.scenario.nunique()),
                "uncertainty_unit": unit,
            }
            for metric in CORE_METRICS:
                mean, std, low, high = t_interval(samples[metric].to_numpy(dtype=float))
                if metric in {"qos_fraction", "all_qos"}:
                    low, high = max(0.0, low), min(1.0, high)
                if metric == "violation":
                    low = max(0.0, low)
                row.update(
                    {
                        f"{metric}_mean": mean,
                        f"{metric}_std": std,
                        f"{metric}_ci95_low": low,
                        f"{metric}_ci95_high": high,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["n_ris", "method"]).reset_index(drop=True)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def paired_tests(drl: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([drl, baselines], ignore_index=True, sort=False)
    rows: list[dict[str, object]] = []
    for n_ris in N_VALUES:
        data = combined[combined.n_ris == n_ris]
        scenario_means = (
            data.groupby(["method", "scenario"], as_index=False)["sum_rate"].mean()
        )
        pivot = scenario_means.pivot(index="scenario", columns="method", values="sum_rate")
        local_rows: list[dict[str, object]] = []
        for first, second in itertools.combinations(METHODS, 2):
            pair = pivot[[first, second]].dropna()
            difference = pair[first].to_numpy() - pair[second].to_numpy()
            t_result = stats.ttest_rel(pair[first], pair[second], nan_policy="raise")
            try:
                w_result = stats.wilcoxon(
                    difference,
                    zero_method="wilcox",
                    correction=False,
                    alternative="two-sided",
                    method="auto",
                )
                wilcoxon_stat = float(w_result.statistic)
                wilcoxon_p = float(w_result.pvalue)
            except ValueError:
                wilcoxon_stat, wilcoxon_p = 0.0, 1.0
            std = float(np.std(difference, ddof=1))
            local_rows.append(
                {
                    "n_ris": n_ris,
                    "method_a": first,
                    "method_b": second,
                    "paired_scenarios": len(pair),
                    "mean_difference_a_minus_b": float(np.mean(difference)),
                    "cohen_dz": float(np.mean(difference) / std) if std > 0 else 0.0,
                    "paired_t_statistic": float(t_result.statistic),
                    "paired_t_p": float(t_result.pvalue),
                    "wilcoxon_statistic": wilcoxon_stat,
                    "wilcoxon_p": wilcoxon_p,
                }
            )
        t_adjusted = holm_adjust([float(item["paired_t_p"]) for item in local_rows])
        w_adjusted = holm_adjust([float(item["wilcoxon_p"]) for item in local_rows])
        for item, t_p, w_p in zip(local_rows, t_adjusted, w_adjusted):
            item["paired_t_holm_p"] = t_p
            item["wilcoxon_holm_p"] = w_p
            item["paired_t_holm_significant_0_05"] = t_p < 0.05
            item["wilcoxon_holm_significant_0_05"] = w_p < 0.05
        rows.extend(local_rows)
    return pd.DataFrame(rows)


def load_latency(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.rglob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        required = {"method", "n_ris", "scenario", "solve_ms"}
        if required.issubset(frame.columns):
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"No latency CSVs found under {root}")
    raw = pd.concat(frames, ignore_index=True)
    raw["method"] = raw.method.astype(str).str.lower()
    raw = raw[raw.method.isin(METHODS)].copy()
    finite_numeric(raw, ("n_ris", "scenario", "solve_ms"), "latency")
    duplicates = raw.duplicated(["method", "n_ris", "scenario"], keep=False)
    if duplicates.any():
        raise RuntimeError("Duplicate latency method/N/scenario keys")
    for method in METHODS:
        for n_ris in N_VALUES:
            group = raw[(raw.method == method) & (raw.n_ris == n_ris)]
            if len(group) < 100:
                raise RuntimeError(f"Latency coverage too small for {method} N={n_ris}: {len(group)}")
    summary = (
        raw.groupby(["method", "n_ris"])["solve_ms"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "solve_ms_mean", "std": "solve_ms_std", "median": "solve_ms_median"})
    )
    return raw, summary


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metric(table: pd.DataFrame, metric: str, ylabel: str, output: Path, log_y: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for method in METHODS:
        group = table[table.method == method].sort_values("n_ris")
        mean = group[f"{metric}_mean"].to_numpy(dtype=float)
        low = group[f"{metric}_ci95_low"].to_numpy(dtype=float)
        high = group[f"{metric}_ci95_high"].to_numpy(dtype=float)
        errors = np.vstack([np.maximum(mean - low, 0.0), np.maximum(high - mean, 0.0)])
        ax.errorbar(group.n_ris, mean, yerr=errors, marker="o", capsize=3, label=method)
    ax.set_xlabel("Number of STAR-RIS elements, N")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    if log_y:
        ax.set_yscale("symlog", linthresh=1e-6)
    save_figure(fig, output)


def plot_latency(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for method in METHODS:
        group = summary[summary.method == method].sort_values("n_ris")
        ax.plot(group.n_ris, group.solve_ms_median, marker="o", label=method)
    ax.set_xlabel("Number of STAR-RIS elements, N")
    ax.set_ylabel("Median CPU decision latency (ms)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    save_figure(fig, output)


def plot_quality_latency(performance: pd.DataFrame, latency: pd.DataFrame, output: Path) -> None:
    merged = performance.merge(latency, on=["method", "n_ris"], how="inner")
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for method in METHODS:
        group = merged[merged.method == method].sort_values("n_ris")
        ax.plot(group.solve_ms_median, group.sum_rate_mean, marker="o", label=method)
        for row in group.itertuples():
            ax.annotate(f"N={row.n_ris}", (row.solve_ms_median, row.sum_rate_mean), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Median CPU decision latency (ms, log scale)")
    ax.set_ylabel("Mean sum-rate")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    save_figure(fig, output)


def write_report(
    output: Path,
    performance: pd.DataFrame,
    latency: pd.DataFrame,
    manifests: list[dict[str, object]],
) -> None:
    best_quality = performance.loc[
        performance.groupby("n_ris")["sum_rate_mean"].idxmax(),
        ["n_ris", "method", "sum_rate_mean"],
    ]
    fastest = latency.loc[
        latency.groupby("n_ris")["solve_ms_median"].idxmin(),
        ["n_ris", "method", "solve_ms_median"],
    ]
    lines = [
        "# Six-method STAR-RIS–RSMA benchmark",
        "",
        "## Validated protocol",
        "",
        "- Learned methods: TD3, DDPG, PPO.",
        "- Traditional methods: AO-SCA, AO-Grid, AnalyticalRIS.",
        "- N = 16, 32, 64, 96, 128.",
        "- DRL: eight seeds and 1,000 locked test scenarios per seed/N.",
        "- Traditional baselines: the same 1,000 locked scenarios per N.",
        "- All DRL stages share one repository commit and the QoS-feasibility-first protocol.",
        "- CPU latency is single-threaded and measured on the same runner for all methods.",
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
        "- AO-SCA remains a local iterative baseline, not a global optimum or upper bound.",
        "- Do not claim one DRL algorithm dominates unless quality, QoS, latency and corrected tests agree.",
        "- Exact numerical claims must be copied from TABLE_SIX_METHOD_PERFORMANCE.csv.",
        "- The paired tests average the eight DRL seeds per locked scenario before comparison, avoiding pseudo-replication.",
        "",
        f"DRL repository commit: `{manifests[0]['repository_commit']}`.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drl-root", type=Path, required=True)
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

    drl, manifests = discover_drl(args.drl_root)
    baselines = discover_baselines(args.baseline_root)
    validate_shared_banks(drl, baselines)
    performance = build_performance_table(drl, baselines)
    tests = paired_tests(drl, baselines)
    latency_raw, latency_summary = load_latency(args.latency_root)

    drl.to_csv(raw_dir / "DRL_TEST_RAW_ALL.csv", index=False)
    baselines.to_csv(raw_dir / "TRADITIONAL_TEST_RAW_ALL.csv", index=False)
    latency_raw.to_csv(raw_dir / "CPU_LATENCY_RAW_ALL.csv", index=False)
    performance.to_csv(tables / "TABLE_SIX_METHOD_PERFORMANCE.csv", index=False)
    tests.to_csv(tables / "TABLE_SIX_METHOD_PAIRED_TESTS_HOLM.csv", index=False)
    latency_summary.to_csv(tables / "TABLE_SIX_METHOD_CPU_LATENCY.csv", index=False)

    plot_metric(performance, "sum_rate", "Mean sum-rate", figures / "fig01_six_method_sum_rate")
    plot_metric(performance, "qos_fraction", "Mean QoS fraction", figures / "fig02_six_method_qos_fraction")
    plot_metric(performance, "all_qos", "Probability all users satisfy QoS", figures / "fig03_six_method_all_qos")
    plot_metric(performance, "violation", "Mean QoS violation", figures / "fig04_six_method_violation", log_y=True)
    plot_latency(latency_summary, figures / "fig05_six_method_cpu_latency")
    plot_quality_latency(performance, latency_summary, figures / "fig06_six_method_quality_latency")
    write_report(args.output / "SIX_METHOD_REVIEW.md", performance, latency_summary, manifests)

    audit = {
        "verdict": "PASS",
        "methods": list(METHODS),
        "n_values": list(N_VALUES),
        "drl_seeds": list(SEEDS),
        "test_scenarios_per_n": 1000,
        "required_stages": sorted(REQUIRED_STAGES),
        "drl_repository_commit": manifests[0]["repository_commit"],
        "shared_scenario_banks": True,
        "finite_core_metrics": True,
        "latency_min_samples_per_method_n": 100,
        "published_tables": sorted(path.name for path in tables.glob("*.csv")),
        "published_figures": sorted(path.name for path in figures.glob("*.png")),
    }
    (args.output / "SIX_METHOD_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
