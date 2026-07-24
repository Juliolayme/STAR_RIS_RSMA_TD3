from __future__ import annotations

"""Quality-data validation, statistical analysis, and academic figures."""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from kaggle_final_report_common import (
    FIGURE_DIR,
    METHODS,
    N_VALUES,
    SEEDS,
    ensure_finite,
    parse_n_seed,
)

QUALITY_METRICS = ("sum_rate", "qos_fraction", "all_qos", "violation")


def load_td3_outputs(
    stage_roots: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and audit all 5 N × 8 seed TD3 outputs.

    Every test file must contain the exact scenario IDs 0..999. Training and
    validation histories are joined only after the N/seed pair has been parsed
    from the canonical output layout.
    """
    test_frames: list[pd.DataFrame] = []
    training_frames: list[pd.DataFrame] = []
    validation_frames: list[pd.DataFrame] = []

    for stage_id in ("td3_low_n", "td3_high_n"):
        root = stage_roots[stage_id]
        for test_path in root.rglob("test.csv"):
            n_ris, seed = parse_n_seed(test_path)
            test = pd.read_csv(test_path)
            test["n_ris"] = n_ris
            test["seed"] = seed
            test["method"] = "td3"
            ensure_finite(test, QUALITY_METRICS, f"TD3 N={n_ris}, seed={seed}")
            test_frames.append(test)

            train_root = test_path.parent / "train"
            training = pd.read_csv(train_root / "training.csv")
            training["n_ris"] = n_ris
            training["seed"] = seed
            training_frames.append(training)

            validation = pd.read_csv(train_root / "validation_summary.csv")
            validation["n_ris"] = n_ris
            validation["seed"] = seed
            validation_frames.append(validation)

    if not test_frames:
        raise RuntimeError("No TD3 test files were discovered")
    test_all = pd.concat(test_frames, ignore_index=True)
    training_all = pd.concat(training_frames, ignore_index=True)
    validation_all = pd.concat(validation_frames, ignore_index=True)

    expected_pairs = {(n, seed) for n in N_VALUES for seed in SEEDS}
    actual_pairs = set(zip(test_all["n_ris"].astype(int), test_all["seed"].astype(int)))
    if actual_pairs != expected_pairs:
        raise RuntimeError(
            "TD3 seed coverage mismatch. "
            f"Missing={sorted(expected_pairs - actual_pairs)}, "
            f"extra={sorted(actual_pairs - expected_pairs)}"
        )

    for (n_ris, seed), group in test_all.groupby(["n_ris", "seed"], sort=True):
        scenarios = sorted(pd.to_numeric(group["scenario"]).astype(int).tolist())
        if len(group) != 1000 or scenarios != list(range(1000)):
            raise RuntimeError(f"Incomplete TD3 N={n_ris}, seed={seed}")
        if group["bank_checksum"].astype(str).nunique() != 1:
            raise RuntimeError(f"Multiple TD3 bank checksums N={n_ris}, seed={seed}")

    return test_all, training_all, validation_all


def load_baseline_outputs(stage_roots: dict[str, Path]) -> pd.DataFrame:
    """Load the three deterministic baseline tables and audit 1,000 scenarios/N."""
    frames: list[pd.DataFrame] = []
    for method in ("ao_grid", "ao_sca", "analytical_ris"):
        files = list(stage_roots[method].glob(f"{method.upper()}_RAW_ALL.csv"))
        if len(files) != 1:
            raise RuntimeError(f"Expected one merged raw CSV for {method}, found {files}")
        frame = pd.read_csv(files[0])
        frame["method"] = method
        ensure_finite(frame, (*QUALITY_METRICS, "solve_ms"), method)
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    for (method, n_ris), group in raw.groupby(["method", "n_ris"], sort=True):
        scenarios = sorted(pd.to_numeric(group["scenario"]).astype(int).tolist())
        if len(group) != 1000 or scenarios != list(range(1000)):
            raise RuntimeError(f"Incomplete {method} N={n_ris}")
        if group["bank_checksum"].astype(str).nunique() != 1:
            raise RuntimeError(f"Multiple {method} checksums N={n_ris}")
    return raw


def validate_cross_method_checksums(
    td3_test: pd.DataFrame,
    baseline_raw: pd.DataFrame,
) -> None:
    """Ensure TD3 and every baseline use the identical locked test bank per N."""
    for n_ris in N_VALUES:
        td3_checksums = set(
            td3_test[td3_test["n_ris"].astype(int) == n_ris]["bank_checksum"].astype(str)
        )
        if len(td3_checksums) != 1:
            raise RuntimeError(f"TD3 checksum disagreement across seeds for N={n_ris}")
        for method in ("ao_grid", "ao_sca", "analytical_ris"):
            baseline_checksums = set(
                baseline_raw[
                    (baseline_raw["method"] == method)
                    & (baseline_raw["n_ris"].astype(int) == n_ris)
                ]["bank_checksum"].astype(str)
            )
            if baseline_checksums != td3_checksums:
                raise RuntimeError(
                    f"Locked-bank mismatch for {method}, N={n_ris}: "
                    f"TD3={td3_checksums}, baseline={baseline_checksums}"
                )


def t_ci95(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return mean, sample SD, and a two-sided Student-t 95% confidence interval."""
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    half = float(stats.t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
    return mean, std, mean - half, mean + half


def build_performance_tables(
    td3_test: pd.DataFrame,
    baseline_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the eight-seed TD3 CI table and complete method comparison table."""
    seed_means = (
        td3_test.groupby(["n_ris", "seed"])[list(QUALITY_METRICS)]
        .mean(numeric_only=True)
        .reset_index()
    )

    td3_rows: list[dict[str, object]] = []
    for n_ris, group in seed_means.groupby("n_ris", sort=True):
        row: dict[str, object] = {"method": "td3", "n_ris": int(n_ris), "seeds": 8}
        for metric in QUALITY_METRICS:
            mean, std, low, high = t_ci95(group[metric].to_numpy(float))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        td3_rows.append(row)
    td3_summary = pd.DataFrame(td3_rows)

    baseline_rows: list[dict[str, object]] = []
    for (method, n_ris), group in baseline_raw.groupby(["method", "n_ris"], sort=True):
        row: dict[str, object] = {"method": method, "n_ris": int(n_ris), "seeds": 1}
        for metric in QUALITY_METRICS:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
            row[f"{metric}_ci95_low"] = np.nan
            row[f"{metric}_ci95_high"] = np.nan
        baseline_rows.append(row)

    final_table = pd.concat(
        [td3_summary, pd.DataFrame(baseline_rows)], ignore_index=True
    ).sort_values(["n_ris", "method"], ignore_index=True)
    return td3_summary, final_table


def build_baseline_timing_table(baseline_raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize the original solver-reported wall time for audit purposes.

    This table is descriptive only. The primary fair online-computation claim
    must use the single-process benchmark generated by notebook 06.
    """
    rows: list[dict[str, object]] = []
    for (method, n_ris), group in baseline_raw.groupby(["method", "n_ris"], sort=True):
        values = pd.to_numeric(group["solve_ms"], errors="raise").to_numpy(float)
        rows.append(
            {
                "method": method,
                "n_ris": int(n_ris),
                "scenarios": int(len(values)),
                "solve_ms_mean": float(np.mean(values)),
                "solve_ms_std": float(np.std(values, ddof=1)),
                "solve_ms_median": float(np.median(values)),
                "solve_ms_p95": float(np.quantile(values, 0.95)),
                "solve_ms_p99": float(np.quantile(values, 0.99)),
            }
        )
    return pd.DataFrame(rows)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Apply Holm's step-down family-wise error correction."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def paired_seed_level_tests(
    td3_test: pd.DataFrame,
    baseline_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Compare TD3 with each baseline using matched scenarios and seed effects.

    For every TD3 seed, the mean paired scenario difference is computed on the
    identical 1,000 channel realizations. The eight independent training-seed
    effects are tested with a one-sample t-test and Wilcoxon signed-rank test.
    Holm correction covers all 15 baseline-by-N hypotheses.
    """
    rows: list[dict[str, object]] = []
    for method in ("ao_sca", "ao_grid", "analytical_ris"):
        for n_ris in N_VALUES:
            baseline = baseline_raw[
                (baseline_raw["method"] == method)
                & (baseline_raw["n_ris"].astype(int) == n_ris)
            ][["scenario", "sum_rate", "bank_checksum"]].rename(
                columns={"sum_rate": "baseline_sum_rate"}
            )

            seed_effects: list[float] = []
            for seed in SEEDS:
                td3 = td3_test[
                    (td3_test["n_ris"].astype(int) == n_ris)
                    & (td3_test["seed"].astype(int) == seed)
                ][["scenario", "sum_rate", "bank_checksum"]]
                matched = td3.merge(
                    baseline,
                    on=["scenario", "bank_checksum"],
                    how="inner",
                    validate="one_to_one",
                )
                if len(matched) != 1000:
                    raise RuntimeError(
                        f"Scenario/checksum mismatch for {method}, N={n_ris}, seed={seed}"
                    )
                seed_effects.append(
                    float((matched["sum_rate"] - matched["baseline_sum_rate"]).mean())
                )

            effects = np.asarray(seed_effects, dtype=float)
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                effects,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
            t_stat, t_p = stats.ttest_1samp(effects, popmean=0.0)
            effect_std = float(np.std(effects, ddof=1))
            rows.append(
                {
                    "baseline": method,
                    "n_ris": n_ris,
                    "td3_minus_baseline_mean": float(np.mean(effects)),
                    "td3_minus_baseline_std": effect_std,
                    "cohen_dz": (
                        float(np.mean(effects) / effect_std)
                        if effect_std > 0
                        else np.nan
                    ),
                    "paired_t_statistic": float(t_stat),
                    "paired_t_p_value": float(t_p),
                    "wilcoxon_statistic": float(wilcoxon_stat),
                    "wilcoxon_p_value": float(wilcoxon_p),
                }
            )

    result = pd.DataFrame(rows)
    result["paired_t_p_holm"] = holm_adjust(result["paired_t_p_value"])
    result["wilcoxon_p_holm"] = holm_adjust(result["wilcoxon_p_value"])
    result["wilcoxon_significant_0_05"] = result["wilcoxon_p_holm"] < 0.05
    return result


def save_figure(fig: plt.Figure, name: str) -> None:
    """Write one publication figure as a 300-DPI PNG and vector PDF."""
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(training: pd.DataFrame, validation: pd.DataFrame) -> None:
    """Plot five TD3 convergence diagnostics with eight-seed uncertainty bands."""
    specifications = [
        (training, "step", "sum_rate", "Training sum-rate", "fig01_training_sum_rate", False),
        (training, "step", "qos_fraction", "Training QoS fraction", "fig02_training_qos_fraction", False),
        (training, "step", "violation", "Training QoS violation", "fig03_training_violation", True),
        (training, "step", "qos_dual", "Adaptive QoS dual multiplier", "fig04_qos_dual", False),
        (validation, "eval_step", "mean_sum_rate", "Validation sum-rate", "fig05_validation_sum_rate", False),
    ]
    for source, step_col, metric, ylabel, name, log_y in specifications:
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        for n_ris in N_VALUES:
            group = source[source["n_ris"].astype(int) == n_ris]
            summary = group.groupby(step_col)[metric].agg(["mean", "std", "count"]).reset_index()
            critical = stats.t.ppf(0.975, np.maximum(summary["count"] - 1, 1))
            ci = critical * summary["std"].fillna(0.0) / np.sqrt(summary["count"])
            ax.plot(summary[step_col], summary["mean"], label=f"N={n_ris}")
            ax.fill_between(
                summary[step_col],
                summary["mean"] - ci,
                summary["mean"] + ci,
                alpha=0.18,
            )
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2)
        save_figure(fig, name)


def plot_final_quality(final_table: pd.DataFrame) -> None:
    """Plot four final metrics; TD3 error bars are eight-seed t-based 95% CIs."""
    specifications = [
        ("sum_rate", "Sum-rate", "fig06_final_sum_rate", False),
        ("qos_fraction", "QoS fraction", "fig07_final_qos_fraction", False),
        ("all_qos", "All-users QoS probability", "fig08_final_all_qos", False),
        ("violation", "Mean QoS violation", "fig09_final_violation", True),
    ]
    for metric, ylabel, name, log_y in specifications:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for method in METHODS:
            group = final_table[final_table["method"] == method].sort_values("n_ris")
            if method == "td3":
                y = group[f"{metric}_mean"].to_numpy(float)
                low = group[f"{metric}_ci95_low"].to_numpy(float)
                high = group[f"{metric}_ci95_high"].to_numpy(float)
                ax.errorbar(
                    group["n_ris"],
                    y,
                    yerr=np.vstack([y - low, high - y]),
                    marker="o",
                    capsize=3,
                    label=method,
                )
            else:
                ax.plot(group["n_ris"], group[f"{metric}_mean"], marker="o", label=method)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Number of STAR-RIS elements, N")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        save_figure(fig, name)
