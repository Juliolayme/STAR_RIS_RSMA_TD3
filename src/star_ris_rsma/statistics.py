from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * float(p_values[idx]))
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def _mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x))
    if x.size < 2:
        return mean, mean, mean
    sem = stats.sem(x)
    half = float(stats.t.ppf((1.0 + confidence) / 2.0, x.size - 1) * sem)
    return mean, mean - half, mean + half


def summarize(df: pd.DataFrame, metric: str = "sum_rate") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, group in df.groupby("method"):
        values = group[metric].to_numpy(float)
        mean, low, high = _mean_ci(values)
        rows.append({
            "method": method,
            "metric": metric,
            "count": len(values),
            "mean": mean,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "ci95_low": low,
            "ci95_high": high,
        })
    return pd.DataFrame(rows).sort_values("mean", ascending=False)


def paired_comparisons(df: pd.DataFrame, metric: str = "sum_rate") -> pd.DataFrame:
    required = {"method", "scenario", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Average repeated training seeds first, then pair methods on identical scenarios.
    collapsed = df.groupby(["method", "scenario"], as_index=False)[metric].mean()
    methods = sorted(collapsed["method"].unique())
    rows: list[dict[str, object]] = []
    p_values: list[float] = []
    for left, right in combinations(methods, 2):
        a = collapsed[collapsed.method == left][["scenario", metric]].rename(columns={metric: "left"})
        b = collapsed[collapsed.method == right][["scenario", metric]].rename(columns={metric: "right"})
        paired = a.merge(b, on="scenario", how="inner")
        diff = paired["left"].to_numpy(float) - paired["right"].to_numpy(float)
        if diff.size < 2:
            continue
        if np.allclose(diff, 0.0):
            t_stat, t_p = 0.0, 1.0
            w_stat, w_p = 0.0, 1.0
        else:
            t_stat, t_p = stats.ttest_rel(paired["left"], paired["right"])
            try:
                w_stat, w_p = stats.wilcoxon(diff)
            except ValueError:
                w_stat, w_p = 0.0, 1.0
        std_diff = float(np.std(diff, ddof=1))
        dz = 0.0 if std_diff < 1e-12 else float(np.mean(diff) / std_diff)
        row = {
            "method_a": left,
            "method_b": right,
            "metric": metric,
            "n_pairs": diff.size,
            "mean_difference": float(np.mean(diff)),
            "paired_t_stat": float(t_stat),
            "paired_t_p": float(t_p),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(w_p),
            "cohen_dz": dz,
        }
        rows.append(row)
        p_values.append(float(t_p))
    if rows:
        adjusted = holm_adjust(p_values)
        for row, value in zip(rows, adjusted):
            row["paired_t_p_holm"] = value
    return pd.DataFrame(rows)


def write_analysis(df: pd.DataFrame, output_dir: str | Path, metric: str = "sum_rate") -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / f"summary_{metric}.csv"
    comparisons_path = target / f"paired_{metric}.csv"
    summarize(df, metric).to_csv(summary_path, index=False)
    paired_comparisons(df, metric).to_csv(comparisons_path, index=False)
    return summary_path, comparisons_path
