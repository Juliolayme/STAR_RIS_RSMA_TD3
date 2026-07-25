from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


N_VALUES = (16, 32, 64, 96, 128)
SEEDS = tuple(range(8))
PROPOSED = "td3_v2_fixed"
BASELINES = ("td3", "ddpg", "ppo")
METRICS = ("sum_rate", "qos_fraction", "all_qos", "violation")
ALPHA = 0.05
EXPECTED_SCENARIOS = 1000


def normalize_method(value: object) -> str:
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "td3": "td3",
        "ddpg": "ddpg",
        "ppo": "ppo",
        "td3_v2_fixed": PROPOSED,
        "v2_fixed_budget": PROPOSED,
    }
    return aliases.get(text, text)


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float)
    mapping = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
    return series.astype(str).str.strip().str.lower().map(mapping).astype(float)


def config_hash_map(root: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        config = payload.get("config")
        config_hash = payload.get("config_hash")
        if isinstance(config, dict) and config_hash and "n_ris" in config:
            n_ris = int(config["n_ris"])
            if n_ris in N_VALUES:
                prior = mapping.get(str(config_hash))
                if prior is not None and prior != n_ris:
                    raise RuntimeError(f"Conflicting n_ris for config hash {config_hash}: {prior} vs {n_ris}")
                mapping[str(config_hash)] = n_ris
    return mapping


def n_from_path(path: Path) -> int | None:
    text = path.as_posix()
    matches = re.findall(r"(?i)(?:^|[^a-z0-9])n(?:=|[_-])?(16|32|64|96|128)(?=$|[^0-9])", text)
    if matches:
        return int(matches[-1])
    return None


def infer_n(path: Path, frame: pd.DataFrame, hash_to_n: dict[str, int]) -> int:
    if "n_ris" in frame.columns:
        values = pd.to_numeric(frame["n_ris"], errors="coerce").dropna().astype(int).unique()
        if len(values) == 1 and int(values[0]) in N_VALUES:
            return int(values[0])
    if "config_hash" in frame.columns:
        hashes = frame["config_hash"].dropna().astype(str).unique()
        mapped = {hash_to_n[h] for h in hashes if h in hash_to_n}
        if len(mapped) == 1:
            return int(next(iter(mapped)))
    inferred = n_from_path(path)
    if inferred is None:
        raise RuntimeError(f"Cannot infer n_ris for test CSV: {path}")
    return inferred


def test_like(frame: pd.DataFrame, path: Path) -> bool:
    required = {"scenario", "seed", *METRICS}
    if not required.issubset(frame.columns):
        return False
    if "split" in frame.columns:
        splits = set(frame["split"].dropna().astype(str).str.lower().unique())
        if splits and splits != {"test"}:
            return False
        if splits == {"test"}:
            return True
    return "test" in path.as_posix().lower()


def load_root(root: Path, source: str) -> pd.DataFrame:
    hash_to_n = config_hash_map(root)
    frames: list[pd.DataFrame] = []
    for path in sorted(root.rglob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if not test_like(frame, path):
            continue
        n_ris = infer_n(path, frame, hash_to_n)
        if source == "proposed":
            method = PROPOSED
        else:
            if "method" not in frame.columns:
                continue
            methods = {normalize_method(value) for value in frame["method"].dropna().unique()}
            methods &= set(BASELINES)
            if len(methods) != 1:
                continue
            method = next(iter(methods))

        selected = frame.copy()
        selected["method"] = method
        selected["n_ris"] = n_ris
        selected["source_path"] = path.as_posix()
        selected["seed"] = pd.to_numeric(selected["seed"], errors="raise").astype(int)
        selected["scenario"] = pd.to_numeric(selected["scenario"], errors="raise").astype(int)
        for metric in ("sum_rate", "qos_fraction", "violation"):
            selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
        selected["all_qos"] = parse_bool_series(selected["all_qos"])
        frames.append(selected)

    if not frames:
        raise RuntimeError(f"No deterministic test CSVs found below {root}")
    return pd.concat(frames, ignore_index=True)


def exact_deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "n_ris", "seed", "scenario"]
    metric_cols = list(METRICS)
    rows: list[pd.DataFrame] = []
    for _, group in frame.groupby(keys, sort=False):
        if len(group) > 1:
            reference = group.iloc[0][metric_cols].astype(float).to_numpy()
            for _, candidate in group.iloc[1:].iterrows():
                values = candidate[metric_cols].astype(float).to_numpy()
                if not np.allclose(reference, values, rtol=1e-10, atol=1e-12, equal_nan=False):
                    raise RuntimeError(
                        "Conflicting duplicate test rows for "
                        f"{tuple(group.iloc[0][key] for key in keys)}"
                    )
        rows.append(group.iloc[[0]])
    return pd.concat(rows, ignore_index=True)


def validate_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    expected_methods = (PROPOSED, *BASELINES)
    records: list[dict[str, object]] = []
    errors: list[str] = []
    for method in expected_methods:
        for n_ris in N_VALUES:
            for seed in SEEDS:
                group = frame[
                    (frame.method == method)
                    & (frame.n_ris == n_ris)
                    & (frame.seed == seed)
                ]
                scenarios = sorted(group.scenario.unique())
                finite = bool(np.isfinite(group[list(METRICS)].to_numpy(dtype=float)).all()) if len(group) else False
                checksum_count = (
                    int(group["bank_checksum"].dropna().astype(str).nunique())
                    if "bank_checksum" in group.columns
                    else 0
                )
                complete = len(group) == EXPECTED_SCENARIOS and len(scenarios) == EXPECTED_SCENARIOS
                if complete:
                    complete = scenarios[0] == 0 and scenarios[-1] == EXPECTED_SCENARIOS - 1
                issue_parts: list[str] = []
                if not complete:
                    issue_parts.append(f"rows/scenarios={len(group)}/{len(scenarios)}")
                if not finite:
                    issue_parts.append("non-finite metric")
                if checksum_count != 1:
                    issue_parts.append(f"bank checksum count={checksum_count}")
                records.append({
                    "method": method,
                    "n_ris": n_ris,
                    "seed": seed,
                    "rows": len(group),
                    "scenario_count": len(scenarios),
                    "finite": finite,
                    "bank_checksum_count": checksum_count,
                    "complete": complete and finite and checksum_count == 1,
                    "issues": "; ".join(issue_parts),
                })
                if issue_parts:
                    errors.append(f"{method} N={n_ris} seed={seed}: {'; '.join(issue_parts)}")

    coverage = pd.DataFrame(records)
    for n_ris in N_VALUES:
        subset = frame[frame.n_ris == n_ris]
        if "bank_checksum" not in subset.columns:
            errors.append(f"N={n_ris}: bank_checksum column missing")
            continue
        checks = subset.bank_checksum.dropna().astype(str).unique()
        if len(checks) != 1:
            errors.append(f"N={n_ris}: expected one shared test-bank checksum, found {len(checks)}")

    if errors:
        raise RuntimeError("Coverage validation failed:\n" + "\n".join(errors[:100]))
    return coverage


def t_interval(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float, float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    n = len(array)
    if n == 0:
        return math.nan, math.nan, math.nan, math.nan, 0
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if n > 1 else 0.0
    if n == 1:
        return mean, std, mean, mean, n
    sem = std / math.sqrt(n)
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1))
    return mean, std, mean - critical * sem, mean + critical * sem, n


def seed_level_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_level = (
        frame.groupby(["method", "n_ris", "seed"], as_index=False)[list(METRICS)]
        .mean()
        .sort_values(["method", "n_ris", "seed"])
    )
    rows: list[dict[str, object]] = []
    for (method, n_ris), group in seed_level.groupby(["method", "n_ris"], sort=True):
        for metric in METRICS:
            mean, std, low, high, n = t_interval(group[metric])
            rows.append({
                "method": method,
                "n_ris": int(n_ris),
                "metric": metric,
                "seed_count": n,
                "mean": mean,
                "std": std,
                "ci95_low": low,
                "ci95_high": high,
                "ci95_half_width": (high - low) / 2.0,
            })
    return seed_level, pd.DataFrame(rows)


def paired_stats(frame: pd.DataFrame) -> pd.DataFrame:
    collapsed = (
        frame.groupby(["method", "n_ris", "scenario"], as_index=False)[list(METRICS)]
        .mean()
        .sort_values(["method", "n_ris", "scenario"])
    )
    rows: list[dict[str, object]] = []
    for n_ris in N_VALUES:
        proposed = collapsed[(collapsed.method == PROPOSED) & (collapsed.n_ris == n_ris)].set_index("scenario")
        for baseline in BASELINES:
            comparison = collapsed[(collapsed.method == baseline) & (collapsed.n_ris == n_ris)].set_index("scenario")
            common = proposed.index.intersection(comparison.index)
            if len(common) != EXPECTED_SCENARIOS:
                raise RuntimeError(f"Paired scenario coverage mismatch for {baseline}, N={n_ris}: {len(common)}")
            for metric in METRICS:
                x = proposed.loc[common, metric].to_numpy(dtype=float)
                y = comparison.loc[common, metric].to_numpy(dtype=float)
                difference = x - y
                mean, std, low, high, n = t_interval(difference)
                t_result = stats.ttest_rel(x, y, nan_policy="raise")
                if np.allclose(difference, 0.0, rtol=0.0, atol=1e-15):
                    wilcoxon_stat, wilcoxon_p = 0.0, 1.0
                else:
                    wilcoxon = stats.wilcoxon(difference, alternative="two-sided", zero_method="wilcox", method="auto")
                    wilcoxon_stat, wilcoxon_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
                dz = mean / std if std > 0 else (math.copysign(math.inf, mean) if mean != 0 else 0.0)
                rows.append({
                    "proposed": PROPOSED,
                    "baseline": baseline,
                    "n_ris": n_ris,
                    "metric": metric,
                    "paired_scenarios": n,
                    "mean_delta_proposed_minus_baseline": mean,
                    "delta_std": std,
                    "delta_ci95_low": low,
                    "delta_ci95_high": high,
                    "paired_t_statistic": float(t_result.statistic),
                    "paired_t_p_raw": float(t_result.pvalue),
                    "wilcoxon_statistic": wilcoxon_stat,
                    "wilcoxon_p_raw": wilcoxon_p,
                    "cohen_dz": dz,
                })
    return pd.DataFrame(rows)


def holm_adjust(values: pd.Series) -> pd.Series:
    p = values.astype(float).to_numpy()
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * p[index])
        running = max(running, candidate)
        adjusted[index] = running
    return pd.Series(adjusted, index=values.index)


def add_holm(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["paired_t_p_holm"] = np.nan
    result["wilcoxon_p_holm"] = np.nan
    # A family contains the 3 baselines x 5 N values for one metric.
    for metric, index in result.groupby("metric").groups.items():
        idx = list(index)
        result.loc[idx, "paired_t_p_holm"] = holm_adjust(result.loc[idx, "paired_t_p_raw"])
        result.loc[idx, "wilcoxon_p_holm"] = holm_adjust(result.loc[idx, "wilcoxon_p_raw"])
    result["paired_t_significant_holm_0_05"] = result["paired_t_p_holm"] < ALPHA
    result["wilcoxon_significant_holm_0_05"] = result["wilcoxon_p_holm"] < ALPHA
    return result


def fmt(value: object, digits: int = 5) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def write_report(ci: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    lines = [
        "# Final TD3 v2 fixed-budget statistics",
        "",
        "## Locked protocol",
        "",
        "- Proposed method: **TD3 v2 fixed-budget**",
        "- RIS sizes: **16, 32, 64, 96, 128**",
        "- Training seeds: **0-7 (8 seeds)**",
        "- Deterministic test scenarios per N: **1,000**",
        "- Seed uncertainty: Student-t **95% confidence interval** over per-seed test means",
        "- Paired tests: repeated seeds are first averaged within each locked scenario; tests then use 1,000 paired scenario values",
        "- Multiplicity: Holm correction separately within each metric family over 15 hypotheses (3 baselines x 5 N values)",
        "- Tests: paired t-test, Wilcoxon signed-rank, Cohen's dz",
        "",
        "## Proposed method: mean ± 95% CI across seeds",
        "",
        "| N | Sum-rate | QoS fraction | All-QoS | Violation |",
        "|---:|---:|---:|---:|---:|",
    ]
    proposed_ci = ci[ci.method == PROPOSED]
    for n_ris in N_VALUES:
        subset = proposed_ci[proposed_ci.n_ris == n_ris].set_index("metric")
        cells = []
        for metric in METRICS:
            row = subset.loc[metric]
            cells.append(f"{fmt(row['mean'])} [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]")
        lines.append(f"| {n_ris} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Holm-corrected paired comparisons",
        "",
        "Positive delta means TD3 v2 fixed-budget is larger. For violation, a negative delta is favorable.",
        "",
        "| Baseline | N | Metric | Mean delta | 95% CI | t-Holm p | Wilcoxon-Holm p | dz |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ])
    for _, row in paired.sort_values(["baseline", "n_ris", "metric"]).iterrows():
        lines.append(
            f"| {row.baseline} | {int(row.n_ris)} | {row.metric} | "
            f"{fmt(row.mean_delta_proposed_minus_baseline)} | "
            f"[{fmt(row.delta_ci95_low)}, {fmt(row.delta_ci95_high)}] | "
            f"{fmt(row.paired_t_p_holm, 6)} | {fmt(row.wilcoxon_p_holm, 6)} | {fmt(row.cohen_dz)} |"
        )

    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "- Reward is not compared because v1 and v2 use different QoS reward definitions.",
        "- Statistical significance does not replace reporting effect size and confidence intervals.",
        "- The primary scientific claims should emphasize sum-rate, QoS feasibility, violation, and seed robustness.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    proposed = load_root(args.proposed_root, source="proposed")
    baselines = load_root(args.baseline_root, source="baseline")
    baselines = baselines[baselines.method.isin(BASELINES)].copy()
    combined = exact_deduplicate(pd.concat([proposed, baselines], ignore_index=True))
    coverage = validate_coverage(combined)
    seed_level, ci = seed_level_summary(combined)
    paired = add_holm(paired_stats(combined))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compact_columns = [
        "method", "n_ris", "seed", "scenario", *METRICS,
        *(["bank_checksum"] if "bank_checksum" in combined.columns else []),
        *(["config_hash"] if "config_hash" in combined.columns else []),
        "source_path",
    ]
    combined[compact_columns].to_csv(args.output_dir / "MERGED_RAW_TEST.csv", index=False)
    coverage.to_csv(args.output_dir / "COVERAGE.csv", index=False)
    seed_level.to_csv(args.output_dir / "SEED_LEVEL_MEANS.csv", index=False)
    ci.to_csv(args.output_dir / "CI95_SUMMARY.csv", index=False)
    paired.to_csv(args.output_dir / "PAIRED_TESTS_HOLM.csv", index=False)
    write_report(ci, paired, args.output_dir / "FINAL_NUMBERS.md")
    (args.output_dir / "STATISTICS_PROTOCOL.json").write_text(
        json.dumps({
            "proposed": PROPOSED,
            "baselines": BASELINES,
            "n_values": N_VALUES,
            "seeds": SEEDS,
            "test_scenarios_per_n": EXPECTED_SCENARIOS,
            "confidence_level": 0.95,
            "alpha": ALPHA,
            "paired_unit": "locked scenario after averaging repeated seeds within method",
            "holm_family": "separate per metric; 3 baselines x 5 N = 15 hypotheses",
            "metrics": METRICS,
        }, indent=2),
        encoding="utf-8",
    )
    print((args.output_dir / "FINAL_NUMBERS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
