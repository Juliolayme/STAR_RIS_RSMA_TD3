from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

N_VALUES = (16, 32, 64, 96, 128)
SOLVERS = ("ao_sca", "ao_grid", "analytical_ris")
PROPOSED = "td3_v2_fixed"
METRICS = ("sum_rate", "qos_fraction", "all_qos", "violation")
EXPECTED_SCENARIOS = 1000
ALPHA = 0.05


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float)
    return series.astype(str).str.strip().str.lower().map(
        {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
    ).astype(float)


def normalize_method(value: object) -> str:
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "td3_v2_fixed": PROPOSED,
        "v2_fixed_budget": PROPOSED,
        "ao_sca_proximal_physical": "ao_sca",
        "ao_grid_coordinate_codebook": "ao_grid",
        "analytical_ris_equal_allocation": "analytical_ris",
    }
    return aliases.get(text, text)


def load_drl(root: Path) -> pd.DataFrame:
    candidates = sorted(root.rglob("MERGED_RAW_TEST.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one MERGED_RAW_TEST.csv below {root}, found {len(candidates)}")
    frame = pd.read_csv(candidates[0])
    if "method" not in frame.columns:
        raise RuntimeError("DRL merged file has no method column")
    frame["method"] = frame["method"].map(normalize_method)
    frame = frame[frame.method == PROPOSED].copy()
    required = {"method", "n_ris", "seed", "scenario", "bank_checksum", *METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"DRL file missing columns: {sorted(missing)}")
    frame["n_ris"] = pd.to_numeric(frame["n_ris"], errors="raise").astype(int)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["scenario"] = pd.to_numeric(frame["scenario"], errors="raise").astype(int)
    for metric in ("sum_rate", "qos_fraction", "violation"):
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame["all_qos"] = parse_bool(frame["all_qos"])
    return frame


def load_solvers(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.rglob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        required = {"method", "scenario", "bank_checksum", *METRICS}
        if not required.issubset(frame.columns):
            continue
        methods = {normalize_method(v) for v in frame.method.dropna().unique()}
        methods &= set(SOLVERS)
        if len(methods) != 1:
            continue
        method = next(iter(methods))
        if "n_ris" in frame.columns:
            n_values = pd.to_numeric(frame.n_ris, errors="coerce").dropna().astype(int).unique()
            if len(n_values) != 1:
                raise RuntimeError(f"Cannot infer N from {path}")
            n_ris = int(n_values[0])
        else:
            import re
            matches = re.findall(r"N(16|32|64|96|128)", path.as_posix(), flags=re.IGNORECASE)
            if not matches:
                raise RuntimeError(f"Cannot infer N from path {path}")
            n_ris = int(matches[-1])
        selected = frame.copy()
        selected["method"] = method
        selected["n_ris"] = n_ris
        selected["scenario"] = pd.to_numeric(selected["scenario"], errors="raise").astype(int)
        selected["seed"] = pd.to_numeric(selected.get("seed", 10000), errors="coerce").fillna(10000).astype(int)
        for metric in ("sum_rate", "qos_fraction", "violation"):
            selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
        selected["all_qos"] = parse_bool(selected["all_qos"])
        if "solve_ms" not in selected.columns:
            selected["solve_ms"] = np.nan
        selected["source_path"] = path.as_posix()
        frames.append(selected)
    if not frames:
        raise RuntimeError(f"No solver CSVs found below {root}")
    return pd.concat(frames, ignore_index=True)


def validate(drl: pd.DataFrame, solvers: pd.DataFrame) -> pd.DataFrame:
    issues: list[str] = []
    records: list[dict[str, object]] = []
    expected_drl = 5 * 8 * EXPECTED_SCENARIOS
    if len(drl) != expected_drl:
        issues.append(f"DRL rows={len(drl)}, expected={expected_drl}")
    drl_keys = ["method", "n_ris", "seed", "scenario"]
    if drl.duplicated(drl_keys).any():
        issues.append("Duplicate DRL method/N/seed/scenario rows")
    if not np.isfinite(drl[list(METRICS)].to_numpy(dtype=float)).all():
        issues.append("Non-finite DRL core metric")

    solver_keys = ["method", "n_ris", "scenario"]
    duplicate_groups = solvers.groupby(solver_keys, sort=False)
    dedup_rows: list[pd.DataFrame] = []
    for key, group in duplicate_groups:
        if len(group) > 1:
            reference = group.iloc[0][list(METRICS)].astype(float).to_numpy()
            for _, candidate in group.iloc[1:].iterrows():
                values = candidate[list(METRICS)].astype(float).to_numpy()
                if not np.allclose(reference, values, rtol=1e-10, atol=1e-12):
                    issues.append(f"Conflicting duplicate solver row {key}")
        dedup_rows.append(group.iloc[[0]])
    solvers_clean = pd.concat(dedup_rows, ignore_index=True)

    for n_ris in N_VALUES:
        drl_n = drl[drl.n_ris == n_ris]
        drl_checks = drl_n.bank_checksum.dropna().astype(str).unique()
        if len(drl_checks) != 1:
            issues.append(f"DRL N={n_ris}: checksum count={len(drl_checks)}")
        for method in SOLVERS:
            group = solvers_clean[(solvers_clean.method == method) & (solvers_clean.n_ris == n_ris)]
            scenarios = sorted(group.scenario.unique())
            finite = bool(np.isfinite(group[list(METRICS)].to_numpy(dtype=float)).all()) if len(group) else False
            checks = group.bank_checksum.dropna().astype(str).unique() if len(group) else []
            complete = (
                len(group) == EXPECTED_SCENARIOS
                and len(scenarios) == EXPECTED_SCENARIOS
                and scenarios[0] == 0
                and scenarios[-1] == EXPECTED_SCENARIOS - 1
            ) if scenarios else False
            same_bank = len(checks) == 1 and len(drl_checks) == 1 and checks[0] == drl_checks[0]
            records.append({
                "method": method,
                "n_ris": n_ris,
                "rows": len(group),
                "scenario_count": len(scenarios),
                "finite": finite,
                "bank_checksum_count": len(checks),
                "same_bank_as_drl": same_bank,
                "complete": complete and finite and same_bank,
            })
            if not complete:
                issues.append(f"{method} N={n_ris}: rows/scenarios={len(group)}/{len(scenarios)}")
            if not finite:
                issues.append(f"{method} N={n_ris}: non-finite core metric")
            if not same_bank:
                issues.append(f"{method} N={n_ris}: test-bank checksum mismatch")
    if issues:
        raise RuntimeError("Coverage validation failed:\n" + "\n".join(issues[:100]))
    return pd.DataFrame(records), solvers_clean


def t_interval(values: np.ndarray) -> tuple[float, float, float, float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return math.nan, math.nan, math.nan, math.nan, 0
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    if n == 1:
        return mean, std, mean, mean, n
    half = float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n))
    return mean, std, mean - half, mean + half, n


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


def summarize(drl: pd.DataFrame, solvers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    drl_scenario = (
        drl.groupby(["n_ris", "scenario"], as_index=False)[list(METRICS)]
        .mean()
        .assign(method=PROPOSED)
    )
    solver_scenario = solvers[["method", "n_ris", "scenario", *METRICS, "solve_ms"]].copy()

    summary_rows: list[dict[str, object]] = []
    combined = pd.concat([
        drl_scenario.assign(solve_ms=np.nan),
        solver_scenario,
    ], ignore_index=True)
    for (method, n_ris), group in combined.groupby(["method", "n_ris"], sort=True):
        for metric in METRICS:
            mean, std, low, high, n = t_interval(group[metric].to_numpy(dtype=float))
            summary_rows.append({
                "method": method,
                "n_ris": int(n_ris),
                "metric": metric,
                "scenario_count": n,
                "mean": mean,
                "scenario_std": std,
                "scenario_ci95_low": low,
                "scenario_ci95_high": high,
            })
    summary = pd.DataFrame(summary_rows)

    runtime_rows: list[dict[str, object]] = []
    for (method, n_ris), group in solvers.groupby(["method", "n_ris"], sort=True):
        values = pd.to_numeric(group.solve_ms, errors="coerce").dropna().to_numpy(dtype=float)
        runtime_rows.append({
            "method": method,
            "n_ris": int(n_ris),
            "scenario_count": len(values),
            "solve_ms_mean": float(np.mean(values)) if len(values) else math.nan,
            "solve_ms_median": float(np.median(values)) if len(values) else math.nan,
            "solve_ms_p95": float(np.quantile(values, 0.95)) if len(values) else math.nan,
            "solve_ms_max": float(np.max(values)) if len(values) else math.nan,
        })
    runtime = pd.DataFrame(runtime_rows)

    paired_rows: list[dict[str, object]] = []
    for n_ris in N_VALUES:
        proposed = drl_scenario[drl_scenario.n_ris == n_ris].set_index("scenario")
        for baseline in SOLVERS:
            comparison = solver_scenario[
                (solver_scenario.method == baseline) & (solver_scenario.n_ris == n_ris)
            ].set_index("scenario")
            common = proposed.index.intersection(comparison.index)
            if len(common) != EXPECTED_SCENARIOS:
                raise RuntimeError(f"Paired coverage mismatch for {baseline}, N={n_ris}: {len(common)}")
            for metric in METRICS:
                x = proposed.loc[common, metric].to_numpy(dtype=float)
                y = comparison.loc[common, metric].to_numpy(dtype=float)
                diff = x - y
                mean, std, low, high, n = t_interval(diff)
                t_result = stats.ttest_rel(x, y, nan_policy="raise")
                if np.allclose(diff, 0.0, atol=1e-15, rtol=0.0):
                    w_stat, w_p = 0.0, 1.0
                else:
                    w = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox", method="auto")
                    w_stat, w_p = float(w.statistic), float(w.pvalue)
                dz = mean / std if std > 0 else (math.copysign(math.inf, mean) if mean != 0 else 0.0)
                paired_rows.append({
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
                    "wilcoxon_statistic": w_stat,
                    "wilcoxon_p_raw": w_p,
                    "cohen_dz": dz,
                })
    paired = pd.DataFrame(paired_rows)
    paired["holm_family"] = paired.metric.map(lambda m: f"traditional_{m}")
    paired["holm_family_size"] = 15
    paired["paired_t_p_holm"] = paired.groupby("metric", group_keys=False)["paired_t_p_raw"].apply(holm_adjust)
    paired["wilcoxon_p_holm"] = paired.groupby("metric", group_keys=False)["wilcoxon_p_raw"].apply(holm_adjust)
    paired["paired_t_reject_holm_0_05"] = paired.paired_t_p_holm < ALPHA
    paired["wilcoxon_reject_holm_0_05"] = paired.wilcoxon_p_holm < ALPHA
    return summary, runtime, paired


def format_num(value: object, digits: int = 5) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(number) else f"{number:.{digits}f}"


def write_report(summary: pd.DataFrame, runtime: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    lines = [
        "# TD3 v2 fixed versus traditional optimization baselines",
        "",
        "AO-SCA is the primary conventional baseline. AO-Grid and AnalyticalRIS are supplementary baselines.",
        "AO-SCA is a local proximal finite-difference method, not a global optimum or upper bound.",
        "DRL values are averaged over eight trained policies for each locked test scenario before paired comparison.",
        "Traditional solvers are deterministic and are evaluated once per locked scenario.",
        "",
        "## Mean performance on 1,000 locked test scenarios",
        "",
        "| Method | N | Sum-rate | QoS fraction | All-QoS | Violation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    pivot = summary.pivot_table(index=["method", "n_ris"], columns="metric", values="mean").reset_index()
    order = {PROPOSED: 0, "ao_sca": 1, "ao_grid": 2, "analytical_ris": 3}
    pivot["_order"] = pivot.method.map(order)
    for _, row in pivot.sort_values(["n_ris", "_order"]).iterrows():
        lines.append(
            f"| {row.method} | {int(row.n_ris)} | {format_num(row.sum_rate)} | "
            f"{format_num(row.qos_fraction)} | {format_num(row.all_qos)} | {format_num(row.violation)} |"
        )
    lines.extend([
        "",
        "## Paired differences: TD3 v2 fixed minus baseline",
        "",
        "Holm correction is applied separately to four metric families; each family contains 15 hypotheses (3 baselines × 5 N values).",
        "",
        "| Baseline | N | Metric | Mean delta | 95% CI | t-Holm p | Wilcoxon-Holm p | Cohen dz |",
        "|---|---:|---|---:|---|---:|---:|---:|",
    ])
    for _, row in paired.sort_values(["baseline", "n_ris", "metric"]).iterrows():
        lines.append(
            f"| {row.baseline} | {int(row.n_ris)} | {row.metric} | "
            f"{format_num(row.mean_delta_proposed_minus_baseline)} | "
            f"[{format_num(row.delta_ci95_low)}, {format_num(row.delta_ci95_high)}] | "
            f"{format_num(row.paired_t_p_holm, 6)} | {format_num(row.wilcoxon_p_holm, 6)} | "
            f"{format_num(row.cohen_dz)} |"
        )
    lines.extend([
        "",
        "## Conventional solver latency",
        "",
        "The table reports solver wall-clock time only. DRL inference latency is not present in the retained final artifact and is therefore not compared here.",
        "",
        "| Method | N | Mean ms | Median ms | P95 ms | Max ms |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for _, row in runtime.sort_values(["method", "n_ris"]).iterrows():
        lines.append(
            f"| {row.method} | {int(row.n_ris)} | {format_num(row.solve_ms_mean, 3)} | "
            f"{format_num(row.solve_ms_median, 3)} | {format_num(row.solve_ms_p95, 3)} | "
            f"{format_num(row.solve_ms_max, 3)} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drl-root", type=Path, required=True)
    parser.add_argument("--solver-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    drl = load_drl(args.drl_root)
    solver_raw = load_solvers(args.solver_root)
    coverage, solvers = validate(drl, solver_raw)
    summary, runtime, paired = summarize(drl, solvers)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.output_dir / "TRADITIONAL_COVERAGE.csv", index=False)
    solvers.sort_values(["method", "n_ris", "scenario"]).to_csv(
        args.output_dir / "TRADITIONAL_RAW_TEST.csv", index=False
    )
    summary.to_csv(args.output_dir / "DRL_TRADITIONAL_SUMMARY.csv", index=False)
    runtime.to_csv(args.output_dir / "TRADITIONAL_RUNTIME.csv", index=False)
    paired.to_csv(args.output_dir / "DRL_TRADITIONAL_PAIRED_HOLM.csv", index=False)
    write_report(summary, runtime, paired, args.output_dir / "DRL_VS_TRADITIONAL.md")
    (args.output_dir / "TRADITIONAL_PROTOCOL.json").write_text(json.dumps({
        "proposed": PROPOSED,
        "traditional_baselines": SOLVERS,
        "primary_traditional_baseline": "ao_sca",
        "n_values": N_VALUES,
        "training_seeds_averaged_for_drl": list(range(8)),
        "test_scenarios_per_n": EXPECTED_SCENARIOS,
        "paired_unit": "locked test scenario after averaging DRL over eight training seeds",
        "holm_families": {metric: 15 for metric in METRICS},
        "reward_compared": False,
        "ao_sca_claim_boundary": "local proximal finite-difference method; not global optimum or upper bound",
    }, indent=2), encoding="utf-8")
    print((args.output_dir / "DRL_VS_TRADITIONAL.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
