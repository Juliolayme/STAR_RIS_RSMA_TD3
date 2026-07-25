from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from star_ris_rsma.result_validation import CORE_METRICS, replace_core_metrics


N_VALUES = (16, 32, 64, 96, 128)
SEEDS = tuple(range(8))
PROPOSED = "td3_v3_constrained"
DRL_BASELINES = ("td3_v2_fixed", "td3", "ddpg", "ppo")
TRADITIONAL_BASELINES = ("ao_sca", "ao_grid", "analytical_ris")
METRICS = tuple(CORE_METRICS)
EXPECTED_SCENARIOS = 1000
ALPHA = 0.05


def _infer_n(path: Path) -> int:
    match = re.search(r"N(16|32|64|96|128)", path.as_posix())
    if match is None:
        raise RuntimeError(f"Cannot infer n_ris from {path}")
    return int(match.group(1))


def _numeric_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("n_ris", "seed", "scenario"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="raise").astype(int)
    return result


def _read_single(root: Path, name: str) -> Path:
    paths = sorted(root.rglob(name))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {name} below {root}, found {len(paths)}")
    return paths[0]


def load_proposed(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_paths = sorted(root.rglob("test.csv"))
    manifest_paths = sorted(root.rglob("manifest.json"))
    expected_runs = len(N_VALUES) * len(SEEDS)
    if len(test_paths) != expected_runs or len(manifest_paths) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} tests/manifests, found "
            f"{len(test_paths)}/{len(manifest_paths)}"
        )

    manifest_rows: list[dict[str, object]] = []
    for path in manifest_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {}) or {}
        if config.get("action_parameterization") != "physical_v3":
            raise RuntimeError(f"Non-physical_v3 manifest: {path}")
        if config.get("qos_dual_enabled") is not True:
            raise RuntimeError(f"Adaptive QoS dual is not enabled: {path}")
        n_ris = int(config.get("n_ris", _infer_n(path)))
        seed = int(payload.get("seed", -1))
        best = payload.get("best_validation", {}) or {}
        dual = payload.get("qos_dual", {}) or {}
        manifest_rows.append({
            "n_ris": n_ris,
            "seed": seed,
            "config_hash": payload.get("config_hash"),
            "training_protocol": payload.get("training_protocol"),
            "best_step": best.get("eval_step"),
            "best_feasible": best.get("feasible"),
            "best_constraint_gap": best.get("constraint_gap"),
            "best_sum_rate": best.get("mean_sum_rate"),
            "best_qos_fraction": best.get("mean_qos_fraction"),
            "best_all_qos": best.get("mean_all_qos"),
            "best_violation": best.get("mean_violation"),
            "final_qos_dual": dual.get("value"),
            "qos_dual_updates": dual.get("updates"),
            "source_path": path.as_posix(),
        })

    frames: list[pd.DataFrame] = []
    for path in test_paths:
        frame = pd.read_csv(path)
        frame = replace_core_metrics(
            frame,
            context=f"proposed {path}",
            require_finite=True,
        )
        frame = _numeric_keys(frame)
        frame["n_ris"] = _infer_n(path)
        frame["method"] = PROPOSED
        frame["source_path"] = path.as_posix()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(manifest_rows)


def load_legacy(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(_read_single(root, "MERGED_RAW_TEST.csv"))
    frame = frame[frame["method"].isin(DRL_BASELINES)].copy()
    frame = replace_core_metrics(frame, context="legacy DRL", require_finite=True)
    return _numeric_keys(frame)


def load_traditional(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(_read_single(root, "TRADITIONAL_RAW_TEST.csv"))
    aliases = {
        "ao_sca_proximal_physical": "ao_sca",
        "ao_grid_coordinate_codebook": "ao_grid",
        "analytical_ris_equal_allocation": "analytical_ris",
    }
    frame["method"] = (
        frame["method"].astype(str).str.strip().str.lower().replace(aliases)
    )
    frame = frame[frame["method"].isin(TRADITIONAL_BASELINES)].copy()
    frame = replace_core_metrics(frame, context="traditional", require_finite=True)
    frame = _numeric_keys(frame)
    runtime_path = _read_single(root, "TRADITIONAL_RUNTIME.csv")
    runtime = pd.read_csv(runtime_path)
    return frame, runtime


def validate(
    proposed: pd.DataFrame,
    manifests: pd.DataFrame,
    legacy: pd.DataFrame,
    traditional: pd.DataFrame,
) -> pd.DataFrame:
    errors: list[str] = []
    records: list[dict[str, object]] = []
    if proposed.duplicated(["method", "n_ris", "seed", "scenario"]).any():
        errors.append("Duplicate proposed method/N/seed/scenario rows")
    if manifests.duplicated(["n_ris", "seed"]).any():
        errors.append("Duplicate proposed manifests")

    for n_ris in N_VALUES:
        proposed_n = proposed[proposed.n_ris == n_ris]
        proposed_checks = proposed_n["bank_checksum"].dropna().astype(str).unique()
        legacy_checks = legacy[legacy.n_ris == n_ris]["bank_checksum"].dropna().astype(str).unique()
        traditional_checks = traditional[traditional.n_ris == n_ris]["bank_checksum"].dropna().astype(str).unique()
        if len(proposed_checks) != 1:
            errors.append(f"N={n_ris}: proposed checksum count={len(proposed_checks)}")
        if len(legacy_checks) != 1 or len(traditional_checks) != 1:
            errors.append(f"N={n_ris}: baseline checksum count mismatch")
        if (
            len(proposed_checks) == 1
            and len(legacy_checks) == 1
            and len(traditional_checks) == 1
            and not (proposed_checks[0] == legacy_checks[0] == traditional_checks[0])
        ):
            errors.append(f"N={n_ris}: locked test-bank checksum mismatch")

        for seed in SEEDS:
            group = proposed_n[proposed_n.seed == seed]
            scenarios = sorted(group.scenario.unique())
            complete = (
                len(group) == EXPECTED_SCENARIOS
                and len(scenarios) == EXPECTED_SCENARIOS
                and scenarios[0] == 0
                and scenarios[-1] == EXPECTED_SCENARIOS - 1
            ) if scenarios else False
            finite = bool(np.isfinite(group[list(METRICS)].to_numpy(dtype=np.float64)).all()) if len(group) else False
            manifest_count = len(manifests[(manifests.n_ris == n_ris) & (manifests.seed == seed)])
            records.append({
                "method": PROPOSED,
                "n_ris": n_ris,
                "seed": seed,
                "rows": len(group),
                "scenario_count": len(scenarios),
                "finite": finite,
                "manifest_count": manifest_count,
                "complete": complete and finite and manifest_count == 1,
            })
            if not complete or not finite or manifest_count != 1:
                errors.append(
                    f"N={n_ris} seed={seed}: rows/scenarios/finite/manifest="
                    f"{len(group)}/{len(scenarios)}/{finite}/{manifest_count}"
                )

    expected_legacy = len(DRL_BASELINES) * len(N_VALUES) * len(SEEDS) * EXPECTED_SCENARIOS
    expected_traditional = len(TRADITIONAL_BASELINES) * len(N_VALUES) * EXPECTED_SCENARIOS
    if len(legacy) != expected_legacy:
        errors.append(f"Legacy rows={len(legacy)}, expected={expected_legacy}")
    if len(traditional) != expected_traditional:
        errors.append(f"Traditional rows={len(traditional)}, expected={expected_traditional}")
    if errors:
        raise RuntimeError("Final coverage validation failed:\n" + "\n".join(errors[:100]))
    return pd.DataFrame(records)


def t_interval(values: np.ndarray) -> tuple[float, float, float, float, int]:
    values = np.asarray(values, dtype=np.float64)
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


def seed_statistics(proposed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_level = (
        proposed.groupby(["method", "n_ris", "seed"], as_index=False)[list(METRICS)]
        .mean()
        .sort_values(["n_ris", "seed"])
    )
    rows: list[dict[str, object]] = []
    for n_ris in N_VALUES:
        group = seed_level[seed_level.n_ris == n_ris]
        for metric in METRICS:
            mean, std, low, high, count = t_interval(group[metric].to_numpy())
            rows.append({
                "method": PROPOSED,
                "n_ris": n_ris,
                "metric": metric,
                "seed_count": count,
                "mean": mean,
                "std": std,
                "ci95_low": low,
                "ci95_high": high,
            })
    return seed_level, pd.DataFrame(rows)


def holm_adjust(values: pd.Series) -> np.ndarray:
    p = values.astype(float).to_numpy()
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    total = len(p)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def paired_statistics(
    proposed: pd.DataFrame,
    legacy: pd.DataFrame,
    traditional: pd.DataFrame,
) -> pd.DataFrame:
    proposed_scenario = proposed.groupby(["n_ris", "scenario"], as_index=False)[list(METRICS)].mean()
    legacy_scenario = legacy.groupby(["method", "n_ris", "scenario"], as_index=False)[list(METRICS)].mean()
    traditional_scenario = traditional[["method", "n_ris", "scenario", *METRICS]].copy()
    comparison = pd.concat([legacy_scenario, traditional_scenario], ignore_index=True)

    rows: list[dict[str, object]] = []
    for n_ris in N_VALUES:
        proposed_n = proposed_scenario[proposed_scenario.n_ris == n_ris].set_index("scenario")
        for baseline in (*DRL_BASELINES, *TRADITIONAL_BASELINES):
            other = comparison[(comparison.method == baseline) & (comparison.n_ris == n_ris)].set_index("scenario")
            common = proposed_n.index.intersection(other.index)
            if len(common) != EXPECTED_SCENARIOS:
                raise RuntimeError(f"Paired coverage mismatch for {baseline}, N={n_ris}")
            family = "drl" if baseline in DRL_BASELINES else "traditional"
            for metric in METRICS:
                delta = (
                    proposed_n.loc[common, metric].to_numpy(dtype=np.float64)
                    - other.loc[common, metric].to_numpy(dtype=np.float64)
                )
                mean, std, low, high, count = t_interval(delta)
                t_result = stats.ttest_1samp(delta, 0.0)
                if np.allclose(delta, 0.0, atol=1e-15, rtol=0.0):
                    wilcoxon_p = 1.0
                else:
                    wilcoxon_p = float(stats.wilcoxon(delta).pvalue)
                rows.append({
                    "proposed": PROPOSED,
                    "baseline": baseline,
                    "family": family,
                    "n_ris": n_ris,
                    "metric": metric,
                    "paired_scenarios": count,
                    "mean_delta_proposed_minus_baseline": mean,
                    "delta_std": std,
                    "delta_ci95_low": low,
                    "delta_ci95_high": high,
                    "paired_t_p_raw": float(t_result.pvalue),
                    "wilcoxon_p_raw": wilcoxon_p,
                    "cohen_dz": mean / std if std > 0 else 0.0,
                })
    result = pd.DataFrame(rows)
    result["paired_t_p_holm"] = np.nan
    result["wilcoxon_p_holm"] = np.nan
    result["holm_family_size"] = 0
    for (family, metric), indexes in result.groupby(["family", "metric"]).groups.items():
        idx = list(indexes)
        result.loc[idx, "paired_t_p_holm"] = holm_adjust(result.loc[idx, "paired_t_p_raw"])
        result.loc[idx, "wilcoxon_p_holm"] = holm_adjust(result.loc[idx, "wilcoxon_p_raw"])
        result.loc[idx, "holm_family_size"] = len(idx)
    result["paired_t_reject_holm_0_05"] = result.paired_t_p_holm < ALPHA
    result["wilcoxon_reject_holm_0_05"] = result.wilcoxon_p_holm < ALPHA
    return result


def fmt(value: object, digits: int = 5) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(number) else f"{number:.{digits}f}"


def write_report(
    ci: pd.DataFrame,
    manifests: pd.DataFrame,
    paired: pd.DataFrame,
    output: Path,
) -> None:
    lines = [
        "# Final TD3 v3 constrained physical-action statistics",
        "",
        "- Proposed method: **TD3 v3 constrained physical action**",
        "- Training seeds: **0-7**",
        "- Locked test scenarios per N: **1,000**",
        "- Adaptive QoS dual penalty: **enabled**",
        "- AO-SCA is a local proximal method, not a global optimum or upper bound.",
        "",
        "## Mean and 95% CI across training seeds",
        "",
        "| N | Sum-rate | QoS fraction | All-QoS | Violation | Feasible validation checkpoints |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n_ris in N_VALUES:
        subset = ci[ci.n_ris == n_ris].set_index("metric")
        feasible_count = int(
            pd.to_numeric(
                manifests[manifests.n_ris == n_ris]["best_feasible"],
                errors="coerce",
            ).fillna(0).astype(bool).sum()
        )
        cells = []
        for metric in METRICS:
            row = subset.loc[metric]
            cells.append(
                f"{fmt(row['mean'])} [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]"
            )
        lines.append(f"| {n_ris} | " + " | ".join(cells) + f" | {feasible_count}/8 |")

    lines.extend([
        "",
        "## Paired comparisons on locked scenarios",
        "",
        "Positive delta favors v3 for sum-rate/QoS metrics; negative delta favors v3 for violation.",
        "Holm correction is separate for DRL and traditional families within each metric.",
        "",
        "| Family | Baseline | N | Metric | Mean delta | 95% CI | t-Holm p | Wilcoxon-Holm p | dz |",
        "|---|---|---:|---|---:|---|---:|---:|---:|",
    ])
    for row in paired.sort_values(["family", "baseline", "n_ris", "metric"]).itertuples(index=False):
        lines.append(
            f"| {row.family} | {row.baseline} | {int(row.n_ris)} | {row.metric} | "
            f"{fmt(row.mean_delta_proposed_minus_baseline)} | "
            f"[{fmt(row.delta_ci95_low)}, {fmt(row.delta_ci95_high)}] | "
            f"{fmt(row.paired_t_p_holm, 6)} | {fmt(row.wilcoxon_p_holm, 6)} | "
            f"{fmt(row.cohen_dz)} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--traditional-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    proposed, manifests = load_proposed(args.proposed_root)
    legacy = load_legacy(args.legacy_root)
    traditional, runtime = load_traditional(args.traditional_root)
    coverage = validate(proposed, manifests, legacy, traditional)
    seed_level, ci = seed_statistics(proposed)
    paired = paired_statistics(proposed, legacy, traditional)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compact = proposed[[
        "method", "n_ris", "seed", "scenario", *METRICS,
        "bank_checksum", "config_hash", "source_path",
    ]].copy()
    compact.to_csv(args.output_dir / "V3_MERGED_RAW_TEST.csv", index=False)
    manifests.sort_values(["n_ris", "seed"]).to_csv(
        args.output_dir / "V3_MANIFEST_SUMMARY.csv", index=False
    )
    coverage.to_csv(args.output_dir / "V3_COVERAGE.csv", index=False)
    seed_level.to_csv(args.output_dir / "V3_SEED_LEVEL_MEANS.csv", index=False)
    ci.to_csv(args.output_dir / "V3_CI95_SUMMARY.csv", index=False)
    paired.to_csv(args.output_dir / "V3_PAIRED_HOLM.csv", index=False)
    runtime.to_csv(args.output_dir / "TRADITIONAL_RUNTIME.csv", index=False)
    write_report(ci, manifests, paired, args.output_dir / "FINAL_V3_NUMBERS.md")
    (args.output_dir / "V3_PROTOCOL.json").write_text(
        json.dumps({
            "proposed": PROPOSED,
            "n_values": N_VALUES,
            "seeds": SEEDS,
            "test_scenarios_per_n": EXPECTED_SCENARIOS,
            "drl_baselines": DRL_BASELINES,
            "traditional_baselines": TRADITIONAL_BASELINES,
            "paired_unit": "locked scenario after averaging eight policies",
            "holm_families": {
                "drl_per_metric": len(DRL_BASELINES) * len(N_VALUES),
                "traditional_per_metric": len(TRADITIONAL_BASELINES) * len(N_VALUES),
            },
            "reward_compared": False,
            "ao_sca_claim_boundary": "local method; not global optimum or upper bound",
        }, indent=2),
        encoding="utf-8",
    )
    print((args.output_dir / "FINAL_V3_NUMBERS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
