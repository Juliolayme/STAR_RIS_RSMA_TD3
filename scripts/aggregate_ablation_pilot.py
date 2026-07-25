from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = ("v1", "norm_only", "v2_fixed_budget", "v2_full")
N_VALUES = (16, 32, 64, 96, 128)
METRICS = (
    "test_mean_sum_rate",
    "test_mean_reward",
    "test_mean_qos_fraction",
    "test_mean_all_qos",
    "test_mean_violation",
)
COMPARISONS = (
    ("v1", "norm_only", "normalization_effect"),
    ("norm_only", "v2_fixed_budget", "algorithmic_v2_effect_fixed_compute"),
    ("v2_fixed_budget", "v2_full", "extra_compute_effect"),
)


def collect(root: Path) -> pd.DataFrame:
    paths = sorted(root.rglob("JOB_SUMMARY.csv"))
    if not paths:
        raise SystemExit(f"No JOB_SUMMARY.csv files found below {root}")
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def expected_rows(seeds: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"arm": arm, "n_ris": n_ris, "seed": seed}
            for arm in ARMS
            for n_ris in N_VALUES
            for seed in seeds
        ]
    )


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    complete = frame[(frame["complete"] == True) & (frame["finite"] == True)].copy()  # noqa: E712
    for (arm, n_ris), group in complete.groupby(["arm", "n_ris"], sort=True):
        row: dict[str, object] = {
            "arm": arm,
            "n_ris": int(n_ris),
            "seed_count": int(group["seed"].nunique()),
        }
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else None
            row[f"{metric}_min"] = float(values.min()) if len(values) else None
            row[f"{metric}_max"] = float(values.max()) if len(values) else None
        for field in (
            "hidden_dim",
            "train_steps_config",
            "warmup_steps",
            "replay_size",
            "observation_normalization",
            "qos_penalty_linear",
            "qos_penalty_quadratic",
            "td3_layer_norm",
            "td3_critic_loss",
        ):
            if field in group.columns:
                unique = group[field].dropna().astype(str).unique()
                row[field] = unique[0] if len(unique) == 1 else "MIXED"
        rows.append(row)
    return pd.DataFrame(rows)


def paired_effects(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    valid = frame[(frame["complete"] == True) & (frame["finite"] == True)].copy()  # noqa: E712
    for n_ris in N_VALUES:
        subset = valid[valid["n_ris"] == n_ris]
        for left, right, label in COMPARISONS:
            left_frame = subset[subset["arm"] == left].set_index("seed")
            right_frame = subset[subset["arm"] == right].set_index("seed")
            seeds = sorted(set(left_frame.index).intersection(right_frame.index))
            for metric in METRICS:
                if metric not in left_frame.columns or metric not in right_frame.columns:
                    continue
                differences = []
                for seed in seeds:
                    a = pd.to_numeric(pd.Series([left_frame.loc[seed, metric]]), errors="coerce").iloc[0]
                    b = pd.to_numeric(pd.Series([right_frame.loc[seed, metric]]), errors="coerce").iloc[0]
                    if np.isfinite(a) and np.isfinite(b):
                        differences.append(float(b - a))
                rows.append({
                    "comparison": label,
                    "arm_left": left,
                    "arm_right": right,
                    "n_ris": n_ris,
                    "metric": metric,
                    "paired_seed_count": len(differences),
                    "mean_delta_right_minus_left": float(np.mean(differences)) if differences else None,
                    "std_delta": float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0 if len(differences) == 1 else None,
                    "min_delta": float(np.min(differences)) if differences else None,
                    "max_delta": float(np.max(differences)) if differences else None,
                })
    return pd.DataFrame(rows)


def fmt(value: object, digits: int = 5) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—" if value is None or pd.isna(value) else str(value)
    return "—" if not np.isfinite(number) else f"{number:.{digits}f}"


def write_report(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    seeds: list[int],
    output: Path,
) -> list[str]:
    findings: list[str] = []
    expected = len(ARMS) * len(N_VALUES) * len(seeds)
    complete_count = int(((runs["complete"] == True) & (runs["finite"] == True)).sum())  # noqa: E712
    if complete_count != expected:
        findings.append(f"Only {complete_count}/{expected} expected arm/N/seed runs are complete and finite.")

    for n_ris in N_VALUES:
        checks = runs[(runs["n_ris"] == n_ris) & (runs["complete"] == True)]["test_bank_checksum"].dropna().astype(str).unique()  # noqa: E712
        if len(checks) > 1:
            findings.append(f"N={n_ris}: test-bank checksum differs across arms or seeds.")

    bad = runs[(runs["complete"] != True) | (runs["finite"] != True) | (runs["issue_count"] > 0)]  # noqa: E712
    for _, row in bad.iterrows():
        findings.append(f"{row.arm} N={int(row.n_ris)} seed={int(row.seed)}: {row.issues}")

    lines = [
        "# TD3 20-job ablation audit",
        "",
        f"- Expected compact runs: **{expected}**",
        f"- Complete and finite: **{complete_count}/{expected}**",
        f"- Seeds: **{', '.join(map(str, seeds))}**",
        "",
        "## Arm-by-N test summary",
        "",
        "| Arm | N | Seeds | Sum-rate | QoS fraction | All-QoS | Violation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.sort_values(["n_ris", "arm"]).iterrows():
        lines.append(
            f"| {row.arm} | {int(row.n_ris)} | {int(row.seed_count)} | "
            f"{fmt(row.get('test_mean_sum_rate_mean'))} | "
            f"{fmt(row.get('test_mean_qos_fraction_mean'))} | "
            f"{fmt(row.get('test_mean_all_qos_mean'))} | "
            f"{fmt(row.get('test_mean_violation_mean'))} |"
        )

    lines.extend([
        "",
        "## Paired causal contrasts",
        "",
        "Positive deltas mean the right-hand arm is larger than the left-hand arm on the same seed.",
        "",
        "| Contrast | N | Metric | Paired seeds | Mean delta |",
        "|---|---:|---|---:|---:|",
    ])
    selected_metrics = {"test_mean_sum_rate", "test_mean_qos_fraction", "test_mean_all_qos", "test_mean_violation"}
    for _, row in effects[effects["metric"].isin(selected_metrics)].sort_values(["n_ris", "comparison", "metric"]).iterrows():
        lines.append(
            f"| {row.comparison} | {int(row.n_ris)} | {row.metric.replace('test_mean_', '')} | "
            f"{int(row.paired_seed_count)} | {fmt(row.mean_delta_right_minus_left)} |"
        )

    lines.extend(["", "## Findings", ""])
    if findings:
        lines.extend([f"- {item}" for item in findings[:100]])
    else:
        lines.append("- No structural, checksum, or core numerical issues detected.")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw = collect(args.root)
    expected = expected_rows(args.seeds)
    merged = expected.merge(raw, on=["arm", "n_ris", "seed"], how="left", validate="one_to_one")
    merged["complete"] = merged["complete"].fillna(False).astype(bool)
    merged["finite"] = merged["finite"].fillna(False).astype(bool)
    merged["issues"] = merged["issues"].fillna("missing job artifact")
    merged["issue_count"] = pd.to_numeric(merged["issue_count"], errors="coerce").fillna(1).astype(int)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(merged)
    effects = paired_effects(merged)
    merged.to_csv(args.output_dir / "ABLATION_RUNS.csv", index=False)
    summary.to_csv(args.output_dir / "ARM_N_SUMMARY.csv", index=False)
    effects.to_csv(args.output_dir / "PAIRED_EFFECTS.csv", index=False)
    findings = write_report(merged, summary, effects, args.seeds, args.output_dir / "AUDIT_REPORT.md")
    (args.output_dir / "AUDIT_REPORT.json").write_text(
        json.dumps({
            "arms": ARMS,
            "n_values": N_VALUES,
            "seeds": args.seeds,
            "expected_runs": len(expected),
            "complete_finite_runs": int(((merged.complete == True) & (merged.finite == True)).sum()),  # noqa: E712
            "findings": findings,
        }, indent=2),
        encoding="utf-8",
    )
    print((args.output_dir / "AUDIT_REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
