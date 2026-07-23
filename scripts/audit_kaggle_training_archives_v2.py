from __future__ import annotations

import argparse
import io
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

METHODS = ("td3", "ddpg", "ppo")
N_VALUES = (16, 32, 64, 96, 128)
SEEDS = tuple(range(8))
CORE_NUMERIC = ("reward", "sum_rate", "qos_fraction", "all_qos", "violation")


def infer_method(text: str) -> str | None:
    low = text.lower()
    for method in METHODS:
        if re.search(rf"(^|[/_\-]){method}([/_\-]|$)", low):
            return method
    return None


def infer_n_seed(member_name: str) -> tuple[int | None, int | None]:
    low = member_name.lower()
    n_values = re.findall(r"(?:^|[/_\-])n(?:ris)?[_\-]?(16|32|64|96|128)(?=[/_\-.]|$)", low)
    seeds = re.findall(r"seed[_\-]?(\d+)", low)
    return (int(n_values[-1]) if n_values else None, int(seeds[-1]) if seeds else None)


def core_finite(df: pd.DataFrame) -> bool:
    for column in CORE_NUMERIC:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            return False
    for column in df.select_dtypes(include=[np.number]).columns:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if np.isinf(values).any():
            return False
    return True


def read_csv_bytes(data: bytes, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        raise RuntimeError(f"Cannot read {label}: {exc}") from exc


def metric_mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns or df.empty:
        return None
    values = pd.to_numeric(df[column], errors="coerce")
    return float(values.mean())


def scan_archives(root: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int], dict[str, list[tuple[str, bytes]]]]]:
    archives: list[dict[str, Any]] = []
    units: dict[tuple[str, int, int], dict[str, list[tuple[str, bytes]]]] = defaultdict(lambda: defaultdict(list))

    for archive in sorted(root.rglob("*.zip")):
        archive_method = infer_method(archive.as_posix())
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            selected = 0
            for info in infos:
                name = info.filename
                suffix = Path(name).suffix.lower()
                if suffix not in {".csv", ".json"}:
                    continue
                method = infer_method(name) or archive_method
                n_ris, seed = infer_n_seed(name)
                if method is None or n_ris is None or seed is None or seed not in SEEDS:
                    continue

                low = name.lower()
                base = Path(name).name.lower()
                if base == "training.csv":
                    kind = "training"
                elif base == "validation_raw.csv":
                    kind = "validation"
                elif base == "manifest.json":
                    kind = "manifest"
                elif base.endswith(".csv") and ("/test/" in low or "/results/test/" in low or re.fullmatch(r"seed_?\d+\.csv", base)):
                    kind = "test"
                else:
                    continue

                units[(method, n_ris, seed)][kind].append((name, zf.read(info)))
                selected += 1

            archives.append({
                "archive": archive.relative_to(root).as_posix(),
                "method": archive_method,
                "archive_bytes": archive.stat().st_size,
                "member_count": len(infos),
                "selected_result_members": selected,
                "uncompressed_bytes": sum(i.file_size for i in infos),
            })
    return archives, units


def choose(items: list[tuple[str, bytes]]) -> tuple[str, bytes] | None:
    if not items:
        return None
    return sorted(items, key=lambda item: (len(Path(item[0]).parts), len(item[0])))[0]


def analyze_unit(method: str, n_ris: int, seed: int, files: dict[str, list[tuple[str, bytes]]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row: dict[str, Any] = {
        "method": method,
        "n_ris": n_ris,
        "seed": seed,
        "training_csv": False,
        "validation_raw_csv": False,
        "manifest_json": False,
        "test_csv": False,
        "complete": False,
        "finite": True,
        "issues": [],
    }
    curve: list[dict[str, Any]] = []

    training = choose(files.get("training", []))
    validation = choose(files.get("validation", []))
    manifest = choose(files.get("manifest", []))
    test = choose(files.get("test", []))

    if training:
        row["training_csv"] = True
        row["training_path"] = training[0]
        df = read_csv_bytes(training[1], training[0])
        row["training_rows"] = len(df)
        row["training_finite"] = core_finite(df)
        row["finite"] = row["finite"] and row["training_finite"]
        if "step" in df.columns and len(df):
            row["training_last_step"] = int(pd.to_numeric(df["step"], errors="coerce").max())
        if "reward" in df.columns and len(df):
            rewards = pd.to_numeric(df["reward"], errors="coerce")
            row["training_reward_first"] = float(rewards.iloc[0])
            row["training_reward_last"] = float(rewards.iloc[-1])
    else:
        row["issues"].append("missing training.csv")

    if validation:
        row["validation_raw_csv"] = True
        row["validation_path"] = validation[0]
        df = read_csv_bytes(validation[1], validation[0])
        row["validation_rows"] = len(df)
        row["validation_finite"] = core_finite(df)
        row["finite"] = row["finite"] and row["validation_finite"]
        if "eval_step" not in df.columns:
            row["issues"].append("validation_raw.csv missing eval_step")
        else:
            for step, group in df.groupby("eval_step", sort=True):
                point: dict[str, Any] = {
                    "method": method,
                    "n_ris": n_ris,
                    "seed": seed,
                    "eval_step": int(step),
                    "n_scenarios": len(group),
                }
                for column in CORE_NUMERIC:
                    point[f"mean_{column}"] = metric_mean(group, column)
                curve.append(point)
            if curve:
                best = max(curve, key=lambda p: -math.inf if p.get("mean_reward") is None else p["mean_reward"])
                row["validation_points"] = len(curve)
                row["first_validation_reward"] = curve[0].get("mean_reward")
                row["last_validation_reward"] = curve[-1].get("mean_reward")
                row["best_validation_reward"] = best.get("mean_reward")
                row["best_validation_step"] = best.get("eval_step")
                if row["first_validation_reward"] is not None and row["best_validation_reward"] is not None:
                    row["validation_reward_gain"] = row["best_validation_reward"] - row["first_validation_reward"]
    else:
        row["issues"].append("missing validation_raw.csv")

    if manifest:
        row["manifest_json"] = True
        row["manifest_path"] = manifest[0]
        try:
            payload = json.loads(manifest[1].decode("utf-8"))
            config = payload.get("config", {}) or {}
            row["manifest_method"] = payload.get("method")
            row["manifest_seed"] = payload.get("seed")
            row["manifest_n_ris"] = config.get("n_ris")
            row["config_hash"] = payload.get("config_hash")
            row["manifest_best_validation_reward"] = payload.get("best_validation_reward")
            banks = payload.get("scenario_banks", {}) or {}
            for split in ("train", "validation", "test"):
                if split in banks:
                    row[f"{split}_bank_checksum"] = banks[split].get("checksum")
            if payload.get("method") not in (None, method):
                row["issues"].append("manifest method mismatch")
            if payload.get("seed") not in (None, seed):
                row["issues"].append("manifest seed mismatch")
            if config.get("n_ris") not in (None, n_ris):
                row["issues"].append("manifest N mismatch")
        except Exception as exc:
            row["issues"].append(f"invalid manifest: {exc}")
    else:
        row["issues"].append("missing manifest.json")

    if test:
        row["test_csv"] = True
        row["test_path"] = test[0]
        df = read_csv_bytes(test[1], test[0])
        row["test_rows"] = len(df)
        row["test_finite"] = core_finite(df)
        row["finite"] = row["finite"] and row["test_finite"]
        for column in CORE_NUMERIC:
            row[f"test_mean_{column}"] = metric_mean(df, column)
            if column in df.columns:
                row[f"test_std_{column}"] = float(pd.to_numeric(df[column], errors="coerce").std(ddof=1))
        if "scenario" in df.columns:
            row["test_unique_scenarios"] = int(df["scenario"].nunique())
            if row["test_unique_scenarios"] != len(df):
                row["issues"].append("duplicate test scenarios")
        if "config_hash" in df.columns:
            values = df["config_hash"].dropna().astype(str).unique()
            row["test_config_hash_count"] = len(values)
            if len(values) != 1:
                row["issues"].append("test config hash not unique")
        if "bank_checksum" in df.columns:
            values = df["bank_checksum"].dropna().astype(str).unique()
            row["test_bank_checksum_count"] = len(values)
            row["test_bank_checksum"] = values[0] if len(values) == 1 else None
            if len(values) != 1:
                row["issues"].append("test bank checksum not unique")
    else:
        row["issues"].append("missing test CSV")

    row["complete"] = all(row[key] for key in ("training_csv", "validation_raw_csv", "manifest_json", "test_csv"))
    if not row["finite"]:
        row["issues"].append("NaN or Inf in core metrics")
    row["issue_count"] = len(row["issues"])
    row["issues"] = "; ".join(row["issues"])
    return row, curve


def summarize(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    complete = units[units["complete"] == True].copy()  # noqa: E712
    metrics = (
        "test_mean_sum_rate",
        "test_mean_reward",
        "test_mean_qos_fraction",
        "test_mean_all_qos",
        "test_mean_violation",
        "best_validation_reward",
        "validation_reward_gain",
    )
    for (method, n_ris), group in complete.groupby(["method", "n_ris"], sort=True):
        result: dict[str, Any] = {
            "method": method,
            "n_ris": int(n_ris),
            "seed_count": int(group["seed"].nunique()),
        }
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            result[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            result[f"{metric}_std_across_seeds"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else None
            result[f"{metric}_min"] = float(values.min()) if len(values) else None
            result[f"{metric}_max"] = float(values.max()) if len(values) else None
        rows.append(result)
    return pd.DataFrame(rows)


def consistency_findings(units: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    for method in METHODS:
        for n_ris in N_VALUES:
            group = units[(units.method == method) & (units.n_ris == n_ris)]
            complete = int(group["complete"].sum())
            if complete != 8:
                findings.append(f"{method.upper()} N={n_ris}: only {complete}/8 complete seeds")
            if "test_bank_checksum" in group.columns:
                checks = group.loc[group["test_csv"] == True, "test_bank_checksum"].dropna().astype(str).unique()  # noqa: E712
                if len(checks) > 1:
                    findings.append(f"{method.upper()} N={n_ris}: inconsistent test-bank checksum across seeds")
    if "test_bank_checksum" in units.columns:
        for n_ris in N_VALUES:
            checks = units.loc[(units.n_ris == n_ris) & (units.test_csv == True), "test_bank_checksum"].dropna().astype(str).unique()  # noqa: E712
            if len(checks) > 1:
                findings.append(f"N={n_ris}: methods do not share one test-bank checksum")
    for _, bad in units[units["issue_count"] > 0].iterrows():
        findings.append(f"{bad.method.upper()} N={int(bad.n_ris)} seed={int(bad.seed)}: {bad.issues}")
    return findings


def write_report(archives: list[dict[str, Any]], units: pd.DataFrame, summary: pd.DataFrame, findings: list[str], output: Path) -> None:
    lines = [
        "# Kaggle training audit v2",
        "",
        f"- Archives: **{len(archives)}**",
        f"- Expected units: **{len(METHODS) * len(N_VALUES) * len(SEEDS)}**",
        f"- Complete units: **{int(units['complete'].sum())}/{len(units)}**",
        f"- Units with non-finite core metrics: **{int((units['finite'] == False).sum())}**",  # noqa: E712
        "",
        "## Coverage and test results",
        "",
        "| Method | N | Complete | Finite | Test sum-rate | QoS fraction | All-QoS | Violation | Val. gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        for n_ris in N_VALUES:
            group = units[(units.method == method) & (units.n_ris == n_ris)]
            agg = summary[(summary.method == method) & (summary.n_ris == n_ris)]
            def fmt(column: str) -> str:
                if agg.empty or column not in agg.columns:
                    return "—"
                value = agg.iloc[0].get(column)
                return "—" if pd.isna(value) else f"{float(value):.6g}"
            lines.append(
                f"| {method.upper()} | {n_ris} | {int(group['complete'].sum())}/8 | {int(group['finite'].sum())}/8 | "
                f"{fmt('test_mean_sum_rate_mean')} | {fmt('test_mean_qos_fraction_mean')} | "
                f"{fmt('test_mean_all_qos_mean')} | {fmt('test_mean_violation_mean')} | "
                f"{fmt('validation_reward_gain_mean')} |"
            )
    lines.extend(["", "## Findings", ""])
    lines.extend([f"- {finding}" for finding in findings[:250]] if findings else ["- No structural, checksum, or core numerical issues detected."])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    archives, classified = scan_archives(args.archives_root)
    pd.DataFrame(archives).to_csv(args.output_dir / "archive_summary.csv", index=False)

    unit_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for n_ris in N_VALUES:
            for seed in SEEDS:
                unit, curve = analyze_unit(method, n_ris, seed, classified.get((method, n_ris, seed), {}))
                unit_rows.append(unit)
                curve_rows.extend(curve)

    units = pd.DataFrame(unit_rows)
    units.to_csv(args.output_dir / "unit_audit.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(args.output_dir / "validation_curve_summary.csv", index=False)
    aggregate = summarize(units)
    aggregate.to_csv(args.output_dir / "method_n_summary.csv", index=False)
    findings = consistency_findings(units)
    (args.output_dir / "audit_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    write_report(archives, units, aggregate, findings, args.output_dir / "AUDIT_REPORT.md")
    print((args.output_dir / "AUDIT_REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
