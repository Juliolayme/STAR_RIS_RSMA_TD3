from __future__ import annotations

import argparse
import csv
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
TEXT_SUFFIXES = {".csv", ".json", ".yaml", ".yml", ".txt", ".log", ".md"}


def _finite_frame(df: pd.DataFrame) -> bool:
    numeric = df.select_dtypes(include=[np.number])
    return bool(np.isfinite(numeric.to_numpy(dtype=float)).all()) if not numeric.empty else True


def _infer_method(path: str) -> str | None:
    low = path.lower()
    for method in METHODS:
        if re.search(rf"(^|[/_\-]){method}([/_\-]|$)", low):
            return method
    return None


def _infer_n_seed(path: str) -> tuple[int | None, int | None]:
    low = path.lower()
    n_match = re.search(r"(?:^|[/_\-])n(?:ris)?[_\-]?(16|32|64|96|128)(?:[/_\-]|$)", low)
    seed_match = re.search(r"seed[_\-]?(\d+)", low)
    return (int(n_match.group(1)) if n_match else None, int(seed_match.group(1)) if seed_match else None)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Cannot read CSV {path}: {exc}") from exc


def extract_lightweight(archives_root: Path, extracted_root: Path) -> list[dict[str, Any]]:
    extracted_root.mkdir(parents=True, exist_ok=True)
    archive_rows: list[dict[str, Any]] = []
    for archive in sorted(archives_root.rglob("*.zip")):
        method = _infer_method(archive.as_posix()) or "unknown"
        target = extracted_root / method / archive.stem
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            members = zf.infolist()
            selected = [m for m in members if Path(m.filename).suffix.lower() in TEXT_SUFFIXES]
            for member in selected:
                safe = Path(member.filename)
                if safe.is_absolute() or ".." in safe.parts:
                    continue
                zf.extract(member, target)
            archive_rows.append({
                "archive": archive.relative_to(archives_root).as_posix(),
                "method": method,
                "archive_bytes": archive.stat().st_size,
                "member_count": len(members),
                "selected_text_members": len(selected),
                "uncompressed_bytes": sum(m.file_size for m in members),
            })
    return archive_rows


def classify_files(root: Path) -> dict[tuple[str, int, int], dict[str, list[Path]]]:
    units: dict[tuple[str, int, int], dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        method = _infer_method(rel)
        n_ris, seed = _infer_n_seed(rel)
        if method is None or n_ris is None or seed is None or seed not in SEEDS:
            continue
        low_name = path.name.lower()
        if low_name == "training.csv":
            kind = "training"
        elif low_name == "validation_raw.csv":
            kind = "validation"
        elif low_name == "manifest.json":
            kind = "manifest"
        elif low_name.endswith(".csv") and ("test" in rel.lower() or re.fullmatch(r"seed_?\d+\.csv", low_name)):
            kind = "test"
        else:
            continue
        units[(method, n_ris, seed)][kind].append(path)
    return units


def choose(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return sorted(paths, key=lambda p: (len(p.parts), len(p.as_posix())))[0]


def metric_mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns or df.empty:
        return None
    return float(pd.to_numeric(df[column], errors="coerce").mean())


def analyze_unit(method: str, n_ris: int, seed: int, files: dict[str, list[Path]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    record: dict[str, Any] = {
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
    validation_curve: list[dict[str, Any]] = []

    training_path = choose(files.get("training", []))
    validation_path = choose(files.get("validation", []))
    manifest_path = choose(files.get("manifest", []))
    test_path = choose(files.get("test", []))

    if training_path:
        record["training_csv"] = True
        df = _read_csv(training_path)
        record["training_rows"] = len(df)
        record["training_finite"] = _finite_frame(df)
        record["finite"] = record["finite"] and record["training_finite"]
        if "step" in df.columns and len(df):
            record["training_last_step"] = int(pd.to_numeric(df["step"], errors="coerce").max())
        record["training_reward_first"] = float(pd.to_numeric(df["reward"], errors="coerce").iloc[0]) if "reward" in df.columns and len(df) else None
        record["training_reward_last"] = float(pd.to_numeric(df["reward"], errors="coerce").iloc[-1]) if "reward" in df.columns and len(df) else None
    else:
        record["issues"].append("missing training.csv")

    if validation_path:
        record["validation_raw_csv"] = True
        df = _read_csv(validation_path)
        record["validation_rows"] = len(df)
        record["validation_finite"] = _finite_frame(df)
        record["finite"] = record["finite"] and record["validation_finite"]
        if "eval_step" in df.columns:
            for step, group in df.groupby("eval_step", sort=True):
                row = {
                    "method": method,
                    "n_ris": n_ris,
                    "seed": seed,
                    "eval_step": int(step),
                    "n_scenarios": len(group),
                }
                for col in ("reward", "sum_rate", "qos_fraction", "all_qos", "violation"):
                    row[f"mean_{col}"] = metric_mean(group, col)
                validation_curve.append(row)
            if validation_curve:
                best = max(validation_curve, key=lambda x: -math.inf if x["mean_reward"] is None else x["mean_reward"])
                first = validation_curve[0]
                last = validation_curve[-1]
                record["validation_points"] = len(validation_curve)
                record["best_validation_step"] = best["eval_step"]
                record["best_validation_reward"] = best["mean_reward"]
                record["first_validation_reward"] = first["mean_reward"]
                record["last_validation_reward"] = last["mean_reward"]
                if first["mean_reward"] is not None and best["mean_reward"] is not None:
                    record["validation_reward_gain"] = best["mean_reward"] - first["mean_reward"]
        else:
            record["issues"].append("validation_raw.csv missing eval_step")
    else:
        record["issues"].append("missing validation_raw.csv")

    if manifest_path:
        record["manifest_json"] = True
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record["manifest_method"] = manifest.get("method")
            record["manifest_seed"] = manifest.get("seed")
            config = manifest.get("config", {}) or {}
            record["manifest_n_ris"] = config.get("n_ris")
            record["config_hash"] = manifest.get("config_hash")
            record["manifest_best_validation_reward"] = manifest.get("best_validation_reward")
            banks = manifest.get("scenario_banks", {}) or {}
            for split in ("train", "validation", "test"):
                if split in banks:
                    record[f"{split}_bank_checksum"] = banks[split].get("checksum")
            if manifest.get("method") not in (None, method):
                record["issues"].append("manifest method mismatch")
            if manifest.get("seed") not in (None, seed):
                record["issues"].append("manifest seed mismatch")
            if config.get("n_ris") not in (None, n_ris):
                record["issues"].append("manifest N mismatch")
        except Exception as exc:
            record["issues"].append(f"invalid manifest: {exc}")
    else:
        record["issues"].append("missing manifest.json")

    if test_path:
        record["test_csv"] = True
        df = _read_csv(test_path)
        record["test_rows"] = len(df)
        record["test_finite"] = _finite_frame(df)
        record["finite"] = record["finite"] and record["test_finite"]
        for col in ("reward", "sum_rate", "qos_fraction", "all_qos", "violation"):
            record[f"test_mean_{col}"] = metric_mean(df, col)
            if col in df.columns:
                record[f"test_std_{col}"] = float(pd.to_numeric(df[col], errors="coerce").std(ddof=1))
        if "scenario" in df.columns:
            record["test_unique_scenarios"] = int(df["scenario"].nunique())
            if record["test_unique_scenarios"] != len(df):
                record["issues"].append("duplicate test scenarios")
        if "config_hash" in df.columns:
            hashes = df["config_hash"].dropna().astype(str).unique()
            record["test_config_hash_count"] = len(hashes)
            if len(hashes) != 1:
                record["issues"].append("test config hash not unique")
        if "bank_checksum" in df.columns:
            checks = df["bank_checksum"].dropna().astype(str).unique()
            record["test_bank_checksum_count"] = len(checks)
            record["test_bank_checksum"] = checks[0] if len(checks) == 1 else None
            if len(checks) != 1:
                record["issues"].append("test bank checksum not unique")
    else:
        record["issues"].append("missing test CSV")

    record["complete"] = all(record[k] for k in ("training_csv", "validation_raw_csv", "manifest_json", "test_csv"))
    if not record["finite"]:
        record["issues"].append("NaN or Inf detected")
    record["issue_count"] = len(record["issues"])
    record["issues"] = "; ".join(record["issues"])
    return record, validation_curve


def summarize_seeds(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    complete = units[units["complete"] == True].copy()  # noqa: E712
    for (method, n_ris), group in complete.groupby(["method", "n_ris"], sort=True):
        row: dict[str, Any] = {"method": method, "n_ris": int(n_ris), "seed_count": int(group["seed"].nunique())}
        for metric in ("test_mean_sum_rate", "test_mean_reward", "test_mean_qos_fraction", "test_mean_all_qos", "test_mean_violation", "best_validation_reward", "validation_reward_gain"):
            if metric in group.columns:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
                row[f"{metric}_std_across_seeds"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else None
                row[f"{metric}_min"] = float(values.min()) if len(values) else None
                row[f"{metric}_max"] = float(values.max()) if len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)


def audit_consistency(units: pd.DataFrame) -> list[str]:
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
    for n_ris in N_VALUES:
        checks = units[(units.n_ris == n_ris) & (units.test_csv == True)]["test_bank_checksum"].dropna().astype(str).unique() if "test_bank_checksum" in units.columns else []  # noqa: E712
        if len(checks) > 1:
            findings.append(f"N={n_ris}: methods do not share one test-bank checksum")
    bad = units[units["issue_count"] > 0]
    for _, row in bad.iterrows():
        findings.append(f"{row.method.upper()} N={int(row.n_ris)} seed={int(row.seed)}: {row.issues}")
    return findings


def write_markdown(archive_rows: list[dict[str, Any]], units: pd.DataFrame, summary: pd.DataFrame, findings: list[str], output: Path) -> None:
    lines = [
        "# Kaggle training audit",
        "",
        f"- Archives: **{len(archive_rows)}**",
        f"- Expected method/N/seed units: **{len(METHODS) * len(N_VALUES) * len(SEEDS)}**",
        f"- Complete units: **{int(units['complete'].sum())}/{len(units)}**",
        f"- Units with NaN/Inf: **{int((units['finite'] == False).sum())}**",  # noqa: E712
        "",
        "## Coverage",
        "",
        "| Method | N | Complete seeds | Finite seeds | Mean test sum-rate | Mean QoS fraction | Mean violation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        for n_ris in N_VALUES:
            g = units[(units.method == method) & (units.n_ris == n_ris)]
            s = summary[(summary.method == method) & (summary.n_ris == n_ris)]
            def val(name: str) -> str:
                if s.empty or name not in s.columns or pd.isna(s.iloc[0].get(name)):
                    return "—"
                return f"{float(s.iloc[0][name]):.6g}"
            lines.append(
                f"| {method.upper()} | {n_ris} | {int(g['complete'].sum())}/8 | {int(g['finite'].sum())}/8 | "
                f"{val('test_mean_sum_rate_mean')} | {val('test_mean_qos_fraction_mean')} | {val('test_mean_violation_mean')} |"
            )
    lines.extend(["", "## Findings", ""])
    if findings:
        lines.extend(f"- {item}" for item in findings[:200])
    else:
        lines.append("- No structural, numerical, or checksum inconsistencies detected.")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    extracted = args.output_dir / "lightweight_extracted"
    archive_rows = extract_lightweight(args.archives_root, extracted)
    pd.DataFrame(archive_rows).to_csv(args.output_dir / "archive_summary.csv", index=False)

    classified = classify_files(extracted)
    unit_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for n_ris in N_VALUES:
            for seed in SEEDS:
                record, curve = analyze_unit(method, n_ris, seed, classified.get((method, n_ris, seed), {}))
                unit_rows.append(record)
                validation_rows.extend(curve)

    units = pd.DataFrame(unit_rows)
    units.to_csv(args.output_dir / "unit_audit.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(args.output_dir / "validation_curve_summary.csv", index=False)
    summary = summarize_seeds(units)
    summary.to_csv(args.output_dir / "method_n_summary.csv", index=False)
    findings = audit_consistency(units)
    write_markdown(archive_rows, units, summary, findings, args.output_dir / "AUDIT_REPORT.md")
    (args.output_dir / "audit_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")

    print((args.output_dir / "AUDIT_REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
