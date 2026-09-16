from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED = {
    ("physical_v5_hard", 0),
    ("physical_v5_hard", 1),
    ("physical_v5_soft", 0),
    ("physical_v5_soft", 1),
}
CHECKPOINTS = ("initial", "best", "latest")


def validate_raw(path: Path, seed: int, checksum: str) -> dict[str, float]:
    frame = pd.read_csv(path)
    if len(frame) != 1000:
        raise RuntimeError(f"{path}: expected 1000 rows, found {len(frame)}")
    scenarios = pd.to_numeric(frame["scenario"], errors="raise").astype(int)
    if set(scenarios) != set(range(1000)):
        raise RuntimeError(f"{path}: scenarios are not exactly 0..999")
    seeds = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    if set(seeds) != {seed}:
        raise RuntimeError(f"{path}: seed mismatch {sorted(set(seeds))} != {seed}")
    checksums = set(frame["bank_checksum"].astype(str))
    if checksums != {checksum}:
        raise RuntimeError(f"{path}: bank checksum mismatch {checksums} != {checksum}")
    metrics = frame[["sum_rate", "qos_fraction", "all_qos", "violation"]].copy()
    for column in ("sum_rate", "qos_fraction", "violation"):
        metrics[column] = pd.to_numeric(metrics[column], errors="raise")
    metrics["all_qos"] = metrics["all_qos"].astype(str).str.lower().map(
        {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
    )
    if metrics.isna().any().any():
        raise RuntimeError(f"{path}: non-numeric metric")
    return {f"{column}_mean": float(metrics[column].mean()) for column in metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summaries = list(args.input.rglob("summary.json"))
    records: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    checksums: set[str] = set()
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        parameterization = str(summary.get("parameterization"))
        seed = int(summary.get("seed", -1))
        key = (parameterization, seed)
        if key not in EXPECTED or key in seen:
            continue
        if int(summary.get("train_steps", -1)) != 100000:
            raise RuntimeError(f"{summary_path}: train_steps is not 100000")
        checksum = str(summary.get("test_bank_checksum", ""))
        if not checksum:
            raise RuntimeError(f"{summary_path}: missing test bank checksum")
        seen.add(key)
        checksums.add(checksum)
        for checkpoint in CHECKPOINTS:
            raw_path = summary_path.parent / f"test_{checkpoint}_raw.csv"
            measured = validate_raw(raw_path, seed, checksum)
            declared = summary["checkpoints"][checkpoint]
            for metric, value in measured.items():
                if abs(float(declared[metric]) - value) > 1e-10:
                    raise RuntimeError(
                        f"{summary_path}: declared {checkpoint}/{metric} does not match raw"
                    )
            records.append({
                "parameterization": parameterization,
                "seed": seed,
                "checkpoint": checkpoint,
                "train_steps": int(summary["train_steps"]),
                "test_bank_checksum": checksum,
                **measured,
            })

    if seen != EXPECTED:
        raise RuntimeError(f"Missing pilot units: {sorted(EXPECTED - seen)}; found={sorted(seen)}")
    if len(checksums) != 1:
        raise RuntimeError(f"Pilot units used different test banks: {sorted(checksums)}")

    raw = pd.DataFrame(records).sort_values(["parameterization", "seed", "checkpoint"])
    raw.to_csv(args.output / "PILOT_CHECKPOINT_METRICS.csv", index=False)
    aggregate = (
        raw.groupby(["parameterization", "checkpoint"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            sum_rate_mean=("sum_rate_mean", "mean"),
            sum_rate_seed_std=("sum_rate_mean", "std"),
            all_qos_mean=("all_qos_mean", "mean"),
            qos_fraction_mean=("qos_fraction_mean", "mean"),
            violation_mean=("violation_mean", "mean"),
        )
    )
    aggregate.to_csv(args.output / "PILOT_HARD_VS_SOFT.csv", index=False)

    best = aggregate[aggregate["checkpoint"] == "best"].set_index("parameterization")
    learning = raw.pivot_table(
        index=["parameterization", "seed"], columns="checkpoint", values="sum_rate_mean"
    ).reset_index()
    learning["learning_gain_best_minus_initial"] = learning["best"] - learning["initial"]
    learning.to_csv(args.output / "PILOT_LEARNING_GAIN.csv", index=False)
    verdict = {
        "audit": "PASS",
        "units": len(seen),
        "rows_per_checkpoint": 1000,
        "test_bank_checksum": next(iter(checksums)),
        "best_two_seed_mean": {
            method: float(best.loc[method, "sum_rate_mean"])
            for method in sorted(best.index)
        },
    }
    (args.output / "PILOT_AUDIT.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    report = [
        "# Structure-aware TD3 N=32 pilot",
        "",
        f"Audit: **PASS**; common test bank `{verdict['test_bank_checksum']}`; 1000 CSI/checkpoint.",
        "",
        aggregate.to_markdown(index=False),
        "",
        "## Per-seed learning gain",
        "",
        learning.to_markdown(index=False),
    ]
    (args.output / "PILOT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
