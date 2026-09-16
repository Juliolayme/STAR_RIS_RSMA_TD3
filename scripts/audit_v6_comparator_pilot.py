from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED = tuple(
    (method, n_ris, seed)
    for method in ("ddpg", "ppo")
    for n_ris in (32, 128)
    for seed in (0, 1)
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    checksums: dict[int, set[str]] = {32: set(), 128: set()}
    for method, n_ris, seed in EXPECTED:
        tag = f"{method}_v6_soft_anchor_n{n_ris}_100k_seed{seed}"
        found = list(args.input.rglob(f"{tag}/summary.json"))
        if len(found) != 1:
            raise RuntimeError(f"{tag}: expected one summary, found {len(found)}")
        root = found[0].parent
        summary = json.loads(found[0].read_text(encoding="utf-8"))
        expected_values = {
            "method": method,
            "seed": seed,
            "parameterization": "physical_v6_soft_anchor",
            "train_steps": 100000,
        }
        for key, expected in expected_values.items():
            if summary.get(key) != expected:
                raise RuntimeError(f"{tag}: {key}={summary.get(key)!r}, expected {expected!r}")
        checksums[n_ris].add(str(summary["test_bank_checksum"]))
        for checkpoint in ("initial", "best", "latest"):
            raw = pd.read_csv(root / f"test_{checkpoint}_raw.csv")
            if len(raw) != 1000:
                raise RuntimeError(f"{tag} {checkpoint}: {len(raw)} rows")
            rows.append({
                "method": method,
                "n_ris": n_ris,
                "seed": seed,
                "checkpoint": checkpoint,
                "test_bank_checksum": summary["test_bank_checksum"],
                **summary["checkpoints"][checkpoint],
            })
    for n_ris, values in checksums.items():
        if len(values) != 1:
            raise RuntimeError(f"N={n_ris}: mismatched checksums {sorted(values)}")

    frame = pd.DataFrame(rows).sort_values(["method", "n_ris", "seed", "checkpoint"])
    frame.to_csv(args.output / "COMPARATOR_CHECKPOINT_METRICS.csv", index=False)
    gains = []
    for (method, n_ris, seed), part in frame.groupby(["method", "n_ris", "seed"]):
        indexed = part.set_index("checkpoint")
        gains.append({
            "method": method,
            "n_ris": n_ris,
            "seed": seed,
            "initial": indexed.loc["initial", "sum_rate_mean"],
            "best": indexed.loc["best", "sum_rate_mean"],
            "latest": indexed.loc["latest", "sum_rate_mean"],
            "learning_gain": indexed.loc["best", "sum_rate_mean"] - indexed.loc["initial", "sum_rate_mean"],
            "best_all_qos": indexed.loc["best", "all_qos_mean"],
            "best_violation": indexed.loc["best", "violation_mean"],
        })
    gain_frame = pd.DataFrame(gains)
    gain_frame.to_csv(args.output / "COMPARATOR_LEARNING_GAIN.csv", index=False)
    aggregate = gain_frame.groupby(["method", "n_ris"], as_index=False).agg(
        seeds=("seed", "count"), initial_mean=("initial", "mean"),
        best_mean=("best", "mean"), latest_mean=("latest", "mean"),
        learning_gain_mean=("learning_gain", "mean"),
        best_all_qos_mean=("best_all_qos", "mean"),
        best_violation_mean=("best_violation", "mean"),
    )
    aggregate.to_csv(args.output / "COMPARATOR_AGGREGATE.csv", index=False)
    audit = {
        "audit": "PASS", "units": len(EXPECTED), "rows_per_checkpoint": 1000,
        "checksums": {str(n): next(iter(v)) for n, v in checksums.items()},
    }
    (args.output / "COMPARATOR_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    report = "# V6 DDPG/PPO comparator pilot\n\nAudit: **PASS**.\n\n" + aggregate.to_markdown(index=False) + "\n"
    (args.output / "COMPARATOR_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
