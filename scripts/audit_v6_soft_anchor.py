from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_SEEDS = (0, 1)
EXPECTED_PARAMETERIZATION = "physical_v6_soft_anchor"
EXPECTED_ROWS = 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    checksums: set[str] = set()
    for seed in EXPECTED_SEEDS:
        summaries = list(args.input.rglob(f"v6_soft_anchor_n32_100k_seed{seed}/summary.json"))
        if len(summaries) != 1:
            raise RuntimeError(f"seed {seed}: expected one summary, found {len(summaries)}")
        root = summaries[0].parent
        summary = json.loads(summaries[0].read_text(encoding="utf-8"))
        if summary["parameterization"] != EXPECTED_PARAMETERIZATION:
            raise RuntimeError(f"seed {seed}: wrong parameterization")
        if int(summary["seed"]) != seed or int(summary["train_steps"]) != 100000:
            raise RuntimeError(f"seed {seed}: wrong seed or training budget")
        checksums.add(str(summary["test_bank_checksum"]))
        for checkpoint in ("initial", "best", "latest"):
            raw = pd.read_csv(root / f"test_{checkpoint}_raw.csv")
            if len(raw) != EXPECTED_ROWS:
                raise RuntimeError(f"seed {seed} {checkpoint}: expected 1000 rows")
            metrics = summary["checkpoints"][checkpoint]
            rows.append({
                "parameterization": EXPECTED_PARAMETERIZATION,
                "seed": seed,
                "checkpoint": checkpoint,
                "train_steps": 100000,
                "test_bank_checksum": summary["test_bank_checksum"],
                **metrics,
            })
    if len(checksums) != 1:
        raise RuntimeError(f"test bank mismatch: {sorted(checksums)}")

    frame = pd.DataFrame(rows).sort_values(["seed", "checkpoint"])
    frame.to_csv(args.output / "V6_CHECKPOINT_METRICS.csv", index=False)
    gains = []
    for seed in EXPECTED_SEEDS:
        part = frame[frame.seed == seed].set_index("checkpoint")
        gains.append({
            "seed": seed,
            "initial": part.loc["initial", "sum_rate_mean"],
            "best": part.loc["best", "sum_rate_mean"],
            "latest": part.loc["latest", "sum_rate_mean"],
            "learning_gain_best_minus_initial": (
                part.loc["best", "sum_rate_mean"] - part.loc["initial", "sum_rate_mean"]
            ),
            "latest_minus_initial": (
                part.loc["latest", "sum_rate_mean"] - part.loc["initial", "sum_rate_mean"]
            ),
        })
    gain_frame = pd.DataFrame(gains)
    gain_frame.to_csv(args.output / "V6_LEARNING_GAIN.csv", index=False)
    audit = {
        "audit": "PASS",
        "parameterization": EXPECTED_PARAMETERIZATION,
        "seeds": list(EXPECTED_SEEDS),
        "rows_per_checkpoint": EXPECTED_ROWS,
        "test_bank_checksum": next(iter(checksums)),
        "best_two_seed_mean": float(
            frame[frame.checkpoint == "best"].sum_rate_mean.mean()
        ),
        "latest_two_seed_mean": float(
            frame[frame.checkpoint == "latest"].sum_rate_mean.mean()
        ),
        "learning_gain_two_seed_mean": float(
            gain_frame.learning_gain_best_minus_initial.mean()
        ),
    }
    (args.output / "V6_AUDIT.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    report = "# Physical V6 soft-anchor N=32 pilot\n\n"
    report += f"Audit: **PASS**; test bank `{audit['test_bank_checksum']}`.\n\n"
    report += gain_frame.to_markdown(index=False) + "\n"
    (args.output / "V6_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
