from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("sum_rate", "qos_fraction", "all_qos", "violation")
_BOOL_MAP = {
    "true": 1.0,
    "false": 0.0,
    "1": 1.0,
    "0": 0.0,
}


def coerce_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the core test metrics as finite float64 columns.

    Pandas can produce an object array when float columns are mixed with a
    boolean ``all_qos`` column. Calling ``np.isfinite`` on that object array
    raises a TypeError, so every column is explicitly converted to float64.
    String booleans are accepted because CSV readers may infer them as object.
    """
    missing = [column for column in METRICS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing metric columns: {missing}")

    numeric = pd.DataFrame(index=frame.index)
    for column in ("sum_rate", "qos_fraction", "violation"):
        numeric[column] = pd.to_numeric(frame[column], errors="coerce")

    all_qos = pd.to_numeric(frame["all_qos"], errors="coerce")
    unresolved = all_qos.isna()
    if unresolved.any():
        mapped = (
            frame.loc[unresolved, "all_qos"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(_BOOL_MAP)
        )
        all_qos.loc[unresolved] = mapped
    numeric["all_qos"] = all_qos

    return numeric.loc[:, list(METRICS)].astype("float64")


def audit(root: Path, expected_seed: int) -> dict[str, float | int]:
    manifest_path = root / "train" / "manifest.json"
    test_path = root / "test.csv"
    if not manifest_path.exists() or not test_path.exists():
        raise SystemExit(
            f"Missing pilot output: manifest={manifest_path.exists()}, test={test_path.exists()}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config", {}).get("action_parameterization") != "physical_v3":
        raise SystemExit("Pilot did not use physical_v3")
    if int(manifest.get("seed", -1)) != int(expected_seed):
        raise SystemExit(
            f"Manifest seed mismatch: {manifest.get('seed')} != {expected_seed}"
        )

    test = pd.read_csv(test_path)
    if len(test) != 1000 or test.get("scenario", pd.Series(dtype=int)).nunique() != 1000:
        raise SystemExit(
            f"Incomplete pilot test coverage: rows={len(test)}, "
            f"scenarios={test.get('scenario', pd.Series(dtype=int)).nunique()}"
        )
    if "seed" not in test.columns:
        raise SystemExit("Test CSV has no seed column")
    seeds = pd.to_numeric(test["seed"], errors="coerce")
    if seeds.isna().any() or seeds.nunique() != 1 or int(seeds.iloc[0]) != int(expected_seed):
        raise SystemExit(f"Test seed mismatch for expected seed {expected_seed}")

    numeric = coerce_metric_frame(test)
    values = numeric.to_numpy(dtype=np.float64, copy=False)
    finite_mask = np.isfinite(values)
    if not finite_mask.all():
        bad_rows, bad_columns = np.where(~finite_mask)
        examples = [
            {
                "row": int(row),
                "column": METRICS[int(column)],
                "raw_value": repr(test.iloc[int(row)][METRICS[int(column)]]),
            }
            for row, column in zip(bad_rows[:10], bad_columns[:10])
        ]
        raise SystemExit(f"Non-finite or non-numeric pilot metrics: {examples}")

    means = numeric.mean()
    result: dict[str, float | int] = {
        "rows": int(len(test)),
        "scenarios": int(test["scenario"].nunique()),
        "seed": int(expected_seed),
    }
    result.update({f"mean_{metric}": float(means[metric]) for metric in METRICS})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
