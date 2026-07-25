from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from star_ris_rsma.result_validation import CORE_METRICS, coerce_core_metrics


def audit(root: Path, expected_seed: int) -> dict[str, float | int]:
    manifest_path = root / "train" / "manifest.json"
    test_path = root / "test.csv"
    if not manifest_path.exists() or not test_path.exists():
        raise SystemExit(
            f"Missing pilot output: manifest={manifest_path.exists()}, "
            f"test={test_path.exists()}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config", {}).get("action_parameterization") != "physical_v3":
        raise SystemExit("Pilot did not use physical_v3")
    if int(manifest.get("seed", -1)) != int(expected_seed):
        raise SystemExit(
            f"Manifest seed mismatch: {manifest.get('seed')} != {expected_seed}"
        )

    test = pd.read_csv(test_path)
    scenarios = (
        pd.to_numeric(test["scenario"], errors="coerce")
        if "scenario" in test.columns
        else pd.Series(dtype=float)
    )
    if (
        len(test) != 1000
        or scenarios.isna().any()
        or scenarios.nunique() != 1000
        or int(scenarios.min()) != 0
        or int(scenarios.max()) != 999
    ):
        raise SystemExit(
            f"Incomplete pilot test coverage: rows={len(test)}, "
            f"scenarios={scenarios.nunique()}"
        )

    if "seed" not in test.columns:
        raise SystemExit("Test CSV has no seed column")
    seeds = pd.to_numeric(test["seed"], errors="coerce")
    if (
        seeds.isna().any()
        or seeds.nunique() != 1
        or int(seeds.iloc[0]) != int(expected_seed)
    ):
        raise SystemExit(f"Test seed mismatch for expected seed {expected_seed}")

    try:
        numeric = coerce_core_metrics(
            test,
            context=f"pilot seed {expected_seed}",
            require_finite=True,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    means = numeric.mean()
    result: dict[str, float | int] = {
        "rows": int(len(test)),
        "scenarios": int(scenarios.nunique()),
        "seed": int(expected_seed),
    }
    result.update(
        {f"mean_{metric}": float(means[metric]) for metric in CORE_METRICS}
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
