from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from star_ris_rsma.baselines.ao_corrected import (
    ALGORITHM_VERSION,
    FROZEN_STATIONARITY_TOL,
)

N_VALUES = (16, 32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.input_root.rglob("*.csv"))
    if not paths:
        raise RuntimeError(f"No pilot CSV files found under {args.input_root}")
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    raw = raw[raw.method.astype(str).str.lower() == "ao_sca"].copy()

    if raw.duplicated(["n_ris", "scenario"]).any():
        raise RuntimeError("Duplicate N/scenario rows in AO stationarity pilot")

    rows: list[dict[str, object]] = []
    for n_ris in N_VALUES:
        group = raw[raw.n_ris.astype(int) == n_ris]
        if len(group) != 1000 or set(group.scenario.astype(int)) != set(range(1000)):
            raise RuntimeError(f"N={n_ris}: incomplete 1,000-scenario coverage")
        versions = set(group.algorithm_version.astype(str))
        if versions != {ALGORITHM_VERSION}:
            raise RuntimeError(f"N={n_ris}: unexpected versions {versions}")

        power = group.power_stationarity_gap.to_numpy(dtype=float)
        common = group.common_stationarity_gap.to_numpy(dtype=float)
        finite = bool(np.isfinite(power).all() and np.isfinite(common).all())
        failures = (power >= FROZEN_STATIONARITY_TOL) | (
            common >= FROZEN_STATIONARITY_TOL
        )
        rows.append(
            {
                "n_ris": n_ris,
                "scenarios": len(group),
                "stationarity_tolerance": FROZEN_STATIONARITY_TOL,
                "finite_gaps": finite,
                "power_gap_max": float(np.max(power)),
                "common_gap_max": float(np.max(common)),
                "stationarity_failures": int(np.sum(failures)),
                "max_iterations_used": int(group.iterations.astype(int).max()),
                "hit_max_iter": int(
                    np.sum(group.iterations.astype(int) >= group.max_iter.astype(int))
                ),
            }
        )

    verdict = "PASS" if all(row["finite_gaps"] and row["stationarity_failures"] == 0 for row in rows) else "FAIL"
    audit = {
        "verdict": verdict,
        "algorithm_version": ALGORITHM_VERSION,
        "criterion": "pairwise_probe=1e-4; post-RIS simplex polish, then relative objective change < 1e-4 AND power/common gaps < 1e-6; max_iter=80",
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    if verdict != "PASS":
        raise SystemExit("AO stationarity pilot failed")


if __name__ == "__main__":
    main()
