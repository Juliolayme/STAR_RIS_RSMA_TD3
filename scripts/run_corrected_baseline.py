from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from star_ris_rsma.action import decode_action
from star_ris_rsma.baselines.analytical_ris import solve as solve_analytical
from star_ris_rsma.baselines.ao_corrected import (
    ALGORITHM_VERSION as AO_VERSION,
    FROZEN_MAX_ITER,
    FROZEN_PAIRWISE_PROBE,
    FROZEN_STATIONARITY_TOL,
    solve as solve_ao,
)
from star_ris_rsma.baselines.ao_grid_corrected import (
    ALGORITHM_VERSION as GRID_VERSION,
    FROZEN_ROUNDS,
    solve as solve_grid,
)
from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.env import StarRisRsmaEnv
from star_ris_rsma.scenario_bank import generate_bank

N_VALUES = (16, 32, 64, 96, 128)
METHODS = ("ao_sca", "ao_grid", "analytical_ris")


def expected_checksum(drl_raw: Path, n_ris: int) -> str:
    frame = pd.read_csv(drl_raw, usecols=["n_ris", "bank_checksum"])
    values = frame.loc[frame.n_ris.astype(int) == int(n_ris), "bank_checksum"].dropna().astype(str).unique()
    if len(values) != 1:
        raise RuntimeError(f"Expected one canonical DRL test-bank checksum for N={n_ris}, got {values}")
    return str(values[0])


def solve_one(method: str, env, scenario: int):
    if method == "ao_sca":
        return solve_ao(env, seed=scenario, max_iter=FROZEN_MAX_ITER)
    if method == "ao_grid":
        return solve_grid(env, seed=scenario, rounds=FROZEN_ROUNDS)
    if method == "analytical_ris":
        return solve_analytical(env)
    raise ValueError(method)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--n-ris", type=int, choices=N_VALUES, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--drl-raw", type=Path, default=Path("results/six_method_v1/raw/DRL_TEST_RAW_ALL.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not (0 <= args.start < args.end <= 1000):
        raise SystemExit(f"Invalid range [{args.start}, {args.end})")

    cfg_path = Path(f"configs/v3/constrained_action_n{args.n_ris}.yaml")
    cfg = ExperimentConfig.from_yaml(cfg_path)
    if int(cfg.n_ris) != int(args.n_ris):
        raise RuntimeError(f"Config mismatch: {cfg.n_ris} != {args.n_ris}")

    bank = generate_bank(cfg, 1000, 33001, "test")
    checksum = bank.checksum()
    canonical = expected_checksum(args.drl_raw, args.n_ris)
    if checksum != canonical:
        raise RuntimeError(
            f"Locked ScenarioBank mismatch for N={args.n_ris}: generated={checksum}, canonical={canonical}"
        )

    rows: list[dict[str, object]] = []
    git_commit = os.environ.get("GITHUB_SHA", "unknown")
    for scenario in range(args.start, args.end):
        env = StarRisRsmaEnv(cfg, seed=scenario)
        env.reset(channel=bank.channel(scenario))
        started = time.perf_counter_ns()
        raw, metrics = solve_one(args.method, env, scenario)
        solve_ms = (time.perf_counter_ns() - started) / 1e6
        decoded = decode_action(
            raw,
            cfg.n_users,
            cfg.n_ris,
            cfg.p_max,
            cfg.action_parameterization,
        )
        powers = np.asarray(decoded.powers, dtype=float)
        common = np.asarray(decoded.common_fractions, dtype=float)

        algorithm_version = str(metrics.get("algorithm_version", metrics.get("solver", args.method)))
        rows.append(
            {
                "method": args.method,
                "n_ris": int(args.n_ris),
                "seed": 0,
                "scenario": int(scenario),
                "split": "test",
                "sum_rate": float(metrics["sum_rate"]),
                "reward": float(metrics["reward"]),
                "qos_fraction": float(metrics["qos_fraction"]),
                "all_qos": bool(metrics["all_qos"]),
                "violation": float(metrics["violation"]),
                "iterations": int(metrics.get("iterations", 0)),
                "evaluations": int(metrics.get("evaluations", 1)),
                "accepted_steps": int(metrics.get("accepted_steps", 0)),
                "solver": str(metrics.get("solver", args.method)),
                "algorithm_version": algorithm_version,
                "max_iter": int(metrics.get("max_iter", 0)),
                "stationarity_tolerance": float(
                    metrics.get("stationarity_tolerance", np.nan)
                ),
                "termination_reason": str(metrics.get("termination_reason", "")),
                "simplex_polish_sweeps": int(
                    metrics.get("simplex_polish_sweeps", 0)
                ),
                "rounds": int(metrics.get("rounds", 0)),
                "initialization": str(metrics.get("initialization", "unknown")),
                "selected_ris_sweep": str(metrics.get("selected_ris_sweep", "")),
                "objective_history": json.dumps(metrics.get("objective_history", [])),
                "grid": json.dumps(metrics.get("grid", {})),
                "power_stationarity_gap": float(metrics.get("power_stationarity_gap", np.nan)),
                "common_stationarity_gap": float(metrics.get("common_stationarity_gap", np.nan)),
                "solve_ms": float(solve_ms),
                "pc": float(powers[0]),
                "p1": float(powers[1]),
                "p2": float(powers[2]),
                "p3": float(powers[3]),
                "p4": float(powers[4]),
                "eta1": float(common[0]),
                "eta2": float(common[1]),
                "eta3": float(common[2]),
                "eta4": float(common[3]),
                "config_hash": cfg.config_hash(),
                "git_commit": git_commit,
                "bank_checksum": checksum,
                "freeze": (
                    f"{AO_VERSION}:max_iter={FROZEN_MAX_ITER}:"
                    f"pairwise_probe={FROZEN_PAIRWISE_PROBE}:"
                    f"stationarity_tol={FROZEN_STATIONARITY_TOL}:"
                    "post_ris_simplex_polish=40"
                    if args.method == "ao_sca"
                    else f"{GRID_VERSION}:rounds={FROZEN_ROUNDS}"
                    if args.method == "ao_grid"
                    else "analytical_ris_unchanged"
                ),
            }
        )
        completed = scenario - args.start + 1
        if completed % 10 == 0 or scenario + 1 == args.end:
            print(
                f"{args.method} N={args.n_ris} {scenario + 1}/{args.end} "
                f"R={rows[-1]['sum_rate']:.6f} solve_ms={solve_ms:.2f}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
