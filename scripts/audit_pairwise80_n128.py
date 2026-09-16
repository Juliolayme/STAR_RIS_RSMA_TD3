from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from audit_ao_baselines_n128 import CHECK, cfg_bank, solve_pairwise
from star_ris_rsma.action import decode_action
from star_ris_rsma.env import StarRisRsmaEnv


def run(repo: Path, start: int, end: int, output: Path) -> None:
    cfg, bank = cfg_bank(repo)
    rows = []
    for scenario in range(start, end):
        env = StarRisRsmaEnv(cfg, seed=scenario)
        env.reset(channel=bank.channel(scenario))
        t0 = time.perf_counter()
        raw, metrics = solve_pairwise(env, seed=scenario, max_iter=80)
        elapsed = time.perf_counter() - t0
        action = decode_action(
            raw,
            cfg.n_users,
            cfg.n_ris,
            cfg.p_max,
            cfg.action_parameterization,
        )
        p = np.asarray(action.powers, dtype=float)
        eta = np.asarray(action.common_fractions, dtype=float)
        rows.append(
            dict(
                method="pairwise80",
                n_ris=128,
                scenario=scenario,
                bank_checksum=CHECK,
                sum_rate=float(metrics["sum_rate"]),
                reward=float(metrics["reward"]),
                all_qos=bool(metrics["all_qos"]),
                qos_fraction=float(metrics["qos_fraction"]),
                violation=float(metrics["violation"]),
                elapsed_s=float(elapsed),
                iterations=int(metrics.get("iterations", 0)),
                evaluations=int(metrics.get("evaluations", 0)),
                accepted_steps=int(metrics.get("accepted_steps", 0)),
                initialization=str(metrics.get("initialization", "")),
                pc=p[0], p1=p[1], p2=p[2], p3=p[3], p4=p[4],
                eta1=eta[0], eta2=eta[1], eta3=eta[2], eta4=eta[3],
                power_stationarity_gap=float(metrics.get("power_stationarity_gap", np.nan)),
                common_stationarity_gap=float(metrics.get("common_stationarity_gap", np.nan)),
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


def summarize(input_dir: Path, output_dir: Path) -> None:
    files = sorted(input_dir.glob("pairwise80_*.csv"))
    if not files:
        raise RuntimeError("No pairwise80 chunk CSV files found")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values("scenario").reset_index(drop=True)
    if len(df) != 1000 or df["scenario"].nunique() != 1000:
        raise RuntimeError(f"Expected 1000 unique scenarios, got {len(df)} rows / {df['scenario'].nunique()} unique")
    if not (df["bank_checksum"] == CHECK).all():
        raise RuntimeError("ScenarioBank checksum mismatch")

    x = df["sum_rate"].to_numpy(float)
    mean = float(x.mean())
    std = float(x.std(ddof=1))
    half = 1.96 * std / np.sqrt(len(x))
    summary = pd.DataFrame([
        dict(
            method="pairwise80",
            scenario_count=len(df),
            sum_rate_mean=mean,
            sum_rate_std=std,
            ci95_low=mean-half,
            ci95_high=mean+half,
            min=float(x.min()),
            max=float(x.max()),
            all_qos_count=int(df["all_qos"].sum()),
            pc_mean=float(df["pc"].mean()),
            common_only_count=int((df["pc"] >= 1.0-1e-9).sum()),
            elapsed_s_mean=float(df["elapsed_s"].mean()),
            evaluations_mean=float(df["evaluations"].mean()),
            iterations_mean=float(df["iterations"].mean()),
            hit_iter_80_count=int((df["iterations"] >= 80).sum()),
            hit_iter_40_count=int((df["iterations"] >= 40).sum()),
            power_gap_max=float(df["power_stationarity_gap"].max()),
            common_gap_max=float(df["common_stationarity_gap"].max()),
        )
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "PAIRWISE80_N128_1000_ALL.csv", index=False)
    summary.to_csv(output_dir / "PAIRWISE80_N128_1000_SUMMARY.csv", index=False)
    print(summary.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--repo", type=Path, required=True)
    r.add_argument("--start", type=int, required=True)
    r.add_argument("--end", type=int, required=True)
    r.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--input-dir", type=Path, required=True)
    s.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.repo, args.start, args.end, args.output)
    else:
        summarize(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
