from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

from star_ris_rsma.baselines.analytical_ris import solve as solve_analytical
from star_ris_rsma.baselines.ao_corrected import FROZEN_MAX_ITER, solve as solve_ao
from star_ris_rsma.baselines.ao_grid_corrected import FROZEN_ROUNDS, solve as solve_grid
from star_ris_rsma.checkpoints import load_checkpoint
from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.env import StarRisRsmaEnv
from star_ris_rsma.scenario_bank import generate_bank

METHODS = ("td3", "ddpg", "ppo", "ao_sca", "ao_grid", "analytical_ris")
LEARNED = {"td3", "ddpg", "ppo"}

# The canonical metrics were produced on different CPU/PyTorch builds.  Tiny
# floating-point drift is expected when replaying an otherwise identical
# checkpoint, especially after the action projection and rate calculation.
# Keep this strict enough to catch a wrong checkpoint/config while allowing
# the few-ppm drift observed across GitHub/Kaggle runners.
CHECKPOINT_VERIFY_RTOL = 1e-5
CHECKPOINT_VERIFY_ATOL = 1e-7


def canonical_reference(path: Path, method: str, n_ris: int, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    subset = frame[
        (frame.method.astype(str).str.lower() == method)
        & (frame.n_ris.astype(int) == int(n_ris))
        & (frame.seed.astype(int) == int(seed))
    ].copy()
    if len(subset) != 1000:
        raise RuntimeError(
            f"Canonical DRL raw coverage mismatch for {method} N={n_ris} seed={seed}: {len(subset)}"
        )
    return subset.sort_values("scenario").reset_index(drop=True)


def policy_action(agent, method: str, obs: np.ndarray):
    if method == "ppo":
        action, _, _ = agent.act(obs, deterministic=True)
        return action
    return agent.act(obs, noise_std=0.0)


def verify_checkpoint(
    agent,
    method: str,
    env: StarRisRsmaEnv,
    bank,
    reference: pd.DataFrame,
    count: int,
) -> None:
    for scenario in range(count):
        obs = env.reset(channel=bank.channel(scenario))
        raw = policy_action(agent, method, obs)
        metrics = env.evaluate_raw_action(raw)
        row = reference.iloc[scenario]
        for key in ("sum_rate", "qos_fraction", "violation"):
            observed = float(metrics[key])
            expected = float(row[key])
            if not np.isclose(
                observed,
                expected,
                rtol=CHECKPOINT_VERIFY_RTOL,
                atol=CHECKPOINT_VERIFY_ATOL,
            ):
                raise RuntimeError(
                    f"Checkpoint compatibility failed for {method} scenario={scenario} {key}: "
                    f"observed={observed}, canonical={expected}"
                )
        if bool(metrics["all_qos"]) != bool(row["all_qos"]):
            raise RuntimeError(
                f"Checkpoint compatibility failed for {method} scenario={scenario} all_qos"
            )
    print(f"Verified {method} checkpoint against {count} canonical scenarios.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--verify-count", type=int, default=5)
    parser.add_argument("--reference-drl-raw", type=Path, default=Path("results/six_method_v1/raw/DRL_TEST_RAW_ALL.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    cfg = ExperimentConfig.from_yaml(args.config)
    bank = generate_bank(cfg, 1000, 33001, "test")
    expected_checksums = pd.read_csv(
        args.reference_drl_raw, usecols=["n_ris", "bank_checksum"]
    )
    checksums = expected_checksums.loc[
        expected_checksums.n_ris.astype(int) == int(cfg.n_ris), "bank_checksum"
    ].dropna().astype(str).unique()
    if len(checksums) != 1 or bank.checksum() != checksums[0]:
        raise RuntimeError(
            f"Latency bank mismatch N={cfg.n_ris}: generated={bank.checksum()}, canonical={checksums}"
        )

    env = StarRisRsmaEnv(cfg, args.seed)
    learned = args.method in LEARNED
    agent = None
    if learned:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for learned methods")
        agent, _ = load_checkpoint(
            args.checkpoint,
            args.method,
            env.observation_dim,
            env.action_dim,
            cfg,
            "cpu",
        )
        reference = canonical_reference(
            args.reference_drl_raw, args.method, cfg.n_ris, args.seed
        )
        verify_checkpoint(
            agent,
            args.method,
            env,
            bank,
            reference,
            min(args.verify_count, len(reference)),
        )

    limit = min(len(bank), args.warmup + args.count)
    rows: list[dict[str, object]] = []
    for scenario in range(limit):
        obs = env.reset(channel=bank.channel(scenario))
        started = time.perf_counter_ns()

        if learned:
            raw = policy_action(agent, args.method, obs)
            inference_ms = (time.perf_counter_ns() - started) / 1e6
            evaluation_started = time.perf_counter_ns()
            env.evaluate_raw_action(raw)
            evaluation_ms = (time.perf_counter_ns() - evaluation_started) / 1e6
            solve_ms = inference_ms + evaluation_ms
        else:
            if args.method == "ao_sca":
                solve_ao(env, seed=args.seed + scenario, max_iter=FROZEN_MAX_ITER)
            elif args.method == "ao_grid":
                solve_grid(env, seed=args.seed + scenario, rounds=FROZEN_ROUNDS)
            else:
                solve_analytical(env)
            solve_ms = (time.perf_counter_ns() - started) / 1e6
            inference_ms = float("nan")
            evaluation_ms = float("nan")

        if scenario >= args.warmup:
            rows.append(
                {
                    "method": args.method,
                    "n_ris": int(cfg.n_ris),
                    "scenario": scenario - args.warmup,
                    "inference_ms": inference_ms,
                    "evaluation_ms": evaluation_ms,
                    "solve_ms": solve_ms,
                    "cpu_threads": 1,
                    "seed": int(args.seed),
                    "config_hash": cfg.config_hash(),
                    "bank_checksum": bank.checksum(),
                    "latency_protocol": "single_thread_same_runner_warmup10_count100_v2",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
