from __future__ import annotations

import gc
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .action import decode_action
from .baselines import analytical_ris, ao_grid, ao_sca
from .checkpoints import load_checkpoint
from .config import ExperimentConfig
from .env import StarRisRsmaEnv
from .result_validation import CORE_METRICS, coerce_core_metrics
from .scenario_bank import ScenarioBank


_METHODS = ("td3", "ao_sca", "ao_grid", "analytical_ris")
_LATENCY_COLUMNS = (
    "td3_actor_ms",
    "td3_decode_ms",
    "td3_decision_ms",
    "td3_end_to_end_ms",
    "ao_sca_end_to_end_ms",
    "ao_grid_end_to_end_ms",
    "analytical_ris_end_to_end_ms",
)


def configure_single_thread_cpu(*, pin_affinity: bool = True) -> None:
    """Disable hidden parallelism and optionally pin the process to one CPU."""
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if pin_affinity and hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        try:
            available = sorted(os.sched_getaffinity(0))
            if available:
                os.sched_setaffinity(0, {available[0]})
        except OSError:
            pass


def cpu_metadata() -> dict[str, Any]:
    model = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                model = line.split(":", 1)[1].strip()
                break
    return {
        "platform": platform.platform(),
        "processor": model,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
    }


def _elapsed_per_call_ms(start_ns: int, repeats: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1e6 / repeats


def _run_solver(
    method: str,
    env: StarRisRsmaEnv,
    *,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    if method == "ao_sca":
        return ao_sca(env, seed=seed)
    if method == "ao_grid":
        return ao_grid(env, seed=seed)
    if method == "analytical_ris":
        return analytical_ris(env)
    raise ValueError(method)


def benchmark_td3_vs_traditional(
    cfg: ExperimentConfig,
    checkpoint: str | Path,
    bank: ScenarioBank,
    *,
    seed: int,
    scenarios: int = 1_000,
    actor_repeats: int = 100,
    decode_repeats: int = 100,
    end_to_end_repeats: int = 20,
    actor_warmup: int = 500,
    solver_warmup_scenarios: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Benchmark TD3 and three traditional methods on the same CPU and channels.

    The timed boundary begins after the channel sample has been loaded. TD3
    decision latency includes actor inference and physical action decoding.
    TD3 end-to-end latency additionally includes one final metric evaluation.
    Traditional latency includes the complete solver call, including all internal
    objective evaluations and its returned final metrics. Channel generation and
    environment reset are excluded from every method.
    """
    if scenarios <= 0 or scenarios > len(bank):
        raise ValueError(f"scenarios must be in [1, {len(bank)}]")
    if min(actor_repeats, decode_repeats, end_to_end_repeats) <= 0:
        raise ValueError("timing repeats must be positive")

    configure_single_thread_cpu(pin_affinity=True)
    env = StarRisRsmaEnv(cfg, seed)
    agent, payload = load_checkpoint(
        checkpoint,
        "td3",
        env.observation_dim,
        env.action_dim,
        cfg,
        device="cpu",
    )
    agent.actor.eval()

    first_channel = bank.channel(0)
    first_obs = env.reset(channel=first_channel)
    last_action = None
    for _ in range(actor_warmup):
        last_action = agent.act(first_obs, noise_std=0.0)
    assert last_action is not None
    for _ in range(actor_warmup):
        decoded = decode_action(
            last_action,
            cfg.n_users,
            cfg.n_ris,
            cfg.p_max,
            cfg.action_parameterization,
        )
        env.evaluate_decoded_action(decoded)

    for scenario in range(min(solver_warmup_scenarios, scenarios)):
        channel = bank.channel(scenario)
        for method in ("ao_sca", "ao_grid", "analytical_ris"):
            env.reset(channel=channel)
            _run_solver(method, env, seed=seed + scenario)

    gc.collect()
    gc.disable()
    checksum = bank.checksum()
    rows: list[dict[str, Any]] = []
    try:
        for scenario in range(scenarios):
            channel = bank.channel(scenario)
            obs = env.reset(channel=channel)

            started = time.perf_counter_ns()
            for _ in range(actor_repeats):
                action = agent.act(obs, noise_std=0.0)
            actor_ms = _elapsed_per_call_ms(started, actor_repeats)

            started = time.perf_counter_ns()
            for _ in range(decode_repeats):
                decoded = decode_action(
                    action,
                    cfg.n_users,
                    cfg.n_ris,
                    cfg.p_max,
                    cfg.action_parameterization,
                )
            decode_ms = _elapsed_per_call_ms(started, decode_repeats)

            started = time.perf_counter_ns()
            for _ in range(end_to_end_repeats):
                timed_action = agent.act(obs, noise_std=0.0)
                timed_decoded = decode_action(
                    timed_action,
                    cfg.n_users,
                    cfg.n_ris,
                    cfg.p_max,
                    cfg.action_parameterization,
                )
                td3_metrics = env.evaluate_decoded_action(timed_decoded)
            td3_end_to_end_ms = _elapsed_per_call_ms(started, end_to_end_repeats)

            row: dict[str, Any] = {
                "n_ris": int(cfg.n_ris),
                "seed": int(seed),
                "scenario": int(scenario),
                "bank_checksum": checksum,
                "checkpoint_step": int(payload["step"]),
                "actor_repeats": int(actor_repeats),
                "decode_repeats": int(decode_repeats),
                "end_to_end_repeats": int(end_to_end_repeats),
                "td3_actor_ms": float(actor_ms),
                "td3_decode_ms": float(decode_ms),
                "td3_decision_ms": float(actor_ms + decode_ms),
                "td3_end_to_end_ms": float(td3_end_to_end_ms),
            }
            for metric in CORE_METRICS:
                row[f"td3_{metric}"] = float(td3_metrics[metric])

            for method in ("ao_sca", "ao_grid", "analytical_ris"):
                env.reset(channel=channel)
                started = time.perf_counter_ns()
                _, metrics = _run_solver(method, env, seed=seed + scenario)
                elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                row[f"{method}_end_to_end_ms"] = float(elapsed_ms)
                row[f"{method}_iterations"] = int(metrics.get("iterations", 0))
                row[f"{method}_evaluations"] = int(metrics.get("evaluations", 1))
                for metric in CORE_METRICS:
                    row[f"{method}_{metric}"] = float(metrics[metric])

            rows.append(row)
    finally:
        gc.enable()

    frame = pd.DataFrame(rows)
    validate_latency_frame(
        frame,
        expected_n=cfg.n_ris,
        expected_seed=seed,
        expected_rows=scenarios,
    )
    metadata = {
        **cpu_metadata(),
        "n_ris": int(cfg.n_ris),
        "seed": int(seed),
        "scenarios": int(scenarios),
        "actor_repeats": int(actor_repeats),
        "decode_repeats": int(decode_repeats),
        "end_to_end_repeats": int(end_to_end_repeats),
        "actor_warmup": int(actor_warmup),
        "solver_warmup_scenarios": int(solver_warmup_scenarios),
        "checkpoint_step": int(payload["step"]),
        "config_hash": cfg.config_hash(),
        "bank_checksum": checksum,
        "timing_boundary": "ready locked channel -> returned decision/result",
        "td3_decision_boundary": "agent.act plus physical decode",
        "td3_end_to_end_boundary": "agent.act plus decode plus one final metric evaluation",
        "traditional_boundary": "complete solver call including internal objective evaluations and final metrics",
        "excluded": ["channel generation", "ScenarioBank loading", "environment reset"],
        "methods": list(_METHODS),
    }
    return frame, metadata


def validate_latency_frame(
    frame: pd.DataFrame,
    *,
    expected_n: int | None = None,
    expected_seed: int | None = None,
    expected_rows: int | None = None,
) -> None:
    required = {
        "n_ris",
        "seed",
        "scenario",
        "bank_checksum",
        *_LATENCY_COLUMNS,
        *{f"{method}_{metric}" for method in _METHODS for metric in CORE_METRICS},
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Latency frame missing columns: {missing}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"Latency rows {len(frame)} != {expected_rows}")
    if frame["scenario"].nunique() != len(frame) or frame["scenario"].duplicated().any():
        raise ValueError("Latency frame must contain one row per unique scenario")
    if expected_n is not None and set(frame["n_ris"].astype(int)) != {int(expected_n)}:
        raise ValueError("Latency N mismatch")
    if expected_seed is not None and set(frame["seed"].astype(int)) != {int(expected_seed)}:
        raise ValueError("Latency seed mismatch")
    if frame["bank_checksum"].nunique() != 1:
        raise ValueError("Latency frame has multiple test-bank checksums")

    latency = frame.loc[:, list(_LATENCY_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    values = latency.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Latency values must be finite and strictly positive")

    for method in _METHODS:
        metrics = frame.rename(columns={f"{method}_{m}": m for m in CORE_METRICS})
        coerce_core_metrics(metrics, context=f"{method} latency quality", require_finite=True)


def summarize_latency(frame: pd.DataFrame) -> pd.DataFrame:
    validate_latency_frame(frame)
    rows: list[dict[str, Any]] = []
    for n_ris, group in frame.groupby("n_ris", sort=True):
        row: dict[str, Any] = {
            "n_ris": int(n_ris),
            "seed": int(group["seed"].iloc[0]),
            "scenarios": int(len(group)),
            "bank_checksum": str(group["bank_checksum"].iloc[0]),
        }
        for column in _LATENCY_COLUMNS:
            values = pd.to_numeric(group[column], errors="raise").to_numpy(dtype=np.float64)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_median"] = float(np.median(values))
            row[f"{column}_p95"] = float(np.quantile(values, 0.95))
            row[f"{column}_p99"] = float(np.quantile(values, 0.99))

        td3 = pd.to_numeric(group["td3_end_to_end_ms"], errors="raise").to_numpy(dtype=np.float64)
        for baseline in ("ao_sca", "ao_grid", "analytical_ris"):
            solver = pd.to_numeric(
                group[f"{baseline}_end_to_end_ms"], errors="raise"
            ).to_numpy(dtype=np.float64)
            speedup = solver / td3
            row[f"{baseline}_over_td3_ratio_of_means"] = float(np.mean(solver) / np.mean(td3))
            row[f"{baseline}_over_td3_speedup_median"] = float(np.median(speedup))
            row[f"{baseline}_over_td3_speedup_p05"] = float(np.quantile(speedup, 0.05))
            row[f"{baseline}_over_td3_speedup_p95"] = float(np.quantile(speedup, 0.95))

        for method in _METHODS:
            renamed = group.rename(columns={f"{method}_{metric}": metric for metric in CORE_METRICS})
            numeric = coerce_core_metrics(
                renamed,
                context=f"{method} latency summary",
                require_finite=True,
            )
            for metric in CORE_METRICS:
                row[f"{method}_{metric}_mean"] = float(numeric[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def write_latency_outputs(
    raw: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: str | Path,
) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    raw.to_csv(target / "LATENCY_RAW.csv", index=False)
    summarize_latency(raw).to_csv(target / "LATENCY_SUMMARY.csv", index=False)
    (target / "LATENCY_METADATA.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
