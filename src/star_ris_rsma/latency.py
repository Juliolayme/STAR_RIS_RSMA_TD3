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
from .baselines import ao_sca
from .checkpoints import load_checkpoint
from .config import ExperimentConfig
from .env import StarRisRsmaEnv
from .result_validation import CORE_METRICS, coerce_core_metrics
from .scenario_bank import ScenarioBank


_LATENCY_COLUMNS = ("td3_actor_ms", "td3_decode_ms", "td3_decision_ms", "ao_sca_ms")


def configure_single_thread_cpu() -> None:
    """Make timing comparable and avoid hidden BLAS/PyTorch parallelism."""
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
        # PyTorch only allows this setting before the inter-op pool starts.
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


def benchmark_td3_vs_ao_sca(
    cfg: ExperimentConfig,
    checkpoint: str | Path,
    bank: ScenarioBank,
    *,
    seed: int,
    scenarios: int = 1_000,
    actor_repeats: int = 100,
    decode_repeats: int = 100,
    actor_warmup: int = 500,
    ao_warmup_scenarios: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Benchmark decision latency from a ready channel sample.

    TD3 timing includes NumPy-to-Torch conversion, actor forward, clipping, and
    physical action decoding. AO-SCA timing includes its complete iterative solve.
    Channel generation/reset and post-decision metric evaluation are excluded from
    both methods because they are shared bookkeeping rather than decision cost.
    """
    if scenarios <= 0 or scenarios > len(bank):
        raise ValueError(f"scenarios must be in [1, {len(bank)}]")
    if actor_repeats <= 0 or decode_repeats <= 0:
        raise ValueError("timing repeats must be positive")

    configure_single_thread_cpu()
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

    first_obs = env.reset(channel=bank.channel(0))
    last_action = None
    for _ in range(actor_warmup):
        last_action = agent.act(first_obs, noise_std=0.0)
    assert last_action is not None
    for _ in range(actor_warmup):
        decode_action(
            last_action,
            cfg.n_users,
            cfg.n_ris,
            cfg.p_max,
            cfg.action_parameterization,
        )
    for scenario in range(min(ao_warmup_scenarios, scenarios)):
        env.reset(channel=bank.channel(scenario))
        ao_sca(env, seed=seed + scenario)

    gc.collect()
    gc.disable()
    rows: list[dict[str, Any]] = []
    try:
        for scenario in range(scenarios):
            obs = env.reset(channel=bank.channel(scenario))

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
            del decoded

            td3_metrics = env.evaluate_raw_action(action)

            started = time.perf_counter_ns()
            _, ao_metrics = ao_sca(env, seed=seed + scenario)
            ao_ms = (time.perf_counter_ns() - started) / 1e6

            row: dict[str, Any] = {
                "n_ris": int(cfg.n_ris),
                "seed": int(seed),
                "scenario": int(scenario),
                "bank_checksum": bank.checksum(),
                "checkpoint_step": int(payload["step"]),
                "actor_repeats": int(actor_repeats),
                "decode_repeats": int(decode_repeats),
                "td3_actor_ms": float(actor_ms),
                "td3_decode_ms": float(decode_ms),
                "td3_decision_ms": float(actor_ms + decode_ms),
                "ao_sca_ms": float(ao_ms),
            }
            for metric in CORE_METRICS:
                row[f"td3_{metric}"] = float(td3_metrics[metric])
                row[f"ao_sca_{metric}"] = float(ao_metrics[metric])
            rows.append(row)
    finally:
        gc.enable()

    frame = pd.DataFrame(rows)
    validate_latency_frame(frame, expected_n=cfg.n_ris, expected_seed=seed, expected_rows=scenarios)
    metadata = {
        **cpu_metadata(),
        "n_ris": int(cfg.n_ris),
        "seed": int(seed),
        "scenarios": int(scenarios),
        "actor_repeats": int(actor_repeats),
        "decode_repeats": int(decode_repeats),
        "actor_warmup": int(actor_warmup),
        "ao_warmup_scenarios": int(ao_warmup_scenarios),
        "checkpoint_step": int(payload["step"]),
        "config_hash": cfg.config_hash(),
        "bank_checksum": bank.checksum(),
        "timing_boundary": "ready channel -> physical action",
        "td3_boundary": "agent.act plus physical decode",
        "ao_sca_boundary": "complete iterative solver",
        "excluded": ["channel generation", "environment reset", "post-decision metric evaluation"],
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
        *{f"td3_{metric}" for metric in CORE_METRICS},
        *{f"ao_sca_{metric}" for metric in CORE_METRICS},
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

    for prefix in ("td3", "ao_sca"):
        metrics = frame.rename(columns={f"{prefix}_{m}": m for m in CORE_METRICS})
        coerce_core_metrics(metrics, context=f"{prefix} latency quality", require_finite=True)


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
        speedup = (
            pd.to_numeric(group["ao_sca_ms"], errors="raise").to_numpy(dtype=np.float64)
            / pd.to_numeric(group["td3_decision_ms"], errors="raise").to_numpy(dtype=np.float64)
        )
        row["speedup_mean_paired"] = float(np.mean(speedup))
        row["speedup_median_paired"] = float(np.median(speedup))
        row["speedup_p05_paired"] = float(np.quantile(speedup, 0.05))
        row["ratio_of_mean_latency"] = float(
            group["ao_sca_ms"].mean() / group["td3_decision_ms"].mean()
        )
        for prefix in ("td3", "ao_sca"):
            for metric in CORE_METRICS:
                row[f"{prefix}_{metric}_mean"] = float(
                    pd.to_numeric(group[f"{prefix}_{metric}"], errors="raise").mean()
                )
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
    (target / "LATENCY_METADATA.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
