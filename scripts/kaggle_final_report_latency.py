from __future__ import annotations

"""Fair CPU latency rerun and quality-latency figures for Kaggle notebook 06."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kaggle_final_report_common import (
    LATENCY_DIR,
    METHODS,
    N_VALUES,
    REPO_DIR,
    create_locked_banks,
    ensure_finite,
    run_command,
)
from kaggle_final_report_quality import save_figure


def locate_seed0_checkpoint(stage_roots: dict[str, Path], n_ris: int) -> Path:
    """Locate the single retained seed-0 best checkpoint for one N."""
    stage_id = "td3_low_n" if n_ris in (16, 32, 64) else "td3_high_n"
    candidates = list(stage_roots[stage_id].rglob(f"N{n_ris}/seed_0/train/best.pt"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one seed-0 checkpoint N={n_ris}: {candidates}")
    return candidates[0]


def run_fair_cpu_latency(
    stage_roots: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    """Benchmark all four methods sequentially on the same 1,000 scenarios/N.

    PyTorch and every supported BLAS backend are limited to one CPU thread. The
    repository benchmark starts timing after channel loading and excludes reset
    and ScenarioBank I/O uniformly for all methods.
    """
    create_locked_banks(N_VALUES)
    raw_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    for n_ris in N_VALUES:
        config = REPO_DIR / "configs" / "v3" / f"constrained_action_n{n_ris}.yaml"
        bank = REPO_DIR / "artifacts" / "scenario_banks" / f"N{n_ris}_test.npz"
        checkpoint = locate_seed0_checkpoint(stage_roots, n_ris)
        output = LATENCY_DIR / f"N{n_ris}"
        raw_path = output / "LATENCY_RAW.csv"
        metadata_path = output / "LATENCY_METADATA.json"

        if not raw_path.exists():
            run_command(
                [
                    sys.executable,
                    "scripts/run_latency_benchmark.py",
                    "--config",
                    config,
                    "--checkpoint",
                    checkpoint,
                    "--bank",
                    bank,
                    "--seed",
                    "0",
                    "--scenarios",
                    "1000",
                    "--actor-repeats",
                    "100",
                    "--decode-repeats",
                    "100",
                    "--end-to-end-repeats",
                    "20",
                    "--actor-warmup",
                    "500",
                    "--solver-warmup-scenarios",
                    "2",
                    "--output-dir",
                    output,
                ],
                cwd=REPO_DIR,
                log_path=output / "latency.log",
                extra_env={
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                },
            )

        raw = pd.read_csv(raw_path)
        scenarios = sorted(pd.to_numeric(raw["scenario"]).astype(int).tolist())
        if len(raw) != 1000 or scenarios != list(range(1000)):
            raise RuntimeError(f"Incomplete latency output for N={n_ris}")
        if set(raw["seed"].astype(int)) != {0}:
            raise RuntimeError(f"Latency checkpoint seed mismatch for N={n_ris}")
        if raw["bank_checksum"].astype(str).nunique() != 1:
            raise RuntimeError(f"Multiple latency bank checksums for N={n_ris}")

        latency_columns = [
            "td3_end_to_end_ms",
            "ao_sca_end_to_end_ms",
            "ao_grid_end_to_end_ms",
            "analytical_ris_end_to_end_ms",
        ]
        ensure_finite(raw, latency_columns, f"latency N={n_ris}")
        if (raw[latency_columns].apply(pd.to_numeric) <= 0).any().any():
            raise RuntimeError(f"Non-positive latency for N={n_ris}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("torch_num_threads", -1)) != 1:
            raise RuntimeError(f"PyTorch was not single-threaded for N={n_ris}")
        if metadata.get("timing_boundary") != "ready locked channel -> returned decision/result":
            raise RuntimeError(f"Unexpected timing boundary for N={n_ris}")
        metadata_rows.append(metadata)
        raw_frames.append(raw)

        row: dict[str, object] = {
            "n_ris": n_ris,
            "scenarios": 1000,
            "bank_checksum": str(raw["bank_checksum"].iloc[0]),
        }
        method_columns = {
            "td3": "td3_end_to_end_ms",
            "ao_sca": "ao_sca_end_to_end_ms",
            "ao_grid": "ao_grid_end_to_end_ms",
            "analytical_ris": "analytical_ris_end_to_end_ms",
        }
        for method, column in method_columns.items():
            values = pd.to_numeric(raw[column], errors="raise").to_numpy(float)
            row[f"{method}_mean_ms"] = float(np.mean(values))
            row[f"{method}_median_ms"] = float(np.median(values))
            row[f"{method}_std_ms"] = float(np.std(values, ddof=1))
            row[f"{method}_p95_ms"] = float(np.quantile(values, 0.95))
            row[f"{method}_p99_ms"] = float(np.quantile(values, 0.99))
        row["ao_sca_over_td3"] = row["ao_sca_mean_ms"] / row["td3_mean_ms"]
        row["ao_grid_over_td3"] = row["ao_grid_mean_ms"] / row["td3_mean_ms"]
        row["analytical_ris_over_td3"] = (
            row["analytical_ris_mean_ms"] / row["td3_mean_ms"]
        )
        summary_rows.append(row)

    raw_all = pd.concat(raw_frames, ignore_index=True)
    if len(raw_all) != 5000:
        raise RuntimeError(f"Expected 5,000 latency rows, found {len(raw_all)}")
    return raw_all, pd.DataFrame(summary_rows), metadata_rows


def plot_latency(latency_summary: pd.DataFrame, final_table: pd.DataFrame) -> None:
    """Create CPU latency, AO/TD3 speedup, and quality-latency trade-off figures."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method in METHODS:
        ax.plot(
            latency_summary["n_ris"],
            latency_summary[f"{method}_mean_ms"],
            marker="o",
            label=method,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Number of STAR-RIS elements, N")
    ax.set_ylabel("Mean end-to-end CPU latency (ms)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, "fig10_cpu_latency")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(
        latency_summary["n_ris"],
        latency_summary["ao_sca_over_td3"],
        marker="o",
        label="AO-SCA / TD3",
    )
    ax.plot(
        latency_summary["n_ris"],
        latency_summary["ao_grid_over_td3"],
        marker="o",
        label="AO-Grid / TD3",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Number of STAR-RIS elements, N")
    ax.set_ylabel("Latency ratio")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, "fig11_td3_speedup")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    latency_sorted = latency_summary.sort_values("n_ris")
    for method in METHODS:
        quality = final_table[final_table["method"] == method].sort_values("n_ris")
        latency = latency_sorted[f"{method}_mean_ms"]
        ax.scatter(latency, quality["sum_rate_mean"], label=method)
        for x_value, y_value, n_ris in zip(
            latency, quality["sum_rate_mean"], quality["n_ris"]
        ):
            ax.annotate(f"N={int(n_ris)}", (x_value, y_value), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Mean end-to-end CPU latency (ms)")
    ax.set_ylabel("Mean sum-rate")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, "fig12_quality_latency_tradeoff")
