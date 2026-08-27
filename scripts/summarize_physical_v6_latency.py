from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("td3", "ddpg", "ppo", "ao_sca", "ao_grid", "analytical_ris")
LEARNED = {"td3", "ddpg", "ppo"}
N_VALUES = (16, 32, 64, 96, 128)
PROTOCOL = "single_thread_same_runner_warmup10_count100_v6"


def validate(raw: pd.DataFrame, checkpoint_index: dict[str, Any]) -> None:
    required = {
        "method", "n_ris", "scenario", "source_scenario", "solve_ms",
        "cpu_threads", "seed", "bank_checksum", "latency_protocol",
        "checkpoint_step", "checkpoint_sha256", "repository_commit",
        "runner_os", "cpu_model", "python_version", "torch_version",
    }
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"Latency raw missing columns: {sorted(missing)}")
    if raw.duplicated(["method", "n_ris", "scenario"]).any():
        raise RuntimeError("Duplicate method/N/latency-scenario rows")
    if set(raw.latency_protocol.astype(str)) != {PROTOCOL}:
        raise RuntimeError("Unexpected latency protocol")
    if set(raw.cpu_threads.astype(int)) != {1}:
        raise RuntimeError("Latency was not uniformly single-threaded")
    for column in ("runner_os", "cpu_model", "python_version", "torch_version", "repository_commit"):
        if raw[column].astype(str).nunique() != 1:
            raise RuntimeError(f"All methods must use one runner/runtime: {column}")
    numeric = raw[["n_ris", "scenario", "source_scenario", "solve_ms"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all() or (numeric.solve_ms <= 0).any():
        raise RuntimeError("Invalid latency values")

    selected = {(row["method"], int(row["n_ris"])): row for row in checkpoint_index["checkpoints"]}
    for method in METHODS:
        for n_ris in N_VALUES:
            group = raw[(raw.method == method) & (raw.n_ris.astype(int) == n_ris)]
            if len(group) != 100 or set(group.scenario.astype(int)) != set(range(100)):
                raise RuntimeError(f"Latency coverage mismatch: {method} N={n_ris}")
            if set(group.source_scenario.astype(int)) != set(range(10, 110)):
                raise RuntimeError(f"Warmup/source-scenario mismatch: {method} N={n_ris}")
            if group.bank_checksum.astype(str).nunique() != 1:
                raise RuntimeError(f"Multiple bank checksums: {method} N={n_ris}")
            if method in LEARNED:
                expected = selected[(method, n_ris)]
                if set(group.checkpoint_sha256.astype(str)) != {expected["checkpoint_sha256"]}:
                    raise RuntimeError(f"Checkpoint SHA mismatch: {method} N={n_ris}")
                if set(group.checkpoint_step.astype(int)) != {int(expected["checkpoint_step"])}:
                    raise RuntimeError(f"Checkpoint step mismatch: {method} N={n_ris}")
            else:
                if set(group.checkpoint_sha256.astype(str)) != {"not_applicable"}:
                    raise RuntimeError(f"Traditional method has checkpoint metadata: {method} N={n_ris}")

    for n_ris in N_VALUES:
        if raw[raw.n_ris.astype(int) == n_ris].bank_checksum.astype(str).nunique() != 1:
            raise RuntimeError(f"Methods do not share ScenarioBank at N={n_ris}")


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, n_ris), group in raw.groupby(["method", "n_ris"]):
        values = group.solve_ms.to_numpy(float)
        inference = group.inference_ms.dropna().to_numpy(float)
        evaluation = group.evaluation_ms.dropna().to_numpy(float)
        rows.append(
            {
                "method": method,
                "n_ris": int(n_ris),
                "count": len(values),
                "solve_ms_mean": values.mean(),
                "solve_ms_std": values.std(ddof=1),
                "solve_ms_median": np.median(values),
                "solve_ms_p95": np.percentile(values, 95),
                "solve_ms_p99": np.percentile(values, 99),
                "solve_ms_min": values.min(),
                "solve_ms_max": values.max(),
                "inference_ms_mean": inference.mean() if len(inference) else np.nan,
                "evaluation_ms_mean": evaluation.mean() if len(evaluation) else np.nan,
                "checkpoint_step": int(group.checkpoint_step.iloc[0]),
                "checkpoint_sha256": str(group.checkpoint_sha256.iloc[0]),
                "cpu_model": str(group.cpu_model.iloc[0]),
                "runner_os": str(group.runner_os.iloc[0]),
                "python_version": str(group.python_version.iloc[0]),
                "torch_version": str(group.torch_version.iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_ris", "method"]).reset_index(drop=True)


def speedups(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot(index="n_ris", columns="method", values="solve_ms_median")
    rows: list[dict[str, Any]] = []
    for n_ris in N_VALUES:
        for baseline in ("ao_sca", "ao_grid", "analytical_ris"):
            rows.append(
                {
                    "n_ris": n_ris,
                    "learned_method": "td3",
                    "baseline_method": baseline,
                    "td3_median_ms": pivot.loc[n_ris, "td3"],
                    "baseline_median_ms": pivot.loc[n_ris, baseline],
                    "baseline_over_td3_speedup": pivot.loc[n_ris, baseline] / pivot.loc[n_ris, "td3"],
                }
            )
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, target: Path) -> None:
    labels = {"td3": "TD3", "ddpg": "DDPG", "ppo": "PPO", "ao_sca": "AO-SCA corrected", "ao_grid": "AO-Grid corrected", "analytical_ris": "AnalyticalRIS"}
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for method in METHODS:
        frame = summary[summary.method == method].sort_values("n_ris")
        ax.plot(frame.n_ris, frame.solve_ms_median, marker="o", label=labels[method])
    ax.set_yscale("log")
    ax.set(xlabel="RIS elements (N)", ylabel="Median CPU decision latency (ms, log scale)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and summarize physical V6 six-method latency")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.input.glob("*.csv"))
    if len(paths) != 30:
        raise RuntimeError(f"Expected 30 latency CSV files, got {len(paths)}")
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    checkpoint_index = json.loads(args.checkpoint_index.read_text(encoding="utf-8"))
    if checkpoint_index.get("audit") != "PASS":
        raise RuntimeError("Checkpoint selection audit is not PASS")
    validate(raw, checkpoint_index)
    table = summarize(raw)
    speedup = speedups(table)

    raw_dir = args.output / "raw"
    tables = args.output / "tables"
    figures = args.output / "figures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    raw.sort_values(["method", "n_ris", "scenario"]).to_csv(raw_dir / "CPU_LATENCY_V6_RAW_ALL.csv", index=False)
    table.to_csv(tables / "TABLE_V6_SIX_METHOD_CPU_LATENCY.csv", index=False)
    speedup.to_csv(tables / "TABLE_V6_TD3_LATENCY_SPEEDUP.csv", index=False)
    (args.output / "LATENCY_CHECKPOINT_INDEX.json").write_text(
        json.dumps(checkpoint_index, indent=2, sort_keys=True), encoding="utf-8"
    )
    plot(table, figures / "fig04_v6_six_method_cpu_latency.png")

    audit = {
        "verdict": "PASS",
        "protocol": PROTOCOL,
        "methods": list(METHODS),
        "n_values": list(N_VALUES),
        "warmup": 10,
        "count_per_method_n": 100,
        "raw_rows": len(raw),
        "cpu_threads": 1,
        "same_runner": True,
        "runner_os": str(raw.runner_os.iloc[0]),
        "cpu_model": str(raw.cpu_model.iloc[0]),
        "python_version": str(raw.python_version.iloc[0]),
        "torch_version": str(raw.torch_version.iloc[0]),
        "checkpoint_selection": "fixed seed-0 best-validation checkpoint for latency; quality claims retain five-seed means",
        "scenario_bank_checksums": {
            str(n_ris): str(raw[raw.n_ris.astype(int) == n_ris].bank_checksum.iloc[0])
            for n_ris in N_VALUES
        },
        "published_tables": ["TABLE_V6_SIX_METHOD_CPU_LATENCY.csv", "TABLE_V6_TD3_LATENCY_SPEEDUP.csv"],
        "published_figure": "fig04_v6_six_method_cpu_latency.png",
        "github_run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
        "repository_commit": os.environ.get("GITHUB_SHA", "unknown"),
    }
    (args.output / "PHYSICAL_V6_LATENCY_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
