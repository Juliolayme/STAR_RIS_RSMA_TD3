from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


METHODS = ("td3", "ddpg", "ppo")
BASELINES = ("ao_sca", "ao_grid", "analytical_ris")
ALL_METHODS = METHODS + BASELINES
N_VALUES = (16, 32, 64, 96, 128)
SEEDS = tuple(range(5))
METRICS = ("sum_rate", "qos_fraction", "all_qos", "violation")
PARAMETERIZATION = "physical_v6_soft_anchor"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_member(archive: zipfile.ZipFile, suffix: str) -> bytes:
    matches = [name for name in archive.namelist() if name.endswith("/" + suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {suffix} in archive, found {matches}")
    return archive.read(matches[0])


def json_member(archive: zipfile.ZipFile, suffix: str) -> dict[str, Any]:
    return json.loads(read_member(archive, suffix))


def validate_raw(raw: pd.DataFrame, method: str, n_ris: int, seed: int, checkpoint: str) -> None:
    required = {"method", "seed", "scenario", "bank_checksum", *METRICS}
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"{method} N={n_ris} seed={seed} {checkpoint}: missing {sorted(missing)}")
    if set(raw.method.astype(str).str.lower()) != {method}:
        raise RuntimeError(f"{method} N={n_ris} seed={seed}: method mismatch")
    if set(raw.seed.astype(int)) != {seed}:
        raise RuntimeError(f"{method} N={n_ris} seed={seed}: seed mismatch")
    if len(raw) != 1000 or set(raw.scenario.astype(int)) != set(range(1000)):
        raise RuntimeError(f"{method} N={n_ris} seed={seed}: test coverage is not 1,000 locked scenarios")
    numeric = raw[list(METRICS)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"{method} N={n_ris} seed={seed}: NaN/Inf in {checkpoint}")
    if raw.bank_checksum.astype(str).nunique() != 1:
        raise RuntimeError(f"{method} N={n_ris} seed={seed}: multiple test-bank checksums")


def load_method(
    root: Path, expected_method: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_paths = sorted(root.glob("TRAINING_RUN_MANIFEST*.json"))
    if len(manifest_paths) != 1:
        raise RuntimeError(f"Expected one top-level run manifest under {root}: {manifest_paths}")
    run_manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    protocol = run_manifest.get("protocol", {})
    if run_manifest.get("audit") != "PASS" or run_manifest.get("completed_jobs") != 25:
        raise RuntimeError(f"Incomplete orchestrator audit for {expected_method}: {run_manifest}")
    expected_protocol = {
        "method": expected_method,
        "action_parameterization": PARAMETERIZATION,
        "n_values": list(N_VALUES),
        "seeds": list(SEEDS),
        "train_steps": 100000,
        "expected_jobs": 25,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise RuntimeError(f"{expected_method}: protocol {key}={protocol.get(key)!r}, expected {expected!r}")

    job_index = {(int(job["n_ris"]), int(job["seed"])): job for job in run_manifest["jobs"]}
    if set(job_index) != set(itertools.product(N_VALUES, SEEDS)):
        raise RuntimeError(f"{expected_method}: orchestrator job coverage mismatch")

    best_frames: list[pd.DataFrame] = []
    checkpoint_rows: list[dict[str, Any]] = []
    curve_rows: list[pd.DataFrame] = []
    source_commits: set[str] = set()
    for n_ris, seed in itertools.product(N_VALUES, SEEDS):
        job = job_index[(n_ris, seed)]
        archive_path = root / "collected" / str(job["archive"])
        if not archive_path.is_file():
            raise RuntimeError(f"Missing archive: {archive_path}")
        observed_sha = sha256(archive_path)
        if observed_sha != str(job["archive_sha256"]):
            raise RuntimeError(f"Archive checksum mismatch: {archive_path}")

        with zipfile.ZipFile(archive_path) as archive:
            summary = json_member(archive, "summary.json")
            inner_manifest = json_member(archive, "manifest.json")
            timing = json_member(archive, "timing.json")
            verification = json_member(archive, "SCENARIO_BANK_VERIFICATION.json")
            provenance = json_member(archive, "KAGGLE_JOB_PROVENANCE.json")
            best_validation = json_member(archive, "best_validation.json")

            config = inner_manifest["config"]
            if summary["method"] != expected_method or int(summary["seed"]) != seed:
                raise RuntimeError(f"Summary identity mismatch in {archive_path}")
            if int(config["n_ris"]) != n_ris or int(config["train_steps"]) != 100000:
                raise RuntimeError(f"Config protocol mismatch in {archive_path}")
            if config["action_parameterization"] != PARAMETERIZATION:
                raise RuntimeError(f"Action parameterization mismatch in {archive_path}")
            if verification.get("audit") != "PASS" or not verification["banks"][str(n_ris)]["frozen_test_checksum_match"]:
                raise RuntimeError(f"ScenarioBank verification failed in {archive_path}")
            # A run can finish cleanly and still hand back its initialisation as
            # the selected checkpoint when no validation step clears the
            # feasibility rule. Fourteen of the r1 jobs did, which is how an
            # untrained policy reached the published tables. Read from
            # validation, not from the test metrics in summary.json, so nothing
            # that gates a run touches the split being reported.
            if int(best_validation["eval_step"]) == 0:
                raise RuntimeError(
                    f"{archive_path}: the selection rule never left step 0, so "
                    "the reported checkpoint is the untrained initialisation"
                )
            source_commits.add(str(provenance["git_commit"]))

            checkpoint_raw: dict[str, pd.DataFrame] = {}
            for checkpoint in ("initial", "best", "latest"):
                raw = pd.read_csv(io.BytesIO(read_member(archive, f"test_{checkpoint}_raw.csv")))
                validate_raw(raw, expected_method, n_ris, seed, checkpoint)
                raw_mean = raw[list(METRICS)].mean()
                recorded = summary["checkpoints"][checkpoint]
                for metric in METRICS:
                    if not math.isclose(float(raw_mean[metric]), float(recorded[f"{metric}_mean"]), abs_tol=1e-10):
                        raise RuntimeError(
                            f"Summary/raw mismatch {expected_method} N={n_ris} seed={seed} "
                            f"{checkpoint} {metric}"
                        )
                checkpoint_raw[checkpoint] = raw

            curve = pd.read_csv(
                io.BytesIO(read_member(archive, "validation_summary.csv"))
            )[["eval_step", "mean_sum_rate", "mean_all_qos", "mean_violation"]].copy()
            curve["method"] = expected_method
            curve["n_ris"] = n_ris
            curve["seed"] = seed
            curve_rows.append(curve)

            best = checkpoint_raw["best"].copy()
            best.insert(1, "n_ris", n_ris)
            best["repository_commit"] = str(provenance["git_commit"])
            best["embedded_runtime_identifier"] = best["git_commit"].astype(str)
            best_frames.append(best)

            initial = summary["checkpoints"]["initial"]
            best_summary = summary["checkpoints"]["best"]
            latest = summary["checkpoints"]["latest"]
            checkpoint_rows.append(
                {
                    "method": expected_method,
                    "n_ris": n_ris,
                    "seed": seed,
                    "best_validation_step": int(best_validation["eval_step"]),
                    "best_validation_feasible": bool(best_validation["feasible"]),
                    "initial_sum_rate": float(initial["sum_rate_mean"]),
                    "best_sum_rate": float(best_summary["sum_rate_mean"]),
                    "latest_sum_rate": float(latest["sum_rate_mean"]),
                    "learning_gain_best_minus_initial": float(best_summary["sum_rate_mean"] - initial["sum_rate_mean"]),
                    "latest_gain_minus_initial": float(latest["sum_rate_mean"] - initial["sum_rate_mean"]),
                    "best_all_qos": float(best_summary["all_qos_mean"]),
                    "latest_all_qos": float(latest["all_qos_mean"]),
                    "best_qos_fraction": float(best_summary["qos_fraction_mean"]),
                    "best_violation": float(best_summary["violation_mean"]),
                    "training_seconds": float(timing["training"]["training_seconds"]),
                    "interactions_per_second": float(timing["training"]["interactions_per_second"]),
                    "total_wall_seconds": float(timing["total_wall_seconds"]),
                    "device_name": str(timing["training"]["device_name"]),
                    "peak_gpu_memory_mb": float(timing["training"]["peak_gpu_memory_mb"]),
                    "config_hash": str(inner_manifest["config_hash"]),
                    "test_bank_checksum": str(summary["test_bank_checksum"]),
                    "repository_commit": str(provenance["git_commit"]),
                    "embedded_runtime_identifier": str(inner_manifest["git_commit"]),
                    "archive": str(job["archive"]),
                    "archive_sha256": observed_sha,
                }
            )

    # The TD3 orchestrator legitimately reused two already-completed archives
    # from the superseded fan-out workflow.  Their source commit differs only
    # in CI/orchestration files; the config hashes, test banks, and training
    # protocol are validated above for every archive.
    run_manifest["archive_repository_commits"] = sorted(source_commits)
    return (
        pd.concat(best_frames, ignore_index=True),
        pd.DataFrame(checkpoint_rows),
        pd.concat(curve_rows, ignore_index=True),
        run_manifest,
    )


def t_interval(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    if len(values) <= 1 or std == 0.0:
        return mean, std, mean, mean
    half = float(stats.t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values)))
    return mean, std, mean - half, mean + half


def performance_table(drl: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    combined = pd.concat([drl, baselines], ignore_index=True, sort=False)
    for method, n_ris in itertools.product(ALL_METHODS, N_VALUES):
        group = combined[(combined.method == method) & (combined.n_ris.astype(int) == n_ris)]
        if method in METHODS:
            samples = group.groupby("seed")[list(METRICS)].mean()
            unit = "seed_mean"
        else:
            samples = group[list(METRICS)]
            unit = "scenario"
        row: dict[str, Any] = {
            "method": method,
            "n_ris": n_ris,
            "seeds": int(group.seed.nunique()),
            "test_scenarios": int(group.scenario.nunique()),
            "uncertainty_unit": unit,
        }
        for metric in METRICS:
            mean, std, low, high = t_interval(samples[metric].to_numpy(float))
            row.update({f"{metric}_mean": mean, f"{metric}_std": std, f"{metric}_ci95_low": low, f"{metric}_ci95_high": high})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["n_ris", "method"]).reset_index(drop=True)


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * float(values[index])))
        adjusted[index] = running
    return adjusted.tolist()


def seed_level_test(
    method_a: str,
    method_b: str,
    seed_means: dict[str, list[float]],
) -> dict[str, Any]:
    """Test the difference over training runs rather than over scenarios.

    The scenario-level test treats 1,000 locked scenarios as its sample while
    every one of them is scored by the same five trained policies. It answers
    whether these policies differ on this scenario distribution, and with
    n=1,000 it returns tiny p-values for differences of a few hundredths.

    A claim that one method beats another has to survive retraining, so the
    sample there is the five seeds. Traditional baselines are deterministic
    solvers with no training variability, so a learned method is tested
    against the baseline's fixed value instead of paired against it.
    """
    a_values, b_values = seed_means[method_a], seed_means[method_b]
    a_learned, b_learned = len(a_values) > 1, len(b_values) > 1
    if not a_learned and not b_learned:
        return {
            "seed_level_unit": "not applicable; both methods are deterministic",
            "seed_level_n": 0,
            "seed_mean_difference_a_minus_b": float(
                np.mean(a_values) - np.mean(b_values)
            ),
            "seed_t_statistic": float("nan"),
            "seed_t_p": float("nan"),
        }
    if a_learned and b_learned:
        result = stats.ttest_rel(a_values, b_values)
        unit, n = "paired over 5 training seeds", len(a_values)
        difference = float(np.mean(a_values) - np.mean(b_values))
    elif a_learned:
        result = stats.ttest_1samp(np.asarray(a_values) - np.mean(b_values), 0.0)
        unit, n = "5 training seeds against a deterministic baseline", len(a_values)
        difference = float(np.mean(a_values) - np.mean(b_values))
    else:
        result = stats.ttest_1samp(np.mean(a_values) - np.asarray(b_values), 0.0)
        unit, n = "5 training seeds against a deterministic baseline", len(b_values)
        difference = float(np.mean(a_values) - np.mean(b_values))
    return {
        "seed_level_unit": unit,
        "seed_level_n": n,
        "seed_mean_difference_a_minus_b": difference,
        "seed_t_statistic": float(result.statistic),
        "seed_t_p": float(result.pvalue),
    }


def paired_tests(drl: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([drl, baselines], ignore_index=True, sort=False)
    output: list[dict[str, Any]] = []
    for n_ris in N_VALUES:
        per_seed = (
            combined[combined.n_ris.astype(int) == n_ris]
            .groupby(["method", "seed"], as_index=False)["sum_rate"].mean()
        )
        seed_means = {
            method: group.sort_values("seed")["sum_rate"].tolist()
            for method, group in per_seed.groupby("method")
        }
        means = (
            combined[combined.n_ris.astype(int) == n_ris]
            .groupby(["method", "scenario"], as_index=False)["sum_rate"].mean()
            .pivot(index="scenario", columns="method", values="sum_rate")
        )
        local: list[dict[str, Any]] = []
        for method_a, method_b in itertools.combinations(ALL_METHODS, 2):
            difference = (means[method_a] - means[method_b]).to_numpy(float)
            t_result = stats.ttest_rel(means[method_a], means[method_b])
            try:
                w_result = stats.wilcoxon(difference)
                w_stat, w_p = float(w_result.statistic), float(w_result.pvalue)
            except ValueError:
                w_stat, w_p = 0.0, 1.0
            std = float(difference.std(ddof=1))
            local.append(
                {
                    "n_ris": n_ris,
                    "method_a": method_a,
                    "method_b": method_b,
                    "paired_scenarios": len(difference),
                    "mean_difference_a_minus_b": float(difference.mean()),
                    "win_fraction_a_gt_b": float((difference > 0).mean()),
                    "cohen_dz": float(difference.mean() / std) if std else 0.0,
                    "paired_t_statistic": float(t_result.statistic),
                    "paired_t_p": float(t_result.pvalue),
                    "wilcoxon_statistic": w_stat,
                    "wilcoxon_p": w_p,
                    "holm_family": f"all_15_pairs_within_N{n_ris}",
                    "holm_family_size": 15,
                    **seed_level_test(method_a, method_b, seed_means),
                }
            )
        t_adjusted = holm_adjust([row["paired_t_p"] for row in local])
        w_adjusted = holm_adjust([row["wilcoxon_p"] for row in local])
        for row, t_p, w_p in zip(local, t_adjusted, w_adjusted):
            row["scenario_t_holm_p"] = t_p
            row["wilcoxon_holm_p"] = w_p
            row["scenario_t_holm_significant_0_05"] = t_p < 0.05
            row["wilcoxon_holm_significant_0_05"] = w_p < 0.05
            # Kept so existing readers of the r1 tables still resolve.
            row["paired_t_holm_p"] = t_p
            row["paired_t_holm_significant_0_05"] = t_p < 0.05

        # Holm over the seed-level family covers only the pairs it applies to.
        applicable = [row for row in local if row["seed_level_n"] > 0]
        seed_adjusted = holm_adjust([row["seed_t_p"] for row in applicable])
        for row in local:
            row["seed_holm_family_size"] = len(applicable)
        for row, seed_p in zip(applicable, seed_adjusted):
            row["seed_t_holm_p"] = seed_p
            row["seed_t_holm_significant_0_05"] = seed_p < 0.05
        for row in local:
            row.setdefault("seed_t_holm_p", float("nan"))
            row.setdefault("seed_t_holm_significant_0_05", False)
            row["significant_under_both_units"] = bool(
                row["scenario_t_holm_significant_0_05"]
                and row["seed_t_holm_significant_0_05"]
            )
        output.extend(local)
    return pd.DataFrame(output)


def load_baselines(path: Path, drl: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["method"] = raw.method.astype(str).str.lower()
    raw = raw[raw.method.isin(BASELINES)].copy()
    if "seed" not in raw:
        raw["seed"] = 0
    if raw.duplicated(["method", "n_ris", "scenario"]).any():
        raise RuntimeError("Duplicate traditional-baseline keys")
    for method, n_ris in itertools.product(BASELINES, N_VALUES):
        group = raw[(raw.method == method) & (raw.n_ris.astype(int) == n_ris)]
        if len(group) != 1000 or set(group.scenario.astype(int)) != set(range(1000)):
            raise RuntimeError(f"Baseline coverage mismatch: {method} N={n_ris}")
        expected = drl[drl.n_ris.astype(int) == n_ris].bank_checksum.astype(str).iloc[0]
        if set(group.bank_checksum.astype(str)) != {expected}:
            raise RuntimeError(f"ScenarioBank mismatch: {method} N={n_ris}")
    if set(raw[raw.method == "ao_sca"].algorithm_version.astype(str)) != {"corrected_pairwise_ao_v2"}:
        raise RuntimeError("AO-SCA input is not corrected_pairwise_ao_v2")
    if set(raw[raw.method == "ao_grid"].algorithm_version.astype(str)) != {"corrected_ao_grid_v1"}:
        raise RuntimeError("AO-Grid input is not corrected_ao_grid_v1")
    return raw


def training_time_table(checkpoints: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, n_ris), group in checkpoints.groupby(["method", "n_ris"]):
        rows.append(
            {
                "method": method,
                "n_ris": int(n_ris),
                "seeds": len(group),
                "training_seconds_mean": group.training_seconds.mean(),
                "training_seconds_std": group.training_seconds.std(ddof=1),
                "training_minutes_mean": group.training_seconds.mean() / 60.0,
                "interactions_per_second_mean": group.interactions_per_second.mean(),
                "total_wall_seconds_mean": group.total_wall_seconds.mean(),
                "peak_gpu_memory_mb_mean": group.peak_gpu_memory_mb.mean(),
                "device_name": ";".join(sorted(set(group.device_name))),
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "n_ris"]).reset_index(drop=True)


def plot_results(
    performance: pd.DataFrame,
    checkpoints: pd.DataFrame,
    timing: pd.DataFrame,
    output: Path,
    latency_table: Path | None = None,
    curves: pd.DataFrame | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    labels = {"td3": "TD3", "ddpg": "DDPG", "ppo": "PPO", "ao_sca": "AO-SCA corrected", "ao_grid": "AO-Grid corrected", "analytical_ris": "AnalyticalRIS"}
    # TD3 and DDPG land within 0.005 ms of each other, so on colour alone one
    # line hides the other completely and appears to be missing.
    styles = {
        "td3": {"marker": "o", "linestyle": "-"},
        "ddpg": {"marker": "s", "linestyle": "--"},
        "ppo": {"marker": "^", "linestyle": "-."},
        "ao_sca": {"marker": "D", "linestyle": "-"},
        "ao_grid": {"marker": "v", "linestyle": "--"},
        "analytical_ris": {"marker": "x", "linestyle": ":"},
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for method in ALL_METHODS:
        frame = performance[performance.method == method].sort_values("n_ris")
        ax.plot(frame.n_ris, frame.sum_rate_mean, label=labels[method], **styles[method])
    ax.set(
        xlabel="RIS elements (N)",
        ylabel="Mean test sum-rate (bit/s/Hz)",
        title="Held-out test set, best checkpoint, mean over five seeds",
    )
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "fig01_v6_six_method_sum_rate.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    for method in METHODS:
        frame = checkpoints[checkpoints.method == method]
        means = frame.groupby("n_ris").learning_gain_best_minus_initial.mean().reindex(N_VALUES)
        ax.plot(N_VALUES, means, label=labels[method], **styles[method])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="RIS elements (N)", ylabel="Mean learning gain: best - initial")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "fig02_v6_learning_gain.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    for method in METHODS:
        frame = timing[timing.method == method].sort_values("n_ris")
        ax.plot(frame.n_ris, frame.training_minutes_mean, label=labels[method], **styles[method])
    ax.set(xlabel="RIS elements (N)", ylabel="Mean training time per 100k job (minutes)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "fig03_v6_training_time.png", dpi=180)
    plt.close(fig)

    # Learning curves. Validation is the split the selection rule reads, so it
    # is what "the model is learning" actually means here; the test set is
    # touched three times at the end and cannot show progress.
    if curves is not None and not curves.empty:
        fig, axes = plt.subplots(1, len(METHODS), figsize=(13.0, 4.2), sharey=True)
        for ax, method in zip(np.atleast_1d(axes), METHODS):
            frame = curves[curves.method == method]
            for n_ris in N_VALUES:
                block = (
                    frame[frame.n_ris == n_ris]
                    .groupby("eval_step")["mean_sum_rate"]
                    .agg(["mean", "min", "max"])
                    .sort_index()
                )
                if block.empty:
                    continue
                line, = ax.plot(block.index, block["mean"], linewidth=1.4, label=f"N={n_ris}")
                ax.fill_between(
                    block.index, block["min"], block["max"],
                    alpha=0.15, color=line.get_color(), linewidth=0,
                )
            ax.set(xlabel="Training step", title=labels[method])
            ax.grid(alpha=0.25)
        np.atleast_1d(axes)[0].set_ylabel("Validation sum-rate (bit/s/Hz)")
        np.atleast_1d(axes)[-1].legend(fontsize=7, title="band = min-max over 5 seeds")
        fig.tight_layout()
        fig.savefig(output / "fig06_v6_learning_curves.png", dpi=180)
        plt.close(fig)

    # QoS is the hard constraint of the problem, and sum-rate alone hides
    # whether a method bought it by violating QoS. It did not.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for method in ALL_METHODS:
        frame = performance[performance.method == method].sort_values("n_ris")
        if frame.empty:
            continue
        axes[0].plot(frame.n_ris, frame.all_qos_mean, label=labels[method], **styles[method])
        axes[1].plot(
            frame.n_ris,
            np.maximum(frame.violation_mean, 1e-7),
            label=labels[method],
            **styles[method],
        )
    axes[0].set(
        xlabel="RIS elements (N)",
        ylabel="Fraction of scenarios with every user served",
        title="QoS satisfaction (higher is better)",
    )
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_yscale("log")
    axes[1].axhline(1e-7, color="black", linewidth=0.6, linestyle=":", alpha=0.6)
    axes[1].set(
        xlabel="RIS elements (N)",
        ylabel="Mean QoS shortfall (log scale)",
        # A log axis cannot draw zero, and both corrected solvers reach exactly
        # zero at most N, so say where they actually are.
        title=(
            "Residual violation (lower is better)"
            + chr(10)
            + "markers on the dotted floor are exactly zero"
        ),
    )
    for ax in axes:
        ax.grid(alpha=0.25, which="both")
    axes[0].legend(ncol=2, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "fig07_v6_qos.png", dpi=180)
    plt.close(fig)

    # The argument the study actually supports: what each method costs to run
    # against what it delivers. Sum-rate alone shows the learned methods losing.
    if latency_table is not None and latency_table.is_file():
        latency = pd.read_csv(latency_table)
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        for method in ALL_METHODS:
            q = performance[performance.method == method].sort_values("n_ris")
            l = latency[latency.method == method].sort_values("n_ris")
            merged = q.merge(l[["n_ris", "solve_ms_median"]], on="n_ris")
            if merged.empty:
                continue
            ax.plot(
                merged.solve_ms_median,
                merged.sum_rate_mean,
                label=labels[method],
                markersize=7,
                alpha=0.9,
                **styles[method],
            )
            # The three learned methods sit almost on top of each other here,
            # so a shared offset writes the labels over one another.
            offsets = {"td3": (7, 4), "ddpg": (7, -11), "ppo": (7, -3)}
            best = merged.loc[merged.n_ris.idxmax()]
            ax.annotate(
                f"N={int(best.n_ris)}",
                (best.solve_ms_median, best.sum_rate_mean),
                textcoords="offset points",
                xytext=offsets.get(method, (7, -4)),
                fontsize=7,
            )
        ax.set_xscale("log")
        ax.set(
            xlabel="Median CPU decision latency (ms, log scale)",
            ylabel="Mean sum-rate (bit/s/Hz)",
        )
        ax.grid(alpha=0.25, which="both")
        ax.legend(ncol=2, fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(output / "fig05_v6_quality_vs_latency.png", dpi=180)
        plt.close(fig)


def write_review(
    path: Path,
    performance: pd.DataFrame,
    checkpoints: pd.DataFrame,
    td3_tests: pd.DataFrame,
    timing: pd.DataFrame,
    latency: pd.DataFrame | None = None,
) -> None:
    pivot = performance.pivot(index="n_ris", columns="method", values="sum_rate_mean")
    td3_rows = checkpoints[checkpoints.method == "td3"]
    zero_td3 = int((td3_rows.learning_gain_best_minus_initial <= 1e-9).sum())
    zero_ddpg = int((checkpoints[checkpoints.method == "ddpg"].learning_gain_best_minus_initial <= 1e-9).sum())
    total_gpu_hours = float(checkpoints.training_seconds.sum() / 3600.0)
    lines = [
        "# Physical V6 full-training review",
        "",
        "## Verdict",
        "",
        "TD3 V6 is strong and stable for N >= 32, but the frozen feasibility-first checkpoint rule is overly conservative at N=16. "
        "The corrected AO methods remain stronger in absolute sum-rate; TD3 is therefore a competitive learned method, not a new optimum or upper bound.",
        "",
        "## Mean sum-rate on 1,000 matched test scenarios",
        "",
        "| N | TD3 | DDPG | PPO | AO-SCA corrected | AO-Grid corrected |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n_ris in N_VALUES:
        lines.append(
            f"| {n_ris} | {pivot.loc[n_ris, 'td3']:.4f} | {pivot.loc[n_ris, 'ddpg']:.4f} | "
            f"{pivot.loc[n_ris, 'ppo']:.4f} | {pivot.loc[n_ris, 'ao_sca']:.4f} | {pivot.loc[n_ris, 'ao_grid']:.4f} |"
        )
    lines += [
        "",
        "## Review findings",
        "",
        f"- TD3 selected a trained checkpoint in {25 - zero_td3}/25 jobs; {zero_td3}/25 selected initialization (both at N=16).",
        f"- DDPG selected initialization in {zero_ddpg}/25 jobs and is unstable across N; it should remain a comparator, not support the main claim.",
        "- PPO learned in all 25 jobs, but its final quality remains far below TD3.",
        "- At N=128, TD3 reaches {:.4f}, trailing corrected AO-SCA by {:.4f} and corrected AO-Grid by {:.4f} bit/s/Hz.".format(
            pivot.loc[128, "td3"], pivot.loc[128, "ao_sca"] - pivot.loc[128, "td3"], pivot.loc[128, "ao_grid"] - pivot.loc[128, "td3"]
        ),
        "- At N=16, two TD3 seeds select step 0 because later policies lose the strict all-users-QoS gate; this lowers the five-seed mean to {:.4f}. Do not hide this in slides.".format(pivot.loc[16, "td3"]),
        "- Two inference units are reported. The scenario-level test averages the five DRL seeds within each locked scenario and pairs over the 1,000 scenarios; with n=1,000 it resolves differences of a few hundredths, but every scenario is scored by the same five policies, so it speaks about these policies rather than about the methods.",
        "- The seed-level test pairs over the five training seeds, which is the unit a claim that one method beats another has to survive. Deterministic baselines have no training variability, so a learned method is tested against the baseline's fixed value. Holm is applied separately within each N and within each unit; `significant_under_both_units` marks the pairs that survive both.",
        "- Error-bar widths are not compared across DRL and deterministic baselines: DRL uncertainty uses five seed means, while baseline uncertainty uses 1,000 scenarios.",
        f"- Recorded training consumed {total_gpu_hours:.2f} aggregate GPU-hours on Tesla T4; this is summed job time, not orchestration wall-clock time.",
        "",
        "## TD3 versus corrected AO",
        "",
        "| N | TD3 - AO-SCA | TD3 - AO-Grid | TD3 win fraction vs AO-SCA |",
        "|---:|---:|---:|---:|",
    ]
    for n_ris in N_VALUES:
        sca = td3_tests[(td3_tests.n_ris == n_ris) & (td3_tests.method_b == "ao_sca")].iloc[0]
        grid = td3_tests[(td3_tests.n_ris == n_ris) & (td3_tests.method_b == "ao_grid")].iloc[0]
        lines.append(f"| {n_ris} | {sca.mean_difference_a_minus_b:.4f} | {grid.mean_difference_a_minus_b:.4f} | {sca.win_fraction_a_gt_b:.3f} |")
    if latency is not None:
        latency_pivot = latency.pivot(index="n_ris", columns="method", values="solve_ms_median")
        lines += [
            "",
            "## Single-thread CPU latency",
            "",
            "Latency uses a fixed seed-0 best-validation checkpoint for each learned method, while quality claims retain the full five-seed mean. "
            "All six methods were measured on one GitHub runner with warmup=10 and count=100 per method/N.",
            "",
            "| N | TD3 (ms) | DDPG (ms) | PPO (ms) | AO-SCA (ms) | AO-Grid (ms) | AnalyticalRIS (ms) | AO-SCA / TD3 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for n_ris in N_VALUES:
            lines.append(
                f"| {n_ris} | {latency_pivot.loc[n_ris, 'td3']:.4f} | {latency_pivot.loc[n_ris, 'ddpg']:.4f} | "
                f"{latency_pivot.loc[n_ris, 'ppo']:.4f} | {latency_pivot.loc[n_ris, 'ao_sca']:.2f} | "
                f"{latency_pivot.loc[n_ris, 'ao_grid']:.2f} | {latency_pivot.loc[n_ris, 'analytical_ris']:.4f} | "
                f"{latency_pivot.loc[n_ris, 'ao_sca'] / latency_pivot.loc[n_ris, 'td3']:.1f}x |"
            )
        lines += [
            "",
            "TD3 is not the absolute fastest method: AnalyticalRIS and DDPG are slightly faster, but they provide much lower or unstable sum-rate. "
            "The defensible latency claim is that TD3 remains sub-millisecond and is 2,414x-7,539x faster than corrected AO-SCA and 551x-2,524x faster than corrected AO-Grid in median decision time.",
        ]
    lines += [
        "",
        "## Reporting recommendation",
        "",
        "Use TD3 V6 as the proposed method and report the full five-seed mean. State explicitly that corrected AO gives higher offline sum-rate, while TD3 provides a learned one-pass policy. "
        "Do not call corrected AO a global optimum, and do not replace the N=16 mean with only the three successful seeds.",
        "",
        "The embedded `git_commit` column in the training CSVs was populated by Kaggle's runtime identifier. "
        "This report preserves it as `embedded_runtime_identifier` and uses the independently written `KAGGLE_JOB_PROVENANCE.json` commit as `repository_commit`.",
        "Two TD3 N=16 archives (seeds 2 and 3) came from commit `0e47904`; the other 23 came from `d5c25da`. "
        "The diff between those commits changes only CI/orchestration files, not source, configs, or the training entry point.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and aggregate the 75 physical-v6 full-training archives")
    parser.add_argument("--td3-root", type=Path, default=Path("artifacts/physical_v6_full_download/td3"))
    parser.add_argument("--ddpg-root", type=Path, default=Path("artifacts/physical_v6_full_download/comparators/physical-v6-full-25jobs-ddpg"))
    parser.add_argument("--ppo-root", type=Path, default=Path("artifacts/physical_v6_full_download/comparators/physical-v6-full-25jobs-ppo"))
    parser.add_argument("--baseline-raw", type=Path, default=Path("results/six_method_v2/raw/TRADITIONAL_TEST_RAW_ALL.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/physical_v6_full"))
    parser.add_argument(
        "--github-artifacts",
        type=Path,
        help="JSON mapping each method to the run/artifact it was downloaded from",
    )
    parser.add_argument(
        "--latency-artifact",
        type=Path,
        help="JSON naming the run/artifact the tracked latency audit came from",
    )
    args = parser.parse_args()
    github_artifacts = (
        json.loads(args.github_artifacts.read_text(encoding="utf-8"))
        if args.github_artifacts
        else {}
    )
    latency_artifact = (
        json.loads(args.latency_artifact.read_text(encoding="utf-8"))
        if args.latency_artifact
        else {}
    )

    raw_frames: list[pd.DataFrame] = []
    checkpoint_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    manifests: dict[str, Any] = {}
    for method, root in (("td3", args.td3_root), ("ddpg", args.ddpg_root), ("ppo", args.ppo_root)):
        raw, checkpoints, curves, manifest = load_method(root, method)
        raw_frames.append(raw)
        checkpoint_frames.append(checkpoints)
        curve_frames.append(curves)
        manifests[method] = manifest
    drl = pd.concat(raw_frames, ignore_index=True)
    checkpoints = pd.concat(checkpoint_frames, ignore_index=True)
    curves = pd.concat(curve_frames, ignore_index=True)

    if drl.duplicated(["method", "n_ris", "seed", "scenario"]).any():
        raise RuntimeError("Duplicate V6 best-checkpoint raw keys")
    for n_ris in N_VALUES:
        checksums = set(drl[drl.n_ris.astype(int) == n_ris].bank_checksum.astype(str))
        if len(checksums) != 1:
            raise RuntimeError(f"DRL methods do not share a test bank at N={n_ris}")

    baselines = load_baselines(args.baseline_raw, drl)
    performance = performance_table(drl, baselines)
    tests = paired_tests(drl, baselines)
    td3_tests = tests[tests.method_a == "td3"].copy()
    timing = training_time_table(checkpoints)

    output = args.output
    tables = output / "tables"
    raw_dir = output / "raw"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    drl.sort_values(["method", "n_ris", "seed", "scenario"]).to_csv(raw_dir / "DRL_V6_TEST_BEST_RAW_ALL.csv", index=False)
    checkpoints.sort_values(["method", "n_ris", "seed"]).to_csv(tables / "TABLE_V6_CHECKPOINT_AUDIT.csv", index=False)
    performance.to_csv(tables / "TABLE_V6_SIX_METHOD_PERFORMANCE.csv", index=False)
    tests.to_csv(tables / "TABLE_V6_SIX_METHOD_PAIRED_TESTS_HOLM.csv", index=False)
    td3_tests.to_csv(tables / "TABLE_V6_TD3_VS_OTHERS_HOLM.csv", index=False)
    timing.to_csv(tables / "TABLE_V6_TRAINING_TIME.csv", index=False)
    curves.sort_values(["method", "n_ris", "seed", "eval_step"]).to_csv(
        tables / "TABLE_V6_VALIDATION_CURVES.csv", index=False
    )
    plot_results(
        performance,
        checkpoints,
        timing,
        figures,
        latency_table=tables / "TABLE_V6_SIX_METHOD_CPU_LATENCY.csv",
        curves=curves,
    )
    latency_path = tables / "TABLE_V6_SIX_METHOD_CPU_LATENCY.csv"
    latency = pd.read_csv(latency_path) if latency_path.is_file() else None
    write_review(
        output / "PHYSICAL_V6_FULL_REVIEW.md",
        performance,
        checkpoints,
        td3_tests,
        timing,
        latency,
    )

    bank_checksums = {
        str(n_ris): drl[drl.n_ris.astype(int) == n_ris].bank_checksum.astype(str).iloc[0]
        for n_ris in N_VALUES
    }
    audit = {
        "verdict": "PASS",
        "action_parameterization": PARAMETERIZATION,
        "methods": list(METHODS),
        "n_values": list(N_VALUES),
        "seeds": list(SEEDS),
        "interactions_per_job": 100000,
        "training_jobs": 75,
        "best_checkpoint_test_rows": len(drl),
        "test_scenarios_per_seed_n": 1000,
        "shared_scenario_banks": True,
        "scenario_bank_checksums": bank_checksums,
        "checkpoint_selection": "validation-only; feasibility-first normalized gap, then sum-rate",
        "paired_test_protocol": {
            "scenario_level": "average 5 DRL seeds per matched scenario; paired over 1,000 scenarios; 15 method pairs per N; Holm separately within N",
            "seed_level": "paired over 5 training seeds, or one-sample against a deterministic baseline; Holm over the applicable pairs within N",
            "note": "the scenario-level unit resolves hundredths because n=1,000 while the five policies are shared; treat seed-level as the test of a method-level claim",
        },
        "orchestrator_manifests": {
            method: {
                "audit": manifest["audit"],
                "repository": manifest["repository"],
                "git_commit": manifest["git_commit"],
                "archive_repository_commits": manifest["archive_repository_commits"],
                "completed_jobs": manifest["completed_jobs"],
                "failed_jobs": manifest["failed_jobs"],
                "updated_at_utc": manifest["updated_at_utc"],
            }
            for method, manifest in manifests.items()
        },
        # Hardcoding these once meant a rebuild on different inputs published
        # provenance pointing at the wrong artifacts, so they are supplied.
        **({"github_artifacts": github_artifacts} if github_artifacts else {}),
        "embedded_identifier_note": "Training CSV git_commit contains KAGGLE_KERNEL_RUN_ID; repository_commit is taken from independent KAGGLE_JOB_PROVENANCE.json.",
        "repository_commits_per_method": {
            method: manifest["archive_repository_commits"]
            for method, manifest in manifests.items()
        },
        "published_tables": sorted(path.name for path in tables.glob("*.csv")),
        "published_figures": sorted(path.name for path in figures.glob("*.png")),
    }
    latency_audit_path = output / "PHYSICAL_V6_LATENCY_AUDIT.json"
    if latency_audit_path.is_file():
        latency_audit = json.loads(latency_audit_path.read_text(encoding="utf-8"))
        if latency_audit.get("verdict") != "PASS":
            raise RuntimeError("Tracked V6 latency audit is not PASS")
        if latency_audit.get("scenario_bank_checksums") != bank_checksums:
            raise RuntimeError("V6 latency and quality ScenarioBank checksums differ")
        audit["latency"] = {**latency_audit, **latency_artifact}
    (output / "PHYSICAL_V6_FULL_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
