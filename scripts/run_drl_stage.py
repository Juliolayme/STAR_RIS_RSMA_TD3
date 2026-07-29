from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.scenario_bank import ScenarioBank


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(command: Sequence[object], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    print("$", " ".join(map(str, command)))
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            list(map(str, command)),
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed ({return_code}); see {log_path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def config_path(n_ris: int) -> Path:
    path = REPO_ROOT / "configs" / "v3" / f"constrained_action_n{n_ris}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def ensure_banks(n_values: Sequence[int], stage_root: Path) -> dict[int, dict[str, str]]:
    bank_dir = REPO_ROOT / "artifacts" / "scenario_banks"
    bank_dir.mkdir(parents=True, exist_ok=True)
    checksums: dict[int, dict[str, str]] = {}
    for n_ris in n_values:
        expected = {
            split: bank_dir / f"N{n_ris}_{split}.npz"
            for split in ("train", "validation", "test")
        }
        if not all(path.exists() for path in expected.values()):
            run_command(
                [
                    sys.executable,
                    "scripts/create_scenario_banks.py",
                    "--config",
                    config_path(n_ris),
                    "--output-dir",
                    bank_dir,
                    "--train-count",
                    "10000",
                    "--validation-count",
                    "1000",
                    "--test-count",
                    "1000",
                ],
                stage_root / "logs" / "scenario_banks" / f"N{n_ris}.log",
            )
        cfg = ExperimentConfig.from_yaml(config_path(n_ris))
        checksums[n_ris] = {
            split: ScenarioBank.load(path, cfg).checksum()
            for split, path in expected.items()
        }
    return checksums


def valid_test_csv(path: Path, method: str, seed: int, n_ris: int) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
        required = {
            "method",
            "seed",
            "scenario",
            "sum_rate",
            "qos_fraction",
            "all_qos",
            "violation",
            "bank_checksum",
        }
        if not required.issubset(frame.columns):
            return False
        numeric = frame[["sum_rate", "qos_fraction", "all_qos", "violation"]].apply(
            pd.to_numeric, errors="coerce"
        )
        scenarios = pd.to_numeric(frame["scenario"], errors="coerce")
        return bool(
            len(frame) == 1000
            and scenarios.notna().all()
            and scenarios.nunique() == 1000
            and int(scenarios.min()) == 0
            and int(scenarios.max()) == 999
            and frame["method"].astype(str).str.lower().eq(method).all()
            and pd.to_numeric(frame["seed"], errors="coerce").eq(seed).all()
            and pd.to_numeric(frame["n_ris"], errors="coerce").eq(n_ris).all()
            and np.isfinite(numeric.to_numpy(dtype=float)).all()
            and frame["bank_checksum"].nunique() == 1
        )
    except Exception:
        return False


def train_and_evaluate(
    method: str,
    n_ris: int,
    seed: int,
    stage_root: Path,
) -> dict[str, object]:
    seed_root = stage_root / f"final_{method}_v3" / f"N{n_ris}" / f"seed_{seed}"
    train_root = seed_root / "train"
    test_csv = seed_root / "test.csv"
    seed_root.mkdir(parents=True, exist_ok=True)

    if valid_test_csv(test_csv, method, seed, n_ris) and (train_root / "best.pt").exists():
        return {"method": method, "n_ris": n_ris, "seed": seed, "status": "reused"}

    run_command(
        [
            sys.executable,
            "scripts/run_train_drl_v3.py",
            "--method",
            method,
            "--config",
            config_path(n_ris),
            "--seed",
            seed,
            "--output",
            train_root,
        ],
        seed_root / "train.log",
    )
    run_command(
        [
            sys.executable,
            "scripts/run_evaluate.py",
            "--method",
            method,
            "--config",
            config_path(n_ris),
            "--checkpoint",
            train_root / "best.pt",
            "--bank",
            REPO_ROOT / "artifacts" / "scenario_banks" / f"N{n_ris}_test.npz",
            "--seed",
            seed,
            "--output",
            test_csv,
        ],
        seed_root / "evaluate.log",
    )

    test = pd.read_csv(test_csv)
    if "n_ris" in test.columns:
        test["n_ris"] = n_ris
    else:
        test.insert(1, "n_ris", n_ris)
    test.to_csv(test_csv, index=False)

    run_command(
        [
            sys.executable,
            "scripts/audit_pilot_output.py",
            "--root",
            seed_root,
            "--seed",
            seed,
        ],
        seed_root / "audit.log",
    )
    if not valid_test_csv(test_csv, method, seed, n_ris):
        raise RuntimeError(f"Invalid test output: {test_csv}")

    latest = train_root / "latest.pt"
    if latest.exists():
        latest.unlink()
    if seed != 0:
        best = train_root / "best.pt"
        if best.exists():
            best.unlink()

    return {"method": method, "n_ris": n_ris, "seed": seed, "status": "completed"}


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def aggregate_and_plot(
    method: str,
    n_values: Sequence[int],
    seeds: Sequence[int],
    stage_root: Path,
) -> None:
    test_frames: list[pd.DataFrame] = []
    training_frames: list[pd.DataFrame] = []
    validation_frames: list[pd.DataFrame] = []
    for n_ris in n_values:
        for seed in seeds:
            train_root = stage_root / f"final_{method}_v3" / f"N{n_ris}" / f"seed_{seed}" / "train"
            test_path = train_root.parent / "test.csv"
            test = pd.read_csv(test_path)
            test_frames.append(test)

            training = pd.read_csv(train_root / "training.csv")
            training.insert(0, "method", method)
            training.insert(1, "n_ris", n_ris)
            training.insert(2, "seed", seed)
            training_frames.append(training)

            validation = pd.read_csv(train_root / "validation_summary.csv")
            validation.insert(0, "method", method)
            validation.insert(1, "n_ris", n_ris)
            validation.insert(2, "seed", seed)
            validation_frames.append(validation)

    tests = pd.concat(test_frames, ignore_index=True)
    training_all = pd.concat(training_frames, ignore_index=True)
    validation_all = pd.concat(validation_frames, ignore_index=True)
    tests.to_csv(stage_root / f"{method.upper()}_TEST_RAW_ALL.csv", index=False)
    training_all.to_csv(stage_root / f"{method.upper()}_TRAINING_RAW.csv", index=False)
    validation_all.to_csv(stage_root / f"{method.upper()}_VALIDATION_RAW.csv", index=False)

    figure_dir = stage_root / "figures"
    for metric, ylabel, source, x_column in (
        ("sum_rate", "Training sum-rate", training_all, "step"),
        ("qos_fraction", "Training QoS fraction", training_all, "step"),
        ("violation", "Training QoS violation", training_all, "step"),
        ("mean_sum_rate", "Validation sum-rate", validation_all, "eval_step"),
        ("mean_qos_fraction", "Validation QoS fraction", validation_all, "eval_step"),
        ("mean_violation", "Validation QoS violation", validation_all, "eval_step"),
    ):
        if metric not in source.columns:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for n_ris in n_values:
            subset = source[source["n_ris"] == n_ris]
            curve = subset.groupby(x_column, as_index=False)[metric].agg(["mean", "std"]).reset_index()
            x = curve[x_column].to_numpy(dtype=float)
            mean = curve["mean"].to_numpy(dtype=float)
            std = curve["std"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, mean, label=f"N={n_ris}")
            ax.fill_between(x, mean - std, mean + std, alpha=0.15)
        ax.set_xlabel("Environment interactions")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(ncol=2)
        if "violation" in metric:
            ax.set_yscale("symlog", linthresh=1e-5)
        save_figure(fig, figure_dir / f"{method}_{metric}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["td3", "ddpg", "ppo"], required=True)
    parser.add_argument("--n-values", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=2)
    args = parser.parse_args()

    n_values = tuple(dict.fromkeys(args.n_values))
    seeds = tuple(dict.fromkeys(args.seeds))
    if not n_values or not seeds:
        raise SystemExit("n-values and seeds must be non-empty")
    if set(n_values) - {16, 32, 64, 96, 128}:
        raise SystemExit(f"Unexpected N values: {n_values}")
    if seeds != tuple(range(8)):
        raise SystemExit("The final protocol requires exactly seeds 0..7")

    stage_root = args.output.resolve()
    stage_root.mkdir(parents=True, exist_ok=True)
    bank_checksums = ensure_banks(n_values, stage_root)
    jobs = [(n_ris, seed) for n_ris in n_values for seed in seeds]
    rows: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        future_map = {
            executor.submit(train_and_evaluate, args.method, n_ris, seed, stage_root): (n_ris, seed)
            for n_ris, seed in jobs
        }
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            rows.append(result)
            print("Finished", result)

    status = pd.DataFrame(rows).sort_values(["n_ris", "seed"])
    status.to_csv(stage_root / "STAGE_STATUS.csv", index=False)
    if len(status) != len(jobs):
        raise RuntimeError("Not all jobs produced a status row")

    aggregate_and_plot(args.method, n_values, seeds, stage_root)
    group = "low_n" if max(n_values) <= 64 else "high_n"
    manifest = {
        "stage_id": f"{args.method}_{group}",
        "method": args.method,
        "repository_commit": git_commit(),
        "training_protocol": "drl_v3_qos_constrained_fair",
        "n_values": list(n_values),
        "seeds": list(seeds),
        "test_scenarios_per_seed_n": 1000,
        "environment_interactions_per_seed_n": 100000,
        "scenario_bank_checksums": bank_checksums,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "output_sha256": {
            path.name: sha256_file(path)
            for path in stage_root.glob("*_RAW*.csv")
        },
    }
    (stage_root / "STAGE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
