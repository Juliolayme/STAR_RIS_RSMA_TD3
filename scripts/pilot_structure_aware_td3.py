from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.experiment import evaluate_checkpoint
from star_ris_rsma.experiment_v3 import train_drl_v3
from star_ris_rsma.scenario_bank import ScenarioBank


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "sum_rate_mean": float(frame["sum_rate"].mean()),
        "qos_fraction_mean": float(frame["qos_fraction"].mean()),
        "all_qos_mean": float(frame["all_qos"].mean()),
        "violation_mean": float(frame["violation"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=("td3", "ddpg", "ppo"), default="td3")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/structure_pilot"))
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    output = args.output_root / f"{args.tag}_seed{args.seed}"
    started = time.perf_counter()
    train_drl_v3(args.method, cfg, args.seed, output)
    train_seconds = time.perf_counter() - started

    bank = ScenarioBank.load(cfg.test_bank_path, cfg)
    checkpoints: dict[str, dict[str, float]] = {}
    for name in ("initial", "best", "latest"):
        checkpoint = output / f"{name}.pt"
        raw_path = output / f"test_{name}_raw.csv"
        evaluate_checkpoint(args.method, cfg, checkpoint, bank, args.seed, raw_path)
        checkpoints[name] = summarize(pd.read_csv(raw_path))

    summary = {
        "tag": args.tag,
        "method": args.method,
        "seed": args.seed,
        "parameterization": cfg.action_parameterization,
        "train_steps": cfg.train_steps,
        "test_bank_checksum": bank.checksum(),
        "train_seconds": train_seconds,
        "checkpoints": checkpoints,
        "learning_gain_vs_initial": (
            checkpoints["best"]["sum_rate_mean"]
            - checkpoints["initial"]["sum_rate_mean"]
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
