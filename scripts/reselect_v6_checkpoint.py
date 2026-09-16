"""Re-select the reported checkpoint under a different violation tolerance.

Training keeps the sum-rate / violation frontier of every QoS-satisfying
validation step, so changing `validation_violation_tolerance` after the fact
is a re-selection plus a test-set evaluation rather than a retrain.

Selection reads validation metrics only. The test bank is touched afterwards,
to score the checkpoint the validation set already chose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import pandas as pd
import torch

from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.experiment import evaluate_checkpoint
from star_ris_rsma.scenario_bank import ScenarioBank


def extract_archives(archives: list[Path], destination: Path) -> None:
    for archive in archives:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)


def run_directories(root: Path) -> list[Path]:
    return sorted({path.parent for path in root.rglob("summary.json")})


def candidate_rows(run: Path) -> list[dict[str, object]]:
    index = run / "candidate_checkpoints.json"
    if not index.is_file():
        raise RuntimeError(
            f"{run} has no candidate_checkpoints.json; it predates checkpoint "
            "retention and can only be re-selected by retraining"
        )
    return list(json.loads(index.read_text(encoding="utf-8"))["candidates"])


def resolve_checkpoint(run: Path, entry: dict[str, object], summary: dict) -> Path:
    """Map a validation step onto whichever file actually holds it."""
    named = run / str(entry["checkpoint"])
    if named.is_file():
        return named
    step = int(entry["eval_step"])
    best_step = int(json.loads((run / "best_validation.json").read_text())["eval_step"])
    for step_value, fallback in (
        (0, "initial.pt"),
        (int(summary["train_steps"]), "latest.pt"),
        (best_step, "best.pt"),
    ):
        if step == step_value and (run / fallback).is_file():
            return run / fallback
    raise RuntimeError(f"{run}: no stored checkpoint for validation step {step}")


def select(entries: list[dict[str, object]], tolerance: float) -> dict[str, object]:
    feasible = [
        item for item in entries if float(item["mean_violation"]) <= tolerance
    ]
    if not feasible:
        raise RuntimeError(
            f"No retained checkpoint clears tolerance {tolerance}; the frontier "
            f"starts at {min(float(i['mean_violation']) for i in entries)}"
        )
    return max(feasible, key=lambda item: float(item["mean_sum_rate"]))


def reselect_run(run: Path, tolerance: float, output: Path) -> dict[str, object]:
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    method, seed = str(summary["method"]), int(summary["seed"])
    entry = select(candidate_rows(run), tolerance)
    checkpoint = resolve_checkpoint(run, entry, summary)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig(**payload["config"])
    bank = ScenarioBank.load(cfg.test_bank_path, cfg)
    if bank.checksum() != summary["test_bank_checksum"]:
        raise RuntimeError(
            f"{run}: test bank checksum {bank.checksum()} does not match the "
            f"training record {summary['test_bank_checksum']}"
        )

    destination = output / "raw" / f"{method}_n{cfg.n_ris}_seed{seed}.csv"
    evaluate_checkpoint(method, cfg, checkpoint, bank, seed, destination)
    rows = pd.read_csv(destination)
    previous = summary["checkpoints"]["best"]
    return {
        "run": run.name,
        "method": method,
        "n_ris": cfg.n_ris,
        "seed": seed,
        "tolerance": tolerance,
        "selected_step": int(entry["eval_step"]),
        "selected_checkpoint": checkpoint.name,
        "validation_sum_rate": float(entry["mean_sum_rate"]),
        "validation_violation": float(entry["mean_violation"]),
        "previous_test_sum_rate": float(previous["sum_rate_mean"]),
        "test_sum_rate": float(rows["sum_rate"].mean()),
        "test_all_qos": float(rows["all_qos"].mean()),
        "test_violation": float(rows["violation"].mean()),
        "raw": str(destination.relative_to(output)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, required=True)
    parser.add_argument(
        "--extract-to",
        type=Path,
        help="Unpack *.zip found under --input here before re-selecting",
    )
    args = parser.parse_args()

    root = args.input
    archives = sorted(args.input.rglob("*.zip"))
    if archives:
        if args.extract_to is None:
            raise SystemExit("--extract-to is required when --input holds archives")
        extract_archives(archives, args.extract_to)
        root = args.extract_to

    runs = run_directories(root)
    if not runs:
        raise SystemExit(f"No training runs found under {root}")
    args.output.mkdir(parents=True, exist_ok=True)

    records = [reselect_run(run, args.tolerance, args.output) for run in runs]
    changed = [r for r in records if r["test_sum_rate"] != r["previous_test_sum_rate"]]
    manifest = {
        "validation_violation_tolerance": args.tolerance,
        "runs": len(records),
        "changed_selection": len(changed),
        "selection_source": "validation metrics only",
        "records": sorted(
            records, key=lambda r: (str(r["method"]), int(r["n_ris"]), int(r["seed"]))
        ),
    }
    (args.output / "RESELECTION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    pd.DataFrame(manifest["records"]).to_csv(
        args.output / "RESELECTION_SUMMARY.csv", index=False
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
