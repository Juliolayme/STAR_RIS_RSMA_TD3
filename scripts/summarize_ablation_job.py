from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CORE_COLUMNS = ("reward", "sum_rate", "qos_fraction", "all_qos", "violation")


def _finite_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    selected = [column for column in columns if column in frame.columns]
    if not selected:
        return True
    numeric = frame[selected].apply(pd.to_numeric, errors="coerce")
    return bool(np.isfinite(numeric.to_numpy(dtype=float)).all())


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean())


def _std(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def summarize_seed(root: Path, arm: str, n_ris: int, seed: int) -> dict[str, object]:
    train_dir = root / "train" / f"seed_{seed}"
    test_path = root / "test" / f"seed_{seed}.csv"
    training_path = train_dir / "training.csv"
    validation_path = train_dir / "validation_raw.csv"
    manifest_path = train_dir / "manifest.json"

    row: dict[str, object] = {
        "arm": arm,
        "n_ris": n_ris,
        "seed": seed,
        "complete": False,
        "finite": True,
    }
    issues: list[str] = []

    training = None
    if training_path.exists():
        training = pd.read_csv(training_path)
        row["training_rows"] = len(training)
        if "step" in training.columns and len(training):
            row["training_last_step"] = int(pd.to_numeric(training["step"], errors="coerce").max())
        row["training_finite"] = _finite_columns(training, ("reward", "sum_rate"))
        row["finite"] = bool(row["finite"]) and bool(row["training_finite"])
    else:
        issues.append("missing training.csv")

    validation = None
    if validation_path.exists():
        validation = pd.read_csv(validation_path)
        row["validation_rows"] = len(validation)
        row["validation_finite"] = _finite_columns(validation, CORE_COLUMNS)
        row["finite"] = bool(row["finite"]) and bool(row["validation_finite"])
        if "eval_step" in validation.columns and not validation.empty:
            grouped = validation.groupby("eval_step", sort=True)
            rewards = grouped["reward"].mean() if "reward" in validation.columns else pd.Series(dtype=float)
            if not rewards.empty:
                row["validation_points"] = len(rewards)
                row["validation_best_step"] = int(rewards.idxmax())
                row["validation_best_reward_recomputed"] = float(rewards.max())
    else:
        issues.append("missing validation_raw.csv")

    manifest: dict[str, object] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = manifest.get("config", {}) or {}
            row.update({
                "config_hash": manifest.get("config_hash"),
                "git_commit": manifest.get("git_commit"),
                "manifest_best_validation_reward": manifest.get("best_validation_reward"),
                "hidden_dim": config.get("hidden_dim"),
                "train_steps_config": config.get("train_steps"),
                "warmup_steps": config.get("warmup_steps"),
                "replay_size": config.get("replay_size"),
                "observation_normalization": config.get("observation_normalization", "global_l2"),
                "qos_penalty_linear": config.get("qos_penalty_linear", 2.0),
                "qos_penalty_quadratic": config.get("qos_penalty_quadratic", 0.0),
                "td3_layer_norm": config.get("td3_layer_norm", False),
                "td3_critic_loss": config.get("td3_critic_loss", "mse"),
            })
            scenario_banks = manifest.get("scenario_banks", {}) or {}
            for split in ("train", "validation", "test"):
                if split in scenario_banks:
                    row[f"{split}_bank_checksum"] = scenario_banks[split].get("checksum")
        except Exception as exc:  # pragma: no cover - defensive audit path
            issues.append(f"invalid manifest.json: {exc}")
    else:
        issues.append("missing manifest.json")

    test = None
    if test_path.exists():
        test = pd.read_csv(test_path)
        row["test_rows"] = len(test)
        row["test_finite"] = _finite_columns(test, CORE_COLUMNS)
        row["finite"] = bool(row["finite"]) and bool(row["test_finite"])
        for column in CORE_COLUMNS:
            row[f"test_mean_{column}"] = _mean(test, column)
            row[f"test_std_{column}"] = _std(test, column)
        if "scenario" in test.columns:
            row["test_unique_scenarios"] = int(test["scenario"].nunique())
            if row["test_unique_scenarios"] != len(test):
                issues.append("duplicate test scenarios")
        if "bank_checksum" in test.columns:
            checksums = test["bank_checksum"].dropna().astype(str).unique()
            row["test_bank_checksum"] = checksums[0] if len(checksums) == 1 else None
            if len(checksums) != 1:
                issues.append("test bank checksum not unique")
        if "config_hash" in test.columns:
            hashes = test["config_hash"].dropna().astype(str).unique()
            row["test_config_hash"] = hashes[0] if len(hashes) == 1 else None
            if len(hashes) != 1:
                issues.append("test config hash not unique")
    else:
        issues.append("missing test CSV")

    if not bool(row["finite"]):
        issues.append("non-finite core metric")
    row["complete"] = all(path.exists() for path in (training_path, validation_path, manifest_path, test_path))
    row["issues"] = "; ".join(issues)
    row["issue_count"] = len(issues)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--n-ris", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [summarize_seed(args.root, args.arm, args.n_ris, seed) for seed in args.seeds]
    frame = pd.DataFrame(rows).sort_values("seed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))

    bad = frame[(frame["complete"] != True) | (frame["finite"] != True) | (frame["issue_count"] > 0)]  # noqa: E712
    if not bad.empty:
        raise SystemExit("Per-job audit failed:\n" + bad[["seed", "issues"]].to_string(index=False))


if __name__ == "__main__":
    main()
