from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


METHODS = ("td3", "ddpg", "ppo")
N_VALUES = (16, 32, 64, 96, 128)
SEED = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith("/" + suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {suffix}, found {matches}")
    return matches[0]


def load_run_manifest(root: Path) -> dict[str, Any]:
    paths = sorted(root.glob("TRAINING_RUN_MANIFEST*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one run manifest under {root}: {paths}")
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    if data.get("audit") != "PASS" or int(data.get("completed_jobs", 0)) != 25:
        raise RuntimeError(f"Training manifest is not complete: {paths[0]}")
    return data


def extract_method(root: Path, method: str, output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_run_manifest(root)
    if manifest["protocol"]["method"] != method:
        raise RuntimeError(f"Method mismatch under {root}")
    jobs = {(int(job["n_ris"]), int(job["seed"])): job for job in manifest["jobs"]}
    records: list[dict[str, Any]] = []
    for n_ris in N_VALUES:
        job = jobs[(n_ris, SEED)]
        archive_path = root / "collected" / str(job["archive"])
        archive_sha = sha256(archive_path)
        if archive_sha != str(job["archive_sha256"]):
            raise RuntimeError(f"Archive SHA-256 mismatch: {archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            summary = json.loads(archive.read(member(archive, "summary.json")))
            run = json.loads(archive.read(member(archive, "manifest.json")))
            provenance = json.loads(archive.read(member(archive, "KAGGLE_JOB_PROVENANCE.json")))
            verification = json.loads(archive.read(member(archive, "SCENARIO_BANK_VERIFICATION.json")))
            if summary["method"] != method or int(summary["seed"]) != SEED:
                raise RuntimeError(f"Summary mismatch in {archive_path}")
            if int(run["config"]["n_ris"]) != n_ris:
                raise RuntimeError(f"N mismatch in {archive_path}")
            if run["config"]["action_parameterization"] != "physical_v6_soft_anchor":
                raise RuntimeError(f"Not a physical_v6_soft_anchor checkpoint: {archive_path}")
            if verification.get("audit") != "PASS":
                raise RuntimeError(f"ScenarioBank verification failed: {archive_path}")

            target = output / f"{method}_N{n_ris}_seed0_best.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member(archive, "best.pt")))
            records.append(
                {
                    "method": method,
                    "n_ris": n_ris,
                    "seed": SEED,
                    "checkpoint": str(target),
                    "checkpoint_sha256": sha256(target),
                    "checkpoint_step": int(run["best_validation"]["eval_step"]),
                    "config_hash": str(run["config_hash"]),
                    "test_bank_checksum": str(summary["test_bank_checksum"]),
                    "repository_commit": str(provenance["git_commit"]),
                    "source_archive": str(job["archive"]),
                    "source_archive_sha256": archive_sha,
                }
            )
    return records, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audited seed-0 V6 best checkpoints for latency")
    parser.add_argument("--td3-root", type=Path, required=True)
    parser.add_argument("--ddpg-root", type=Path, required=True)
    parser.add_argument("--ppo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    for method, root in (("td3", args.td3_root), ("ddpg", args.ddpg_root), ("ppo", args.ppo_root)):
        selected, manifest = extract_method(root, method, args.output)
        records.extend(selected)
        manifests[method] = {
            "git_commit": manifest["git_commit"],
            "updated_at_utc": manifest["updated_at_utc"],
            "completed_jobs": manifest["completed_jobs"],
        }
    if len(records) != 15:
        raise RuntimeError(f"Expected 15 latency checkpoints, got {len(records)}")
    index = {
        "audit": "PASS",
        "selection": "best validation checkpoint; fixed seed 0 for latency only",
        "checkpoints": records,
        "training_manifests": manifests,
    }
    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.index.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
