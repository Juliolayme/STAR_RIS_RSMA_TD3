from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "Juliolayme/STAR_RIS_RSMA_TD3"
RELEASE_TAG = "physical-v6-r2-evidence-9547c62"
REVIEWED_SOURCE_COMMIT = "9547c6227f588a0b5c2e8ccd5ffd90c847f2234c"
TRAINING_COMMIT = "3e96df18eab0a8b3a6a3f1006d74c31e09add2a1"
N_VALUES = (16, 32, 64, 96, 128)
SEEDS = tuple(range(5))
ASSET_FILES = {
    "scenario_banks": "scenario_banks_v6.zip",
    "td3_training": "td3_training_archives_r2.zip",
    "ddpg_training": "ddpg_training_archives_r2.zip",
    "ppo_training": "ppo_training_archives_r2_complete.zip",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_outer_zip(path: Path, expected_files: set[str]) -> None:
    with zipfile.ZipFile(path) as archive:
        observed = {
            Path(item.filename).as_posix()
            for item in archive.infolist()
            if not item.is_dir()
        }
        bad_member = archive.testzip()
    if bad_member is not None:
        raise RuntimeError(f"{path}: CRC failure in {bad_member}")
    if observed != expected_files:
        raise RuntimeError(
            f"{path}: member mismatch; missing={sorted(expected_files - observed)}, "
            f"unexpected={sorted(observed - expected_files)}"
        )


def training_inventory(root: Path, method: str) -> dict[str, Any]:
    method_root = root / method
    manifest_path = method_root / f"TRAINING_RUN_MANIFEST_{method}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_jobs = {(n_ris, seed) for n_ris in N_VALUES for seed in SEEDS}
    observed_jobs = {
        (int(job["n_ris"]), int(job["seed"])) for job in manifest["jobs"]
    }
    if manifest.get("audit") != "PASS" or observed_jobs != expected_jobs:
        raise RuntimeError(f"{method}: incomplete training manifest")
    if manifest.get("git_commit") != TRAINING_COMMIT:
        raise RuntimeError(f"{method}: unexpected training commit")

    jobs: list[dict[str, Any]] = []
    for job in sorted(
        manifest["jobs"], key=lambda item: (int(item["n_ris"]), int(item["seed"]))
    ):
        archive = method_root / "collected" / str(job["archive"])
        observed_sha = sha256(archive)
        if observed_sha != str(job["archive_sha256"]):
            raise RuntimeError(f"Training archive checksum mismatch: {archive}")
        jobs.append(
            {
                "n_ris": int(job["n_ris"]),
                "seed": int(job["seed"]),
                "file": archive.name,
                "bytes": archive.stat().st_size,
                "sha256": observed_sha,
            }
        )
    return {
        "audit": manifest["audit"],
        "training_commit": manifest["git_commit"],
        "jobs": len(jobs),
        "manifest_file": manifest_path.name,
        "manifest_sha256": sha256(manifest_path),
        "archives": jobs,
    }


def bank_inventory(bank_root: Path) -> list[dict[str, Any]]:
    paths = sorted(bank_root.glob("N*_*.npz"))
    expected_names = {
        f"N{n_ris}_{split}.npz"
        for n_ris in N_VALUES
        for split in ("train", "validation", "test")
    }
    if {path.name for path in paths} != expected_names:
        raise RuntimeError("ScenarioBank coverage is not 5 N x 3 splits")
    return [
        {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in paths
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the tracked manifest for the permanent V6 r2 evidence release"
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=ROOT / "artifacts/release_staging" / RELEASE_TAG,
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=ROOT / "artifacts/physical_v6_full_r2",
    )
    parser.add_argument(
        "--bank-root",
        type=Path,
        default=ROOT / "artifacts/scenario_banks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/physical_v6_full_r2/evidence/EVIDENCE_MANIFEST.json",
    )
    args = parser.parse_args()

    inventories = {
        method: training_inventory(args.training_root, method)
        for method in ("td3", "ddpg", "ppo")
    }
    banks = bank_inventory(args.bank_root)

    expected_members = {
        "scenario_banks": {item["file"] for item in banks},
        **{
            f"{method}_training": {
                f"collected/{item['file']}"
                for item in inventories[method]["archives"]
            }
            | {inventories[method]["manifest_file"]}
            for method in ("td3", "ddpg", "ppo")
        },
    }

    release_url = f"https://github.com/{REPOSITORY}/releases/tag/{RELEASE_TAG}"
    download_base = f"https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}"
    assets: dict[str, dict[str, Any]] = {}
    for key, filename in ASSET_FILES.items():
        path = args.release_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        verify_outer_zip(path, expected_members[key])
        assets[key] = {
            "file": filename,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "download_url": f"{download_base}/{filename}",
            "members": len(expected_members[key]),
        }

    payload = {
        "schema_version": 1,
        "purpose": "Permanent evidence for the frozen physical V6 r2 results",
        "repository": REPOSITORY,
        "reviewed_source_commit": REVIEWED_SOURCE_COMMIT,
        "training_commit": TRAINING_COMMIT,
        "release_tag": RELEASE_TAG,
        "release_url": release_url,
        "results_directory": "results/physical_v6_full_r2",
        "protocol": {
            "methods": ["td3", "ddpg", "ppo"],
            "n_values": list(N_VALUES),
            "seeds": list(SEEDS),
            "interactions_per_job": 100_000,
            "training_jobs": 75,
            "scenario_bank_splits": ["train", "validation", "test"],
        },
        "assets": assets,
        "scenario_banks": banks,
        "training": inventories,
        "integrity": (
            "Verify every downloaded release asset against assets.*.sha256, then "
            "verify each nested training archive and ScenarioBank against this manifest."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sums = args.output.with_name("SHA256SUMS.txt")
    sums.write_text(
        "".join(
            f"{item['sha256']}  {item['file']}\n" for item in assets.values()
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "assets": len(assets),
                "release_bytes": sum(item["bytes"] for item in assets.values()),
                "training_archives": sum(item["jobs"] for item in inventories.values()),
                "scenario_banks": len(banks),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
