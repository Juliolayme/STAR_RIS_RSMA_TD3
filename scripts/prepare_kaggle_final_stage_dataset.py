from __future__ import annotations

"""Download successful Kaggle stage outputs and publish one private bundle dataset.

Notebook 06 runs under the PPO Kaggle account, while notebooks 01-04 are private
kernels owned by the TD3 and DDPG accounts. Kaggle does not reliably mount private
kernel sources across accounts. This script authenticates as each owner, downloads
only the evidence required by report 06, validates the stage manifests and expected
file counts, archives each stage, and publishes a private dataset owned by the PPO
account.
"""

import datetime as dt
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

DATASET_REF = "duythanhb1909984/star-ris-stage-outputs-final"
DATASET_TITLE = "STAR RIS Stage Outputs Final"
REQUIRED_STAGES = {
    "td3_low_n": {
        "kernel": "thanhnguyen1423/star-ris-td3-low-n-final",
        "username": "thanhnguyen1423",
        "secret": "KAGGLE_TD3_KEY",
        "test_count": 24,
        "checkpoint_count": 3,
    },
    "td3_high_n": {
        "kernel": "thanhnguyen1423/star-ris-td3-high-n-final",
        "username": "thanhnguyen1423",
        "secret": "KAGGLE_TD3_KEY",
        "test_count": 16,
        "checkpoint_count": 2,
    },
    "ao_grid": {
        "kernel": "ronganminh/star-ris-ao-grid-final",
        "username": "ronganminh",
        "secret": "KAGGLE_DDPG_KEY",
        "raw_name": "AO_GRID_RAW_ALL.csv",
    },
    "ao_sca": {
        "kernel": "ronganminh/star-ris-ao-sca-final",
        "username": "ronganminh",
        "secret": "KAGGLE_DDPG_KEY",
        "raw_name": "AO_SCA_RAW_ALL.csv",
    },
    "analytical_ris": {
        "kernel": "duythanhb1909984/star-ris-analytical-ris-final",
        "username": "duythanhb1909984",
        "secret": "KAGGLE_PPO_KEY",
        "raw_name": "ANALYTICAL_RIS_RAW_ALL.csv",
    },
}


def kaggle_env(username: str, secret_name: str) -> dict[str, str]:
    key = os.environ.get(secret_name, "").strip()
    if not key:
        raise RuntimeError(f"Missing GitHub Actions secret {secret_name}")
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key
    return env


def run(
    command: list[str],
    *,
    env: dict[str, str],
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(command))
    result = subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=capture,
        check=False,
    )
    if capture and result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0 and check:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{detail}")
    return result


def copy_file(source: Path, stage_root: Path, destination_root: Path) -> None:
    relative = source.relative_to(stage_root)
    destination = destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def unique_file(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} below {root}, found {matches}")
    return matches[0]


def select_td3_evidence(
    stage_id: str,
    downloaded_root: Path,
    destination_root: Path,
    expected_test_count: int,
    expected_checkpoint_count: int,
) -> dict[str, int]:
    manifest = unique_file(downloaded_root, "STAGE_MANIFEST.json")
    stage_root = manifest.parent
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("stage_id") != stage_id:
        raise RuntimeError(
            f"Stage mismatch for {stage_id}: manifest says {payload.get('stage_id')}"
        )
    copy_file(manifest, stage_root, destination_root)

    test_files = list(stage_root.rglob("test.csv"))
    training_files = list(stage_root.rglob("train/training.csv"))
    validation_files = list(stage_root.rglob("train/validation_summary.csv"))
    checkpoint_files = list(stage_root.rglob("seed_0/train/best.pt"))

    if len(test_files) != expected_test_count:
        raise RuntimeError(
            f"{stage_id}: expected {expected_test_count} test CSVs, found {len(test_files)}"
        )
    if len(training_files) != expected_test_count:
        raise RuntimeError(
            f"{stage_id}: expected {expected_test_count} training CSVs, "
            f"found {len(training_files)}"
        )
    if len(validation_files) != expected_test_count:
        raise RuntimeError(
            f"{stage_id}: expected {expected_test_count} validation CSVs, "
            f"found {len(validation_files)}"
        )
    if len(checkpoint_files) != expected_checkpoint_count:
        raise RuntimeError(
            f"{stage_id}: expected {expected_checkpoint_count} seed-0 checkpoints, "
            f"found {len(checkpoint_files)}"
        )

    for source in test_files + training_files + validation_files + checkpoint_files:
        copy_file(source, stage_root, destination_root)
    return {
        "test_csv": len(test_files),
        "training_csv": len(training_files),
        "validation_csv": len(validation_files),
        "seed0_checkpoint": len(checkpoint_files),
    }


def select_baseline_evidence(
    stage_id: str,
    downloaded_root: Path,
    destination_root: Path,
    raw_name: str,
) -> dict[str, int]:
    manifest = unique_file(downloaded_root, "STAGE_MANIFEST.json")
    stage_root = manifest.parent
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("stage_id") != stage_id:
        raise RuntimeError(
            f"Stage mismatch for {stage_id}: manifest says {payload.get('stage_id')}"
        )
    raw = unique_file(stage_root, raw_name)
    copy_file(manifest, stage_root, destination_root)
    copy_file(raw, stage_root, destination_root)
    return {"merged_raw_csv": 1}


def download_and_package(work_root: Path) -> Path:
    download_root = work_root / "downloads"
    selected_root = work_root / "selected"
    dataset_root = work_root / "dataset"
    for path in (download_root, selected_root, dataset_root):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    bundle_manifest: dict[str, object] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_ref": DATASET_REF,
        "stages": {},
    }

    for stage_id, spec in REQUIRED_STAGES.items():
        owner_env = kaggle_env(str(spec["username"]), str(spec["secret"]))
        kernel = str(spec["kernel"])
        status = run(
            ["kaggle", "kernels", "status", kernel], env=owner_env, capture=True
        )
        status_text = f"{status.stdout}\n{status.stderr}".lower()
        if "complete" not in status_text:
            raise RuntimeError(f"Kernel is not complete: {kernel}\n{status_text}")

        downloaded = download_root / stage_id
        downloaded.mkdir(parents=True, exist_ok=True)
        run(
            ["kaggle", "kernels", "output", kernel, "-p", str(downloaded), "-o"],
            env=owner_env,
        )

        selected = selected_root / stage_id
        selected.mkdir(parents=True, exist_ok=True)
        if stage_id.startswith("td3_"):
            counts = select_td3_evidence(
                stage_id,
                downloaded,
                selected,
                int(spec["test_count"]),
                int(spec["checkpoint_count"]),
            )
        else:
            counts = select_baseline_evidence(
                stage_id,
                downloaded,
                selected,
                str(spec["raw_name"]),
            )

        archive = shutil.make_archive(
            str(dataset_root / stage_id), "zip", root_dir=selected
        )
        bundle_manifest["stages"][stage_id] = {
            "kernel": kernel,
            "archive": Path(archive).name,
            "selected_files": counts,
        }
        print(f"Packaged {stage_id}: {archive}")

    (dataset_root / "BUNDLE_MANIFEST.json").write_text(
        json.dumps(bundle_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (dataset_root / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": DATASET_TITLE,
                "id": DATASET_REF,
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_root


def publish_dataset(dataset_root: Path) -> None:
    ppo_env = kaggle_env("duythanhb1909984", "KAGGLE_PPO_KEY")
    existing = run(
        ["kaggle", "datasets", "status", DATASET_REF],
        env=ppo_env,
        check=False,
        capture=True,
    )
    if existing.returncode == 0:
        run(
            [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(dataset_root),
                "-m",
                "Refresh successful STAR-RIS stages 01-05 for report 06",
            ],
            env=ppo_env,
        )
    else:
        run(
            ["kaggle", "datasets", "create", "-p", str(dataset_root)],
            env=ppo_env,
        )

    for attempt in range(60):
        status = run(
            ["kaggle", "datasets", "status", DATASET_REF],
            env=ppo_env,
            check=False,
            capture=True,
        )
        text = f"{status.stdout}\n{status.stderr}".lower()
        if status.returncode == 0 and "ready" in text:
            print(f"Dataset ready: {DATASET_REF}")
            return
        if "error" in text or "failed" in text:
            raise RuntimeError(f"Dataset creation failed: {text}")
        print(f"Dataset not ready yet (attempt {attempt + 1}/60)")
        time.sleep(10)
    raise TimeoutError(f"Dataset did not become ready: {DATASET_REF}")


def main() -> None:
    work_root = Path(".kaggle_final_stage_bundle")
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)
    dataset_root = download_and_package(work_root)
    publish_dataset(dataset_root)


if __name__ == "__main__":
    main()
