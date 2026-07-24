from __future__ import annotations

"""Robust V2 publisher for the cross-account STAR-RIS stage bundle.

The original packager is reused for downloading and auditing stages 01-05.  This
wrapper switches to a fresh dataset slug and makes dataset creation diagnostics
explicit, including the case where Kaggle accepts an asynchronous creation but
the CLI still exits non-zero.
"""

import json
import time
from pathlib import Path

import prepare_kaggle_final_stage_dataset as base

DATASET_REF = "duythanhb1909984/star-ris-stage-outputs-final-v2"
DATASET_TITLE = "STAR RIS Stage Outputs Final V2"

base.DATASET_REF = DATASET_REF
base.DATASET_TITLE = DATASET_TITLE


def _combined(result) -> str:
    return "\n".join(
        part.strip()
        for part in (result.stdout or "", result.stderr or "")
        if part and part.strip()
    )


def _status(env):
    return base.run(
        ["kaggle", "datasets", "status", DATASET_REF],
        env=env,
        check=False,
        capture=True,
    )


def publish_dataset(dataset_root: Path) -> None:
    env = base.kaggle_env("duythanhb1909984", "KAGGLE_PPO_KEY")

    metadata_path = dataset_root / "dataset-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("id") != DATASET_REF or metadata.get("title") != DATASET_TITLE:
        raise RuntimeError(f"Unexpected dataset metadata: {metadata}")

    print("Dataset upload contents:")
    for path in sorted(dataset_root.iterdir()):
        if path.is_file():
            print(f"  {path.name}: {path.stat().st_size / (1024 ** 2):.2f} MiB")

    existing = _status(env)
    if existing.returncode == 0:
        command = [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(dataset_root),
            "-m",
            "Refresh successful STAR-RIS stages 01-05 for report 06",
        ]
    else:
        command = ["kaggle", "datasets", "create", "-p", str(dataset_root)]

    result = base.run(command, env=env, check=False, capture=True)
    if result.returncode != 0:
        # Kaggle dataset creation is asynchronous.  Some CLI/server combinations
        # can return a non-zero code after the server has already accepted it.
        accepted = _status(env)
        if accepted.returncode != 0:
            detail = _combined(result) or f"exit code {result.returncode}"
            status_detail = _combined(accepted) or f"exit code {accepted.returncode}"
            raise RuntimeError(
                f"Dataset publish command failed:\n{detail}\n"
                f"Post-failure status check also failed:\n{status_detail}"
            )
        print("Dataset exists after non-zero publish return; continuing to poll.")

    for attempt in range(90):
        status = _status(env)
        text = _combined(status).lower()
        if status.returncode == 0 and "ready" in text:
            print(f"Dataset ready: {DATASET_REF}")
            return
        if "error" in text or "failed" in text:
            raise RuntimeError(f"Dataset creation failed: {text}")
        print(f"Dataset not ready yet (attempt {attempt + 1}/90): {text or 'no status text'}")
        time.sleep(10)

    raise TimeoutError(f"Dataset did not become ready: {DATASET_REF}")


base.publish_dataset = publish_dataset


if __name__ == "__main__":
    base.main()
