from __future__ import annotations

"""Robust V2 publisher for the cross-account STAR-RIS stage bundle.

Kaggle accepts private dataset creation asynchronously, but the
``datasets status`` endpoint can return HTTP 403 even for the dataset owner.
This wrapper therefore uses ``datasets files`` as the readiness probe and
requires all five audited stage archives before notebook 06 is submitted.
"""

import json
import time
from pathlib import Path

import prepare_kaggle_final_stage_dataset as base

DATASET_REF = "duythanhb1909984/star-ris-stage-outputs-final-v2"
DATASET_TITLE = "STAR RIS Stage Outputs Final V2"
EXPECTED_ARCHIVES = {
    "td3_low_n.zip",
    "td3_high_n.zip",
    "ao_grid.zip",
    "ao_sca.zip",
    "analytical_ris.zip",
}

base.DATASET_REF = DATASET_REF
base.DATASET_TITLE = DATASET_TITLE


def _combined(result) -> str:
    return "\n".join(
        part.strip()
        for part in (result.stdout or "", result.stderr or "")
        if part and part.strip()
    )


def _files_probe(env):
    return base.run(
        ["kaggle", "datasets", "files", DATASET_REF],
        env=env,
        check=False,
        capture=True,
    )


def _probe_has_all_archives(result) -> bool:
    if result.returncode != 0:
        return False
    text = _combined(result)
    return all(name in text for name in EXPECTED_ARCHIVES)


def _wait_until_files_ready(env, attempts: int = 90) -> None:
    for attempt in range(attempts):
        probe = _files_probe(env)
        text = _combined(probe)
        if _probe_has_all_archives(probe):
            print(f"Dataset files ready: {DATASET_REF}")
            return

        # HTTP 403/404 is observed while a newly-created private dataset is still
        # being finalized.  It is not treated as a terminal failure here.
        compact = " ".join(text.split())
        print(
            f"Dataset files not ready (attempt {attempt + 1}/{attempts}): "
            f"{compact or 'no response text'}"
        )
        time.sleep(10)

    raise TimeoutError(
        f"Dataset files did not become available with all archives: {DATASET_REF}"
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

    # A prior workflow may already have successfully submitted the asynchronous
    # creation before failing on the forbidden status endpoint.  Reuse it when
    # all expected files are visible instead of creating a duplicate version.
    existing = _files_probe(env)
    if _probe_has_all_archives(existing):
        print(f"Dataset already ready; reusing {DATASET_REF}")
        return

    result = base.run(
        ["kaggle", "datasets", "create", "-p", str(dataset_root)],
        env=env,
        check=False,
        capture=True,
    )
    detail = _combined(result)
    accepted_text = detail.lower()
    accepted = result.returncode == 0 or "dataset is being created" in accepted_text
    already_exists = any(
        marker in accepted_text
        for marker in ("already exists", "409", "conflict")
    )
    if not accepted and not already_exists:
        raise RuntimeError(
            f"Dataset publish command failed (exit {result.returncode}):\n"
            f"{detail or 'no response text'}"
        )

    if already_exists:
        print("Dataset already exists or creation is in progress; waiting for files.")
    else:
        print("Dataset creation accepted; waiting for all five archives.")

    _wait_until_files_ready(env)


base.publish_dataset = publish_dataset


if __name__ == "__main__":
    base.main()
