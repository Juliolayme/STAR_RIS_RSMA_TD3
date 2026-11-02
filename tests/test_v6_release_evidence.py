from __future__ import annotations

import json
from pathlib import Path
import runpy
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/physical_v6_full_r2/evidence"
MANIFEST = json.loads(
    (EVIDENCE / "EVIDENCE_MANIFEST.json").read_text(encoding="utf-8")
)
DOWNLOADER = runpy.run_path(EVIDENCE / "download_evidence.py")


def test_release_manifest_covers_all_frozen_evidence() -> None:
    assert MANIFEST["reviewed_source_commit"] == (
        "9547c6227f588a0b5c2e8ccd5ffd90c847f2234c"
    )
    assert MANIFEST["protocol"]["training_jobs"] == 75
    assert len(MANIFEST["scenario_banks"]) == 15
    assert set(MANIFEST["assets"]) == {
        "scenario_banks",
        "td3_training",
        "ddpg_training",
        "ppo_training",
    }
    assert all(len(item["sha256"]) == 64 for item in MANIFEST["assets"].values())
    assert sum(item["jobs"] for item in MANIFEST["training"].values()) == 75


def test_release_urls_are_permanent_release_assets_not_actions_artifacts() -> None:
    tag = MANIFEST["release_tag"]
    for item in MANIFEST["assets"].values():
        assert f"/releases/download/{tag}/" in item["download_url"]
        assert "/actions/" not in item["download_url"]


def test_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.txt", "unsafe")
    with pytest.raises(RuntimeError, match="Unsafe archive member"):
        DOWNLOADER["safe_extract"](archive, tmp_path / "extract")
