from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "STAR_RIS_RSMA_TD3_physical_v5_soft_review"
CODE_PREFIXES = (
    "src/",
    "tests/",
)
CODE_FILES = {
    ".github/workflows/run-kaggle-structure-aware-soft.yml",
    ".github/workflows/collect-structure-aware-pilot.yml",
    "configs/v3/pilot_v5_soft_n32.yaml",
    "scripts/create_scenario_banks.py",
    "scripts/pilot_structure_aware_td3.py",
    "scripts/audit_structure_aware_pilot.py",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
}
RESULT_SUFFIXES = {".csv", ".json", ".md"}


def git_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True, encoding="utf-8"
    )
    selected = []
    for line in output.splitlines():
        path = Path(line)
        normalized = path.as_posix()
        if normalized in CODE_FILES or normalized.startswith(CODE_PREFIXES):
            selected.append(path)
    return sorted(selected)


def result_files(artifact: Path) -> list[Path]:
    selected: list[Path] = []
    audit = artifact / "structure_pilot_audit"
    selected.extend(path for path in audit.rglob("*") if path.is_file())
    for summary in artifact.rglob("summary.json"):
        payload = json.loads(summary.read_text(encoding="utf-8"))
        if payload.get("parameterization") != "physical_v5_soft":
            continue
        selected.extend(
            path
            for path in summary.parent.rglob("*")
            if path.is_file() and path.suffix.lower() in RESULT_SUFFIXES
        )
    return sorted(set(selected))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit_path = args.artifact / "structure_pilot_audit/PILOT_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("audit") != "PASS" or int(audit.get("rows_per_checkpoint", 0)) != 1000:
        raise RuntimeError("Refusing to package a pilot that did not pass the locked audit")

    results = result_files(args.artifact)
    summaries = [path for path in results if path.name == "summary.json"]
    if len(summaries) != 2:
        raise RuntimeError(f"Expected two soft seed summaries, found {len(summaries)}")

    provenance = {
        "package": "physical_v5_soft N=32 two-seed review bundle",
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "collector_run": 33036935449,
        "collector_artifact": "structure-aware-td3-pilot-n32",
        "parameterization": "physical_v5_soft",
        "seeds": [0, 1],
        "train_steps_per_seed": 100000,
        "test_scenarios_per_checkpoint": 1000,
        "test_bank_checksum": audit["test_bank_checksum"],
        "audit": "PASS",
    }
    review = """# Physical V5 Soft TD3 review bundle

This package contains the implementation and the audited N=32 pilot evidence
for `physical_v5_soft` only. The two runs use seeds 0 and 1, 100,000 training
steps, and the same locked 1,000-scenario test bank. Each seed includes raw test
CSV files for the initial, best-validation, and latest checkpoints.

The frozen Six-Method V2 results are not replaced by this pilot. The aggregate
hard-vs-soft tables are included only to preserve the CI audit context.
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in git_files():
            archive.write(ROOT / relative, f"{ARCHIVE_ROOT}/{relative.as_posix()}")
        for source in results:
            relative = source.relative_to(args.artifact).as_posix()
            archive.write(source, f"{ARCHIVE_ROOT}/pilot_results/{relative}")
        archive.writestr(f"{ARCHIVE_ROOT}/REVIEW_README.md", review)
        archive.writestr(
            f"{ARCHIVE_ROOT}/PROVENANCE.json",
            json.dumps(provenance, indent=2),
        )

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest().upper()
    print(f"created={args.output}")
    print(f"bytes={args.output.stat().st_size}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
