from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "STAR_RIS_RSMA_TD3_V6_corrected_review"
RESULT_SUFFIXES = {".csv", ".json", ".md", ".pdf", ".png", ".tex"}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True, encoding="utf-8"
    )
    include_roots = (
        ".github/workflows/", "configs/", "experiments/", "scripts/", "src/", "tests/",
    )
    include_exact = {
        "README.md", "pyproject.toml", "requirements.txt", "requirements-lock.txt",
    }
    files = []
    for name in output.splitlines():
        normalized = name.replace("\\", "/")
        if normalized in include_exact or normalized.startswith(include_roots):
            path = ROOT / normalized
            if path.is_file():
                files.append(path)
    return files


def add_file(archive: zipfile.ZipFile, source: Path, relative: Path) -> None:
    archive.write(source, f"{ARCHIVE_ROOT}/{relative.as_posix()}")


def add_tree_filtered(
    archive: zipfile.ZipFile, source_root: Path, destination_root: Path
) -> int:
    count = 0
    for source in sorted(source_root.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in RESULT_SUFFIXES:
            continue
        relative = destination_root / source.relative_to(source_root)
        add_file(archive, source, relative)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "STAR_RIS_RSMA_TD3_V6_corrected_review_2026-08-27.zip",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output

    sources = {
        "v6_td3_n32": ROOT / "artifacts/v6_soft_anchor_download",
        "v6_td3_n128": ROOT / "artifacts/v6_soft_anchor_n128_download_complete",
        "v6_comparators": ROOT / "artifacts/v6_comparator_pilot_download",
        "six_method_v2_corrected": ROOT / "results/six_method_v2",
    }
    missing = [name for name, path in sources.items() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing review sources: {missing}")

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    provenance = {
        "package": "V6 soft-anchor DRL plus corrected six-method review bundle",
        "repository_commit": commit,
        "branch": "audit/ao-baseline-full-n128",
        "v6_parameterization": "physical_v6_soft_anchor",
        "training_budget": 100000,
        "pilot_seeds": [0, 1],
        "test_scenarios_per_checkpoint": 1000,
        "github_runs": {
            "td3_v6_n32": 33038119621,
            "td3_v6_n128": 33039540684,
            "comparator_repair": 33047836088,
            "comparator_collector": 33049071587,
            "corrected_six_method": 32994891506,
        },
        "scenario_bank_checksums": {
            "N32": "6ec54735c1c2c35cd253d9d3d23295783293b6d64c65204dfcab16b2d59ebbef",
            "N128": "f4c80269e5fb3cf553900b2e82f235af875b7c2e33b4ff71ec5d85cc25eb2b4e",
        },
        "excluded": ["PyTorch checkpoints (*.pt)", "nested ZIP files", "runtime logs"],
    }
    readme = """# STAR-RIS RSMA V6 corrected review bundle

This bundle contains the complete repository code needed to inspect and rerun
`physical_v6_soft_anchor`, the corrected AO-SCA/AO-Grid baselines, tests,
configs, Kaggle/GitHub workflows, and auditable CSV/JSON results.

## Headline two-seed pilots

| Method | N | Mean best sum-rate |
|---|---:|---:|
| TD3 V6 | 32 | 16.444854 |
| TD3 V6 | 128 | 20.960908 |
| Corrected AO-SCA | 32 | 17.725581 |
| Corrected AO-SCA | 128 | 21.609453 |
| Corrected AO-Grid | 32 | 18.504410 |
| Corrected AO-Grid | 128 | 22.391796 |

The TD3 V6 values are pilots with two independent training seeds, not the
final five-seed estimates. `results/six_method_v2` contains the frozen
corrected six-method artifact. `pilot_results` contains raw initial/best/latest
CSV files and audit metadata for V6.

PyTorch checkpoints are intentionally excluded to keep the review package
portable. They remain in the GitHub Actions/Kaggle artifacts identified in
`PROVENANCE.json`.
"""

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in tracked_files():
            add_file(archive, source, source.relative_to(ROOT))
        result_counts = {}
        for name, source in sources.items():
            destination = Path("results/six_method_v2") if name == "six_method_v2_corrected" else Path("pilot_results") / name
            result_counts[name] = add_tree_filtered(archive, source, destination)
        provenance["included_result_file_counts"] = result_counts
        archive.writestr(f"{ARCHIVE_ROOT}/REVIEW_README.md", readme)
        archive.writestr(
            f"{ARCHIVE_ROOT}/PROVENANCE.json",
            json.dumps(provenance, indent=2),
        )

    digest = hashlib.sha256(output.read_bytes()).hexdigest().upper()
    print(json.dumps({
        "output": str(output.resolve()), "bytes": output.stat().st_size,
        "sha256": digest, "result_counts": provenance["included_result_file_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
