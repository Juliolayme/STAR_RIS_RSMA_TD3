from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "STAR_RIS_RSMA_TD3_physical_v6_full_review"
METHOD_ROOTS = {
    "td3": ROOT / "artifacts/physical_v6_full_download/td3",
    "ddpg": ROOT / "artifacts/physical_v6_full_download/comparators/physical-v6-full-25jobs-ddpg",
    "ppo": ROOT / "artifacts/physical_v6_full_download/comparators/physical-v6-full-25jobs-ppo",
}
ROOT_FILES = {
    ".gitignore",
    "CODEX_REPOSITORY_GUIDE.md",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-lock.txt",
}
INCLUDED_PREFIXES = (
    ".github/workflows/",
    "configs/",
    "experiments/structure_aware/",
    "results/physical_v6_full/",
    "results/six_method_v2/",
    "scripts/",
    "src/",
    "tests/",
)
TRAINING_METADATA = (
    "summary.json",
    "timing.json",
    "manifest.json",
    "best_validation.json",
    "training.csv",
    "validation_summary.csv",
    "KAGGLE_JOB_PROVENANCE.json",
    "SCENARIO_BANK_VERIFICATION.json",
)
FORBIDDEN_SUFFIXES = {".pt", ".npz", ".zip", ".pem", ".key"}
FORBIDDEN_NAMES = {".env", "kaggle.json", "kaggle_jobs.json"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_review_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True, encoding="utf-8"
    )
    paths = [Path(line) for line in output.splitlines() if line]
    selected = sorted(
        path
        for path in paths
        if path.as_posix() in ROOT_FILES
        or path.as_posix().startswith(INCLUDED_PREFIXES)
    )
    for path in selected:
        if path.name.lower() in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise RuntimeError(f"Forbidden file selected for review package: {path}")
    return selected


def validate_published_results(files: list[Path]) -> dict[str, Any]:
    included = {path.as_posix() for path in files}
    required = {
        "results/physical_v6_full/PHYSICAL_V6_FULL_AUDIT.json",
        "results/physical_v6_full/PHYSICAL_V6_FULL_REVIEW.md",
        "results/physical_v6_full/raw/DRL_V6_TEST_BEST_RAW_ALL.csv",
        "results/physical_v6_full/raw/CPU_LATENCY_V6_RAW_ALL.csv",
        "results/physical_v6_full/tables/TABLE_V6_SIX_METHOD_PERFORMANCE.csv",
        "results/physical_v6_full/tables/TABLE_V6_SIX_METHOD_PAIRED_TESTS_HOLM.csv",
        "results/physical_v6_full/tables/TABLE_V6_SIX_METHOD_CPU_LATENCY.csv",
        "results/six_method_v2/raw/TRADITIONAL_TEST_RAW_ALL.csv",
    }
    missing = sorted(required - included)
    if missing:
        raise RuntimeError(f"Review package is missing required tracked results: {missing}")
    audit = json.loads(
        (ROOT / "results/physical_v6_full/PHYSICAL_V6_FULL_AUDIT.json").read_text(
            encoding="utf-8"
        )
    )
    if audit.get("verdict") != "PASS" or int(audit.get("training_jobs", 0)) != 75:
        raise RuntimeError("Physical V6 full audit is not PASS/75 jobs")
    latency = audit.get("latency", {})
    if latency.get("verdict") != "PASS" or int(latency.get("raw_rows", 0)) != 3000:
        raise RuntimeError("Physical V6 latency audit is not PASS/3,000 rows")
    return audit


def load_training_manifest(root: Path, method: str) -> dict[str, Any]:
    paths = sorted(root.glob("TRAINING_RUN_MANIFEST*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one training manifest under {root}: {paths}")
    manifest = json.loads(paths[0].read_text(encoding="utf-8"))
    if manifest.get("audit") != "PASS" or int(manifest.get("completed_jobs", 0)) != 25:
        raise RuntimeError(f"Training evidence is incomplete for {method}")
    if manifest["protocol"]["method"] != method:
        raise RuntimeError(f"Training manifest method mismatch for {method}")
    return manifest


def find_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith("/" + suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {suffix} in training archive, got {matches}")
    return matches[0]


def review_readme(commit: str) -> str:
    return f"""# STAR-RIS RSMA physical V6 full review bundle

Repository branch: `experiment/physical-v6-full`  
Packaging commit: `{commit}`

This self-contained review bundle includes the tracked source, configs,
workflows, tests, corrected traditional baselines, V6 quality/statistical
tables, 75,000 best-checkpoint test rows, 3,000 latency samples, figures, and
compact training metadata for all 75 GPU jobs.

## Headline results

| N | TD3 sum-rate | AO-SCA corrected | AO-Grid corrected | TD3 median CPU latency |
|---:|---:|---:|---:|---:|
| 16 | 10.2881 | 14.3449 | 16.5802 | 0.3317 ms |
| 32 | 16.1728 | 17.7256 | 18.5044 | 0.3575 ms |
| 64 | 18.7154 | 19.9050 | 20.4401 | 0.3903 ms |
| 96 | 19.9279 | 20.9103 | 21.5862 | 0.4335 ms |
| 128 | 20.9018 | 21.6095 | 22.3918 | 0.4547 ms |

TD3 is 2,414x-7,539x faster than corrected AO-SCA and 551x-2,524x faster
than corrected AO-Grid in median single-thread decision time. AnalyticalRIS
is faster than TD3 but has much lower sum-rate. Quality uses five training
seeds; latency uses one fixed seed-0 best-validation checkpoint per N.

## Start here

- `results/physical_v6_full/PHYSICAL_V6_FULL_REVIEW.md`
- `results/physical_v6_full/PHYSICAL_V6_FULL_AUDIT.json`
- `results/physical_v6_full/tables/`
- `review_evidence/training_runs/` for per-job learning/timing metadata
- `PACKAGE_PROVENANCE.json` and `PACKAGE_FILE_MANIFEST.json`

Run the test suite with `python -m pytest -q` after installing the package.

PyTorch checkpoints (`*.pt`), ScenarioBank arrays (`*.npz`), nested ZIPs, and
credentials are intentionally excluded. Exact checkpoint/archive SHA-256
values and GitHub run/artifact IDs are retained in the audits and provenance.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a portable physical V6 code/result review ZIP"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "STAR_RIS_RSMA_TD3_physical_v6_full_review_2026-08-27.zip",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output

    files = tracked_review_files()
    audit = validate_published_results(files)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()

    payload: dict[str, dict[str, Any]] = {}
    training_manifests: dict[str, dict[str, Any]] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        def add_bytes(relative: str, data: bytes) -> None:
            normalized = relative.replace("\\", "/")
            if Path(normalized).name.lower() in FORBIDDEN_NAMES or Path(normalized).suffix.lower() in FORBIDDEN_SUFFIXES:
                raise RuntimeError(f"Forbidden archive member: {normalized}")
            archive.writestr(f"{ARCHIVE_ROOT}/{normalized}", data)
            payload[normalized] = {"bytes": len(data), "sha256": sha256_bytes(data)}

        for relative in files:
            add_bytes(relative.as_posix(), (ROOT / relative).read_bytes())

        for method, root in METHOD_ROOTS.items():
            manifest = load_training_manifest(root, method)
            training_manifests[method] = {
                "git_commit": manifest["git_commit"],
                "completed_jobs": manifest["completed_jobs"],
                "failed_jobs": manifest["failed_jobs"],
                "updated_at_utc": manifest["updated_at_utc"],
            }
            manifest_name = next(root.glob("TRAINING_RUN_MANIFEST*.json")).name
            add_bytes(
                f"review_evidence/training_runs/{method}/{manifest_name}",
                next(root.glob("TRAINING_RUN_MANIFEST*.json")).read_bytes(),
            )
            for job in sorted(manifest["jobs"], key=lambda item: (int(item["n_ris"]), int(item["seed"]))):
                archive_path = root / "collected" / str(job["archive"])
                if sha256_file(archive_path) != str(job["archive_sha256"]):
                    raise RuntimeError(f"Training archive checksum mismatch: {archive_path}")
                destination = (
                    f"review_evidence/training_runs/{method}/"
                    f"N{int(job['n_ris'])}/seed{int(job['seed'])}"
                )
                with zipfile.ZipFile(archive_path) as training_archive:
                    for name in TRAINING_METADATA:
                        add_bytes(
                            f"{destination}/{name}",
                            training_archive.read(find_member(training_archive, name)),
                        )

        provenance = {
            "package": "STAR-RIS RSMA physical V6 full code and result review",
            "repository": "Juliolayme/STAR_RIS_RSMA_TD3",
            "branch": branch,
            "repository_commit": commit,
            "action_parameterization": "physical_v6_soft_anchor",
            "training_protocol": "5 N x 5 seeds x 3 learned methods x 100,000 interactions",
            "quality_test_rows": 75000,
            "latency_rows": 3000,
            "github_runs": {
                "td3_training": 33053693666,
                "ddpg_ppo_training": 33061462093,
                "six_method_latency": 33076374485,
                "publish_ci": 33079269691,
            },
            "github_artifacts": audit["github_artifacts"] | {
                "latency": {
                    "run_id": 33076374485,
                    "artifact_id": 9649135691,
                    "artifact_name": "PHYSICAL_V6_SIX_METHOD_LATENCY",
                }
            },
            "scenario_bank_checksums": audit["scenario_bank_checksums"],
            "training_manifests": training_manifests,
            "excluded": ["*.pt", "*.npz", "nested ZIPs", "credentials", "download caches"],
        }
        add_bytes("REVIEW_README.md", review_readme(commit).encode("utf-8"))
        add_bytes(
            "PACKAGE_PROVENANCE.json",
            json.dumps(provenance, indent=2, sort_keys=True).encode("utf-8"),
        )
        file_manifest = {
            "payload_file_count": len(payload),
            "payload_bytes": sum(item["bytes"] for item in payload.values()),
            "files": payload,
        }
        archive.writestr(
            f"{ARCHIVE_ROOT}/PACKAGE_FILE_MANIFEST.json",
            json.dumps(file_manifest, indent=2, sort_keys=True).encode("utf-8"),
        )

    digest = sha256_file(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "bytes": output.stat().st_size,
                "sha256": digest,
                "sidecar": str(sidecar.resolve()),
                "tracked_files": len(files),
                "payload_files": len(payload),
                "training_jobs": sum(item["completed_jobs"] for item in training_manifests.values()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
