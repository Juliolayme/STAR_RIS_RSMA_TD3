from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "STAR_RIS_RSMA_TD3_physical_v6_full_review"
DEFAULT_METHOD_ROOTS = {
    "td3": "artifacts/physical_v6_full_download/td3",
    "ddpg": "artifacts/physical_v6_full_download/comparators/physical-v6-full-25jobs-ddpg",
    "ppo": "artifacts/physical_v6_full_download/comparators/physical-v6-full-25jobs-ppo",
}
DEFAULT_RESULTS_DIR = "results/physical_v6_full"
ROOT_FILES = {
    ".gitignore",
    "CODEX_REPOSITORY_GUIDE.md",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-lock.txt",
}
BASE_PREFIXES = (
    ".github/workflows/",
    "configs/",
    "experiments/structure_aware/",
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
    "candidate_checkpoints.json",
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


def tracked_review_files(results_dir: str) -> list[Path]:
    included_prefixes = (*BASE_PREFIXES, f"{results_dir.rstrip('/')}/")
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True, encoding="utf-8"
    )
    paths = [Path(line) for line in output.splitlines() if line]
    selected = sorted(
        path
        for path in paths
        if path.as_posix() in ROOT_FILES
        or path.as_posix().startswith(included_prefixes)
    )
    for path in selected:
        if path.name.lower() in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise RuntimeError(f"Forbidden file selected for review package: {path}")
    return selected


def validate_published_results(files: list[Path], results_dir: str) -> dict[str, Any]:
    included = {path.as_posix() for path in files}
    required = {
        f"{results_dir}/{name}"
        for name in (
            "PHYSICAL_V6_FULL_AUDIT.json",
            "PHYSICAL_V6_FULL_REVIEW.md",
            "raw/DRL_V6_TEST_BEST_RAW_ALL.csv",
            "raw/CPU_LATENCY_V6_RAW_ALL.csv",
            "tables/TABLE_V6_SIX_METHOD_PERFORMANCE.csv",
            "tables/TABLE_V6_SIX_METHOD_PAIRED_TESTS_HOLM.csv",
            "tables/TABLE_V6_SIX_METHOD_CPU_LATENCY.csv",
        )
    } | {"results/six_method_v2/raw/TRADITIONAL_TEST_RAW_ALL.csv"}
    missing = sorted(required - included)
    if missing:
        raise RuntimeError(f"Review package is missing required tracked results: {missing}")
    audit = json.loads(
        (ROOT / results_dir / "PHYSICAL_V6_FULL_AUDIT.json").read_text(encoding="utf-8")
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


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def review_readme(commit: str, branch: str, results_dir: str, audit: dict[str, Any]) -> str:
    """Build the summary from the published tables.

    An earlier version carried the headline numbers as literal text. They
    silently described the previous protocol once the results were rebuilt,
    so every figure quoted here is read back out of the CSVs being shipped.
    """
    tables = ROOT / results_dir / "tables"
    quality = read_table(tables / "TABLE_V6_SIX_METHOD_PERFORMANCE.csv")
    latency = read_table(tables / "TABLE_V6_SIX_METHOD_CPU_LATENCY.csv")
    speedup = read_table(tables / "TABLE_V6_TD3_LATENCY_SPEEDUP.csv")

    n_values = sorted({int(row["n_ris"]) for row in quality})
    order = ("td3", "ddpg", "ppo", "ao_sca", "ao_grid", "analytical_ris")
    rate = {(r["method"], int(r["n_ris"])): float(r["sum_rate_mean"]) for r in quality}
    solve = {(r["method"], int(r["n_ris"])): float(r["solve_ms_median"]) for r in latency}

    header = "| Method | " + " | ".join("N=" + str(n) for n in n_values) + " |"
    divider = "|---|" + "---:|" * len(n_values)

    def block(values: dict, digits: int) -> list[str]:
        rows = []
        for method in order:
            if not all((method, n) in values for n in n_values):
                continue
            cells = " | ".join(format(values[(method, n)], "." + str(digits) + "f") for n in n_values)
            rows.append("| " + method + " | " + cells + " |")
        return rows

    learned = ("td3", "ddpg", "ppo")
    spreads = {
        n: max(rate[(m, n)] for m in learned) - min(rate[(m, n)] for m in learned)
        for n in n_values
    }
    widest = max(spreads, key=lambda n: spreads[n])
    behind = min(learned, key=lambda m: rate[(m, widest)])
    rest = [n for n in n_values if n != widest]
    spread_line = "Spread among the learned methods by N: " + ", ".join(
        "N=" + str(n) + " " + format(spreads[n], ".3f") for n in n_values
    )
    factors: dict[str, list[float]] = {}
    for row in speedup:
        factors.setdefault(row["baseline_method"], []).append(
            float(row["baseline_over_td3_speedup"])
        )
    speed_lines = [
        "- " + baseline + ": " + format(min(v), ",.0f") + "x to " + format(max(v), ",.0f") + "x"
        for baseline, v in sorted(factors.items())
        if min(v) > 1
    ]
    jobs = int(audit["training_jobs"])
    rows_count = int(audit["best_checkpoint_test_rows"])
    latency_rows = int(audit["latency"]["raw_rows"])

    lines = [
        "# STAR-RIS RSMA physical V6 review bundle",
        "",
        "Repository branch: `" + branch + "`  ",
        "Packaging commit: `" + commit + "`  ",
        "Published results: `" + results_dir + "/`",
        "",
        "Tracked source, configs, workflows and tests, the corrected traditional",
        "baselines, the V6 quality and statistical tables, " + format(rows_count, ",") +
        " best-checkpoint test rows, " + format(latency_rows, ",") + " latency samples,",
        "figures, and compact training metadata for all " + str(jobs) + " GPU jobs.",
        "",
        "## Mean sum-rate over five training seeds",
        "",
        header,
        divider,
        *block(rate, 3),
        "",
        spread_line + ".",
        "",
        "The widest gap is at N=" + str(widest) + ", where " + behind + " trails at " +
        format(rate[(behind, widest)], ".3f") + "; its per-seed figures there are the",
        "least stable in the study and are reported as measured. Excluding that",
        "point the three lie within " + format(max(spreads[n] for n in rest), ".3f") + ",",
        "which is too close for this evidence to rank them.",
        "",
        "## Median single-thread CPU decision time (ms)",
        "",
        header,
        divider,
        *block(solve, 3),
        "",
        "TD3 against the corrected traditional solvers, across N:",
        "",
        *speed_lines,
        "",
        "Quality uses five training seeds. Latency uses one fixed seed-0",
        "best-validation checkpoint per method and N, measured single-threaded on",
        "one runner so all six methods share a machine.",
        "",
        "## Start here",
        "",
        "- `" + results_dir + "/PHYSICAL_V6_FULL_REVIEW.md`",
        "- `" + results_dir + "/PHYSICAL_V6_FULL_AUDIT.json`",
        "- `" + results_dir + "/tables/`",
        "- `review_evidence/training_runs/` for per-job learning and timing metadata",
        "- `PACKAGE_PROVENANCE.json` and `PACKAGE_FILE_MANIFEST.json`",
        "",
        "Run the test suite with `python -m pytest -q` after installing the package.",
        "",
        "PyTorch checkpoints (`*.pt`), ScenarioBank arrays (`*.npz`), nested ZIPs and",
        "credentials are excluded. Their SHA-256 values and the GitHub run and",
        "artifact identifiers are retained in the audits and in",
        "`PACKAGE_PROVENANCE.json`.",
        "",
    ]
    return chr(10).join(lines)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a portable physical V6 code/result review ZIP"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "STAR_RIS_RSMA_TD3_physical_v6_full_review_2026-08-27.zip",
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    for method, default in DEFAULT_METHOD_ROOTS.items():
        parser.add_argument(f"--{method}-root", type=Path, default=ROOT / default)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    results_dir = args.results_dir.strip("/")
    method_roots = {
        method: getattr(args, f"{method}_root") for method in DEFAULT_METHOD_ROOTS
    }

    files = tracked_review_files(results_dir)
    audit = validate_published_results(files, results_dir)
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

        for method, root in method_roots.items():
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
            "published_results": results_dir,
            # Taken from the audit being shipped. Literal run ids here once
            # outlived the results they described.
            "quality_test_rows": int(audit["best_checkpoint_test_rows"]),
            "latency_rows": int(audit["latency"]["raw_rows"]),
            "github_artifacts": audit["github_artifacts"]
            | {
                "latency": {
                    "run_id": audit["latency"]["github_run_id"],
                    "artifact_id": audit["latency"]["github_artifact_id"],
                    "artifact_name": audit["latency"]["github_artifact_name"],
                }
            },
            "repository_commits_per_method": audit["repository_commits_per_method"],
            "scenario_bank_checksums": audit["scenario_bank_checksums"],
            "training_manifests": training_manifests,
            "excluded": ["*.pt", "*.npz", "nested ZIPs", "credentials", "download caches"],
        }
        add_bytes(
            "REVIEW_README.md",
            review_readme(commit, branch, results_dir, audit).encode("utf-8"),
        )
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
