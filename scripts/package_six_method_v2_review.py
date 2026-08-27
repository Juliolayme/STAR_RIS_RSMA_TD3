from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "STAR_RIS_RSMA_six_method_v2_review"
ROOT_FILES = {
    "CODEX_REPOSITORY_GUIDE.md",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
}
INCLUDED_PREFIXES = (
    ".github/workflows/",
    "configs/",
    "experiments/six_method/",
    "experiments/six_method_v2/",
    "results/six_method_v1/",
    "results/six_method_v2/",
    "scripts/",
    "src/",
    "tests/",
)
REQUIRED_PROVENANCE_FILES = (
    "results/six_method_v1/raw/DRL_TEST_RAW_ALL.csv",
    "results/six_method_v1/tables/TABLE_SIX_METHOD_CPU_LATENCY.csv",
    "experiments/six_method/README.md",
    "results/six_method_v2/SIX_METHOD_V2_AUDIT.json",
)


def tracked_review_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )
    paths = [Path(line) for line in output.splitlines() if line]
    return sorted(
        path
        for path in paths
        if path.as_posix() in ROOT_FILES
        or path.as_posix().startswith(INCLUDED_PREFIXES)
    )


def validate_inputs(files: list[Path]) -> None:
    included = {path.as_posix() for path in files}
    missing = [path for path in REQUIRED_PROVENANCE_FILES if path not in included]
    if missing:
        raise RuntimeError(f"Review package is missing required inputs: {missing}")

    audit_path = ROOT / "results/six_method_v2/SIX_METHOD_V2_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    source = str(audit["drl_source"]).removeprefix("frozen ")
    if source not in included or not (ROOT / source).is_file():
        raise RuntimeError(f"V2 audit DRL source is not packaged: {source}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a self-contained Six-Method V2 code/result review ZIP."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "STAR_RIS_RSMA_six_method_v2_review.zip",
    )
    args = parser.parse_args()

    files = tracked_review_files()
    validate_inputs(files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for relative in files:
            source = ROOT / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            archive.write(source, f"{ARCHIVE_ROOT}/{relative.as_posix()}")

    print(f"created={args.output}")
    print(f"files={len(files)}")
    print(f"bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
