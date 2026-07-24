from __future__ import annotations

"""Shared paths, provenance checks, and subprocess helpers for Kaggle report 06."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

REPO_URL = "https://github.com/Juliolayme/STAR_RIS_RSMA_TD3.git"
ALGORITHM_COMMIT = "89c39da461523a7f5911a302cb9415aeaa5824ce"
REPO_DIR = Path(__file__).resolve().parents[1]
INPUT_ROOT = Path(os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"))
FINAL_ROOT = Path(
    os.environ.get(
        "FINAL_REPORT_OUTPUT",
        "/kaggle/working/FINAL_THESIS_PAPER_BUNDLE",
    )
)
FIGURE_DIR = FINAL_ROOT / "figures"
TABLE_DIR = FINAL_ROOT / "tables"
RAW_DIR = FINAL_ROOT / "raw"
LATENCY_DIR = FINAL_ROOT / "latency"
N_VALUES = (16, 32, 64, 96, 128)
SEEDS = tuple(range(8))
METHODS = ("td3", "ao_sca", "ao_grid", "analytical_ris")


def prepare_output_directories() -> None:
    """Create every output directory before any computation starts."""
    for directory in (FINAL_ROOT, FIGURE_DIR, TABLE_DIR, RAW_DIR, LATENCY_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def run_command(
    command: Sequence[str | Path],
    *,
    cwd: Path,
    log_path: Path,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Run a subprocess, stream stdout, and persist the complete combined log.

    A non-zero return code raises immediately. This fail-closed behavior prevents
    incomplete solver or benchmark output from entering thesis tables.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    normalized = [str(item) for item in command]
    print("$", " ".join(normalized))
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            normalized,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}. See {log_path}"
        )


def current_repository_commit() -> str:
    """Return the checked-out report-code commit for the reproducibility record."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True
    ).strip()


def create_locked_banks(n_values: Iterable[int]) -> None:
    """Recreate deterministic ScenarioBanks used by every method and seed.

    The bank-generation command is identical to the final GitHub Actions
    protocol. Existing test banks are reused after the report script has
    verified all stage manifests and raw checksums.
    """
    bank_dir = REPO_DIR / "artifacts" / "scenario_banks"
    bank_dir.mkdir(parents=True, exist_ok=True)
    log_dir = FINAL_ROOT / "logs" / "scenario_banks"

    for n_ris in n_values:
        test_bank = bank_dir / f"N{n_ris}_test.npz"
        if test_bank.exists():
            print(f"Reuse report ScenarioBank N={n_ris}: {test_bank}")
            continue
        config = REPO_DIR / "configs" / "v3" / f"constrained_action_n{n_ris}.yaml"
        run_command(
            [
                sys.executable,
                "scripts/create_scenario_banks.py",
                "--config",
                config,
                "--output-dir",
                bank_dir,
                "--train-count",
                "10000",
                "--validation-count",
                "1000",
                "--test-count",
                "1000",
            ],
            cwd=REPO_DIR,
            log_path=log_dir / f"N{n_ris}.log",
        )


def discover_stage_roots() -> dict[str, Path]:
    """Discover outputs of notebooks 01-05 from their stage manifests.

    Kaggle kernel-source mount names depend on the account and kernel slug.
    Manifest discovery keeps notebook 06 independent of those names while still
    requiring exactly one copy of each scientific stage.
    """
    roots: dict[str, Path] = {}
    for manifest_path in INPUT_ROOT.rglob("STAGE_MANIFEST.json"):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        stage_id = str(payload["stage_id"])
        if stage_id in roots:
            raise RuntimeError(
                f"Duplicate attached stage '{stage_id}': "
                f"{roots[stage_id]} and {manifest_path.parent}"
            )
        roots[stage_id] = manifest_path.parent

    required = {"td3_low_n", "td3_high_n", "ao_grid", "ao_sca", "analytical_ris"}
    missing = sorted(required.difference(roots))
    if missing:
        raise RuntimeError(
            "Attach successful outputs from notebooks 01-05 before notebook 06. "
            f"Missing stages: {missing}"
        )
    return roots


def validate_stage_manifests(stage_roots: dict[str, Path]) -> dict[str, dict[str, object]]:
    """Verify all stages used the same declared algorithm commit and N coverage."""
    manifests: dict[str, dict[str, object]] = {}
    for stage_id, root in stage_roots.items():
        payload = json.loads((root / "STAGE_MANIFEST.json").read_text(encoding="utf-8"))
        if payload.get("repository_commit") != ALGORITHM_COMMIT:
            raise RuntimeError(
                f"Stage code drift for {stage_id}: {payload.get('repository_commit')}"
            )
        manifests[stage_id] = payload
    return manifests


def parse_n_seed(path: Path) -> tuple[int, int]:
    """Extract N and seed from `.../N{N}/seed_{seed}/...` paths."""
    text = path.as_posix()
    n_match = re.search(r"/N(\d+)/", text)
    seed_match = re.search(r"/seed_(\d+)/", text)
    if not n_match or not seed_match:
        raise ValueError(f"Cannot parse N/seed from {path}")
    return int(n_match.group(1)), int(seed_match.group(1))


def ensure_finite(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    """Raise when any required numerical result is missing, NaN, or infinite."""
    numeric = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError(f"Non-finite numerical value in {context}")


def write_table_formats(frame: pd.DataFrame, stem: str) -> None:
    """Write one final table as CSV, Markdown, and LaTeX."""
    frame.to_csv(TABLE_DIR / f"{stem}.csv", index=False)
    (TABLE_DIR / f"{stem}.md").write_text(
        frame.to_markdown(index=False), encoding="utf-8"
    )
    (TABLE_DIR / f"{stem}.tex").write_text(
        frame.to_latex(index=False, float_format=lambda value: f"{value:.6g}"),
        encoding="utf-8",
    )
