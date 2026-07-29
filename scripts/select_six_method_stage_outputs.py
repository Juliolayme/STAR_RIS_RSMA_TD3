from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


EXPECTED_STAGES = (
    "td3_low_n",
    "td3_high_n",
    "ddpg_low_n",
    "ddpg_high_n",
    "ppo_low_n",
    "ppo_high_n",
)
EXPECTED_PROTOCOL = "drl_v3_qos_constrained_fair"
FORBIDDEN_PARTS = {"STAR_RIS_RSMA_TD3", "kaggle_runs", ".git", "results"}


def load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_manifest(stage_root: Path, stage_id: str) -> tuple[Path, dict[str, object]]:
    candidates: list[tuple[int, Path, dict[str, object]]] = []
    rejected: list[str] = []
    for path in sorted(stage_root.rglob("STAGE_MANIFEST.json")):
        try:
            manifest = load_manifest(path)
        except Exception as exc:
            rejected.append(f"{path}: unreadable ({exc})")
            continue
        if manifest.get("stage_id") != stage_id:
            continue
        if manifest.get("training_protocol") != EXPECTED_PROTOCOL:
            rejected.append(f"{path}: protocol={manifest.get('training_protocol')!r}")
            continue
        relative = path.relative_to(stage_root)
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            rejected.append(f"{path}: nested repository/archive snapshot")
            continue
        candidates.append((len(relative.parts), path, manifest))

    if not candidates:
        detail = "\n".join(rejected) if rejected else "no matching manifests found"
        raise RuntimeError(f"No canonical manifest for {stage_id} under {stage_root}:\n{detail}")

    minimum_depth = min(item[0] for item in candidates)
    shallowest = [item for item in candidates if item[0] == minimum_depth]
    if len(shallowest) != 1:
        paths = [str(item[1]) for item in shallowest]
        raise RuntimeError(f"Ambiguous canonical manifests for {stage_id}: {paths}")
    _, path, manifest = shallowest[0]
    return path, manifest


def copy_selected_stage(manifest_path: Path, destination: Path) -> None:
    source = manifest_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, destination / "STAGE_MANIFEST.json")

    test_csvs = sorted(source.glob("*_TEST_RAW_ALL.csv"))
    if len(test_csvs) != 1:
        raise RuntimeError(
            f"Expected exactly one TEST_RAW_ALL CSV beside {manifest_path}, got {test_csvs}"
        )
    shutil.copy2(test_csvs[0], destination / test_csvs[0].name)

    for pattern in ("*_TRAINING_RAW.csv", "*_VALIDATION_RAW.csv", "STAGE_STATUS.csv"):
        for path in source.glob(pattern):
            if path.is_file():
                shutil.copy2(path, destination / path.name)

    figure_source = source / "figures"
    if figure_source.is_dir():
        shutil.copytree(figure_source, destination / "figures", dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drl-root", type=Path, required=True)
    parser.add_argument("--selected-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()

    if args.selected_root.exists():
        shutil.rmtree(args.selected_root)
    args.selected_root.mkdir(parents=True, exist_ok=True)

    selected: dict[str, dict[str, object]] = {}
    commits: set[str] = set()
    for stage_id in EXPECTED_STAGES:
        stage_root = args.drl_root / stage_id
        if not stage_root.is_dir():
            raise RuntimeError(f"Missing downloaded stage directory: {stage_root}")
        manifest_path, manifest = select_manifest(stage_root, stage_id)
        repository_commit = str(manifest.get("repository_commit", ""))
        if not repository_commit:
            raise RuntimeError(f"Manifest has no repository_commit: {manifest_path}")
        commits.add(repository_commit)
        copy_selected_stage(manifest_path, args.selected_root / stage_id)
        selected[stage_id] = {
            "manifest_path": str(manifest_path),
            "stage_root": str(manifest_path.parent),
            "repository_commit": repository_commit,
            "n_values": manifest.get("n_values"),
            "scenario_bank_checksums": manifest.get("scenario_bank_checksums"),
        }

    if len(commits) != 1:
        details = {key: value["repository_commit"] for key, value in selected.items()}
        raise RuntimeError(f"Canonical DRL stages used different commits: {details}")

    payload = {
        "repository_commit": next(iter(commits)),
        "training_protocol": EXPECTED_PROTOCOL,
        "selected_stages": selected,
    }
    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.index.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
