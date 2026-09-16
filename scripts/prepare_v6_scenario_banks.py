from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.scenario_bank import ScenarioBank, assert_disjoint, generate_bank


N_VALUES = (16, 32, 64, 96, 128)
SPLITS = {
    "train": (10_000, 11_001),
    "validation": (1_000, 22_001),
    "test": (1_000, 33_001),
}
FROZEN_TEST_CHECKSUMS = {
    16: "ee98d51d23f2f29facb379bf4250dad3e33d568085dfd332a582372e84151ce7",
    32: "6ec54735c1c2c35cd253d9d3d23295783293b6d64c65204dfcab16b2d59ebbef",
    64: "ae41070d268d283dcc283b0aaf9b960eae1f4ddd22d423577b272c18802a3aae",
    96: "4e84455d08cacd9f85c432f7838c3be2cb0f000b55edc3487c6dee9520cc0b63",
    128: "f4c80269e5fb3cf553900b2e82f235af875b7c2e33b4ff71ec5d85cc25eb2b4e",
}
CANONICAL_ARTIFACT = {
    "repository": "Juliolayme/STAR_RIS_RSMA_TD3",
    "workflow_run_id": 30_422_028_560,
    "artifact_id": 8_712_801_218,
    "artifact_name": "star-ris-six-method-canonical-evidence",
    "url": (
        "https://github.com/Juliolayme/STAR_RIS_RSMA_TD3/actions/runs/"
        "30422028560/artifacts/8712801218"
    ),
}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/scenario_banks"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/physical_v6_full/SCENARIO_BANK_MANIFEST.json"),
    )
    parser.add_argument(
        "--n-ris",
        type=int,
        nargs="+",
        choices=N_VALUES,
        default=list(N_VALUES),
        help="RIS sizes to generate/verify; defaults to all five frozen sizes.",
    )
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()

    manifest: dict[str, object] = {
        "audit": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "generator": "star_ris_rsma.physics.generate_channel/v1",
        "bank_origin": (
            {"mode": "verified_existing", **CANONICAL_ARTIFACT}
            if args.verify_existing
            else {"mode": "generated_locally"}
        ),
        "banks": {},
    }
    for n_ris in args.n_ris:
        cfg_path = Path(f"configs/v3/pilot_v6_soft_anchor_n{n_ris}.yaml")
        cfg = ExperimentConfig.from_yaml(cfg_path)
        banks: dict[str, ScenarioBank] = {}
        # Captured before generation, which overwrites the file it would be
        # compared against.
        stored_test: ScenarioBank | None = None
        stored_path = args.output_dir / f"N{n_ris}_test.npz"
        if stored_path.is_file():
            try:
                stored_test = ScenarioBank.load(stored_path, cfg)
            except (OSError, ValueError):
                stored_test = None
        for split, (count, seed) in SPLITS.items():
            path = args.output_dir / f"N{n_ris}_{split}.npz"
            if args.verify_existing and path.is_file():
                bank = ScenarioBank.load(path, cfg)
            else:
                bank = generate_bank(cfg, count, seed, split)
                bank.save(path)
            if len(bank) != count or bank.metadata.get("seed") != seed:
                raise RuntimeError(f"N={n_ris} {split}: invalid count or seed")
            banks[split] = bank
        assert_disjoint(*banks.values())
        test_checksum = banks["test"].checksum()
        if test_checksum != FROZEN_TEST_CHECKSUMS[n_ris]:
            # ScenarioBank.checksum hashes raw bytes, so a single float differing
            # by one ULP between numpy builds changes the whole digest while the
            # channels stay numerically identical. Say which of the two it is
            # rather than leaving a rebuild on other hardware to guess.
            detail = ""
            if stored_test is not None:
                stored = stored_test
                gaps = {
                    name: float(
                        np.max(np.abs(getattr(stored, name) - getattr(banks["test"], name)))
                    )
                    for name in ("h_direct", "g_br", "h_ru")
                    if getattr(stored, name).shape == getattr(banks["test"], name).shape
                }
                worst = max(gaps.values(), default=float("inf"))
                detail = (
                    f"; largest elementwise gap against the stored bank is {worst:.3e} "
                    + (
                        "which is floating-point rounding, so the data agrees and only "
                        "the byte-exact digest differs"
                        if worst < 1e-12
                        else "which is a real difference in the generated channels"
                    )
                )
            raise RuntimeError(
                f"N={n_ris}: test checksum {test_checksum} != frozen "
                f"{FROZEN_TEST_CHECKSUMS[n_ris]}{detail}"
            )
        manifest["banks"][str(n_ris)] = {
            "config": cfg_path.as_posix(),
            "config_hash": cfg.config_hash(),
            "splits": {
                split: {
                    "path": (args.output_dir / f"N{n_ris}_{split}.npz").as_posix(),
                    "count": len(bank),
                    "seed": int(bank.metadata["seed"]),
                    "checksum": bank.checksum(),
                }
                for split, bank in banks.items()
            },
            "frozen_test_checksum_match": True,
        }
        print(f"N={n_ris}: test checksum PASS {test_checksum}", flush=True)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"audit": "PASS", "manifest": str(args.manifest)}, indent=2))


if __name__ == "__main__":
    main()
