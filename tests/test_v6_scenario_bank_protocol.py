from pathlib import Path
import runpy
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = runpy.run_path(ROOT / "scripts" / "prepare_v6_scenario_banks.py")
CANONICAL_ARTIFACT = PROTOCOL["CANONICAL_ARTIFACT"]
FROZEN_TEST_CHECKSUMS = PROTOCOL["FROZEN_TEST_CHECKSUMS"]
N_VALUES = PROTOCOL["N_VALUES"]
SPLITS = PROTOCOL["SPLITS"]


def test_v6_bank_protocol_is_the_frozen_six_method_protocol() -> None:
    assert N_VALUES == (16, 32, 64, 96, 128)
    assert SPLITS == {
        "train": (10_000, 11_001),
        "validation": (1_000, 22_001),
        "test": (1_000, 33_001),
    }
    assert set(FROZEN_TEST_CHECKSUMS) == set(N_VALUES)
    assert all(len(checksum) == 64 for checksum in FROZEN_TEST_CHECKSUMS.values())
    assert CANONICAL_ARTIFACT["artifact_id"] == 8_712_801_218
    assert CANONICAL_ARTIFACT["workflow_run_id"] == 30_422_028_560


def test_v6_bank_preparer_accepts_one_ris_size() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_v6_scenario_banks.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--n-ris" in completed.stdout


def test_user_side_width_is_pinned_so_the_checksum_is_portable() -> None:
    """ScenarioBank.checksum hashes raw bytes.

    numpy's default integer is int32 on Windows and int64 on Linux, so leaving
    user_side to the platform made the frozen bank checksums unreproducible
    off Linux while the channel data itself was identical.
    """
    import numpy as np

    from star_ris_rsma.physics import generate_channel

    channel = generate_channel(np.random.default_rng(0), 4, 16)
    assert channel.user_side.dtype == np.int64
    assert channel.user_side.tolist() == [0, 1, 0, 1]


def test_the_frozen_n16_checksum_reproduces_from_source() -> None:
    import runpy
    from pathlib import Path

    from star_ris_rsma.config import ExperimentConfig
    from star_ris_rsma.scenario_bank import generate_bank

    root = Path(__file__).resolve().parents[1]
    frozen = runpy.run_path(
        root / "scripts" / "prepare_v6_scenario_banks.py"
    )["FROZEN_TEST_CHECKSUMS"]
    cfg = ExperimentConfig.from_yaml(root / "configs/v3/pilot_v6_soft_anchor_n16.yaml")
    assert generate_bank(cfg, 1000, 33001, "test").checksum() == frozen[16]
