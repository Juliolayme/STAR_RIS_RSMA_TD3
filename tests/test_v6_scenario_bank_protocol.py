from pathlib import Path
import runpy


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
