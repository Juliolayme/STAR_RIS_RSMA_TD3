from pathlib import Path

import numpy as np
import pytest

from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.scenario_bank import ScenarioBank, assert_disjoint, generate_bank


def test_scenario_bank_roundtrip(tmp_path: Path):
    cfg = ExperimentConfig(n_users=2, n_ris=3)
    bank = generate_bank(cfg, 5, 123, "test")
    path = tmp_path / "bank.npz"
    bank.save(path)
    loaded = ScenarioBank.load(path, cfg)
    assert loaded.checksum() == bank.checksum()
    assert np.allclose(loaded.h_ru, bank.h_ru)


def test_locked_splits_are_disjoint():
    cfg = ExperimentConfig(n_users=2, n_ris=3)
    assert_disjoint(
        generate_bank(cfg, 3, 1, "train"),
        generate_bank(cfg, 3, 2, "validation"),
        generate_bank(cfg, 3, 3, "test"),
    )


def test_duplicate_banks_are_rejected():
    cfg = ExperimentConfig(n_users=2, n_ris=3)
    bank = generate_bank(cfg, 2, 1, "train")
    with pytest.raises(ValueError):
        assert_disjoint(bank, bank)
