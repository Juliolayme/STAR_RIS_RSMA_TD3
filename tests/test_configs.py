from pathlib import Path

from star_ris_rsma.config import ExperimentConfig


def test_scalability_configs_exist_and_match_n():
    for n in [16, 32, 64, 96, 128]:
        path = Path(f"configs/siso_n{n}.yaml")
        assert path.exists()
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.n_ris == n
        assert cfg.train_bank_path == f"artifacts/scenario_banks/N{n}_train.npz"
