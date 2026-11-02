from pathlib import Path

from star_ris_rsma.config import ExperimentConfig


N_VALUES = (16, 32, 64, 96, 128)


def test_v6_configs_share_one_fixed_budget_protocol() -> None:
    reference = None
    ignored = {"n_ris", "train_bank_path", "validation_bank_path", "test_bank_path"}
    for n_ris in N_VALUES:
        path = Path(f"configs/v3/pilot_v6_soft_anchor_n{n_ris}.yaml")
        assert path.is_file()
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.n_ris == n_ris
        assert cfg.action_parameterization == "physical_v6_soft_anchor"
        assert cfg.train_steps == 100_000
        assert cfg.validation_scenarios == 1_000
        assert cfg.train_bank_path == f"artifacts/scenario_banks/N{n_ris}_train.npz"
        assert cfg.validation_bank_path == f"artifacts/scenario_banks/N{n_ris}_validation.npz"
        assert cfg.test_bank_path == f"artifacts/scenario_banks/N{n_ris}_test.npz"
        protocol = {key: value for key, value in cfg.to_dict().items() if key not in ignored}
        if reference is None:
            reference = protocol
        else:
            assert protocol == reference
