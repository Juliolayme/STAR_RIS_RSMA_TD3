from pathlib import Path

from star_ris_rsma.config import ExperimentConfig


N_VALUES = (16, 32, 64, 96, 128)


def test_constrained_v3_configs_share_one_fixed_budget_protocol() -> None:
    reference = None
    ignored = {"n_ris", "train_bank_path", "validation_bank_path", "test_bank_path"}
    for n_ris in N_VALUES:
        path = Path(f"configs/v3/constrained_action_n{n_ris}.yaml")
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.n_ris == n_ris
        assert cfg.action_parameterization == "physical_v3"
        assert cfg.observation_normalization == "blockwise_v2"
        assert cfg.qos_dual_enabled is True
        assert cfg.hidden_dim == 256
        assert cfg.train_steps == 100_000
        assert cfg.validation_scenarios == 1_000
        assert cfg.validation_qos_fraction_target == 0.99
        assert cfg.validation_all_qos_target == 0.95
        assert cfg.validation_violation_tolerance == 0.001
        assert cfg.qos_dual_target_violation == 0.001
        assert cfg.qos_dual_min <= cfg.qos_dual_initial <= cfg.qos_dual_max

        protocol = {
            key: value
            for key, value in cfg.to_dict().items()
            if key not in ignored
        }
        if reference is None:
            reference = protocol
        else:
            assert protocol == reference
