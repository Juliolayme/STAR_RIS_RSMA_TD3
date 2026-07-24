from pathlib import Path

from star_ris_rsma.config import ExperimentConfig


N_VALUES = (16, 32, 64, 96, 128)
COMPUTE_FIELDS = (
    "hidden_dim",
    "batch_size",
    "replay_size",
    "warmup_steps",
    "train_steps",
    "validation_interval",
    "validation_scenarios",
)


def test_norm_only_changes_only_observation_normalization():
    for n_ris in N_VALUES:
        v1 = ExperimentConfig.from_yaml(Path(f"configs/siso_n{n_ris}.yaml"))
        norm_only = ExperimentConfig.from_yaml(
            Path(f"configs/ablations/norm_only_n{n_ris}.yaml")
        )

        differences = {
            key
            for key, value in v1.to_dict().items()
            if norm_only.to_dict()[key] != value
        }
        assert differences == {"observation_normalization"}
        assert norm_only.observation_normalization == "blockwise_v2"


def test_v2_fixed_budget_matches_v1_compute_budget():
    for n_ris in N_VALUES:
        v1 = ExperimentConfig.from_yaml(Path(f"configs/siso_n{n_ris}.yaml"))
        fixed = ExperimentConfig.from_yaml(
            Path(f"configs/ablations/v2_fixed_budget_n{n_ris}.yaml")
        )
        for field in COMPUTE_FIELDS:
            assert getattr(fixed, field) == getattr(v1, field), (n_ris, field)

        assert fixed.observation_normalization == "blockwise_v2"
        assert fixed.qos_penalty_linear == 8.0
        assert fixed.qos_penalty_quadratic == 4.0
        assert fixed.td3_critic_loss == "huber"
        assert fixed.td3_layer_norm is True
