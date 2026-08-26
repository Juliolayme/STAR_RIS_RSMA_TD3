from __future__ import annotations

import argparse
from pathlib import Path

from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.experiment_v3 import train_drl_v3


parser = argparse.ArgumentParser(
    description="Train TD3, DDPG, or PPO with the QoS-constrained v3 protocol used by the thesis."
)
parser.add_argument("--method", choices=["td3", "ddpg", "ppo"], required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

config = ExperimentConfig.from_yaml(args.config)

# The generic entrypoint is intentionally restricted to the thesis protocol so
# legacy YAML files cannot silently reproduce a different state/action/reward
# definition under the same command name. Historical runs remain available via
# their explicitly versioned scripts.
required = {
    "observation_normalization": "blockwise_v2",
    "action_parameterization": "physical_v3",
    "qos_dual_enabled": True,
}
for field, expected in required.items():
    observed = getattr(config, field)
    if observed != expected:
        raise SystemExit(
            f"{field}={observed!r} is not the thesis v3 protocol; expected {expected!r}. "
            "Use a v3 constrained-action configuration."
        )

for field in ("train_bank_path", "validation_bank_path", "test_bank_path"):
    if not getattr(config, field):
        raise SystemExit(
            f"{field} must point to a locked ScenarioBank for the thesis v3 protocol."
        )

train_drl_v3(args.method, config, args.seed, Path(args.output))
