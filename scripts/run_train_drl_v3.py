from __future__ import annotations

import argparse
from pathlib import Path

from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.experiment_v3 import train_drl_v3


parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=["td3", "ddpg", "ppo"], required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

config = ExperimentConfig.from_yaml(args.config)
train_drl_v3(args.method, config, args.seed, Path(args.output))
