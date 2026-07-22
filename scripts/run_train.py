from __future__ import annotations

import argparse
from pathlib import Path

from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.experiment import train_off_policy, train_ppo

p = argparse.ArgumentParser()
p.add_argument("--method", choices=["td3", "ddpg", "ppo"], required=True)
p.add_argument("--config", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
cfg = ExperimentConfig.from_yaml(a.config)
if a.method == "ppo": train_ppo(cfg, a.seed, Path(a.output))
else: train_off_policy(a.method, cfg, a.seed, Path(a.output))
