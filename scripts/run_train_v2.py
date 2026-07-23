from __future__ import annotations

import argparse
from pathlib import Path

from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.experiment_v2 import train_td3_v2


p = argparse.ArgumentParser()
p.add_argument("--method", choices=["td3"], default="td3")
p.add_argument("--config", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--output", required=True)
a = p.parse_args()

cfg = ExperimentConfig.from_yaml(a.config)
train_td3_v2(cfg, a.seed, Path(a.output))
