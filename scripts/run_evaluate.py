from __future__ import annotations

import argparse
from pathlib import Path

from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.experiment import evaluate_checkpoint
from star_ris_rsma.scenario_bank import ScenarioBank

p = argparse.ArgumentParser()
p.add_argument("--method", choices=["td3", "ddpg", "ppo"], required=True)
p.add_argument("--config", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--bank", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
cfg = ExperimentConfig.from_yaml(a.config)
bank = ScenarioBank.load(a.bank, cfg)
evaluate_checkpoint(a.method, cfg, Path(a.checkpoint), bank, a.seed, Path(a.output))
