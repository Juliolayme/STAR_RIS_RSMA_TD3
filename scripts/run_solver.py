from __future__ import annotations

import argparse
from pathlib import Path

from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.experiment import evaluate_solver

p = argparse.ArgumentParser()
p.add_argument("--method", choices=["ao_sca", "ao_grid", "analytical_ris"], required=True)
p.add_argument("--config", required=True)
p.add_argument("--seed", type=int, default=10000)
p.add_argument("--start", type=int, default=0)
p.add_argument("--count", type=int, required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
evaluate_solver(a.method, ExperimentConfig.from_yaml(a.config), a.seed, a.start, a.count, Path(a.output))
