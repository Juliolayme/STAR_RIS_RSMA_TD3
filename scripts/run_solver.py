from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.experiment import evaluate_solver
from star_ris_rsma.scenario_bank import ScenarioBank

p = argparse.ArgumentParser()
p.add_argument("--method", choices=["ao_sca", "ao_grid", "analytical_ris"], required=True)
p.add_argument("--config", required=True)
p.add_argument("--bank", required=True)
p.add_argument("--seed", type=int, default=10000)
p.add_argument("--start", type=int, default=0)
p.add_argument("--count", type=int, required=True)
p.add_argument("--output", required=True)
a = p.parse_args()

cfg = ExperimentConfig.from_yaml(a.config)
bank = ScenarioBank.load(a.bank, cfg)
output = Path(a.output)
evaluate_solver(a.method, cfg, a.seed, a.start, a.count, output, bank)

# Baseline notebooks aggregate chunk CSVs across several RIS sizes.  Keep the
# system dimension in every row so downstream auditing never has to infer it
# from a directory name.
frame = pd.read_csv(output)
expected_n_ris = int(cfg.n_ris)
if "n_ris" in frame.columns:
    observed = pd.to_numeric(frame["n_ris"], errors="raise").astype(int)
    if not (observed == expected_n_ris).all():
        raise RuntimeError(
            f"Inconsistent n_ris in {output}: expected {expected_n_ris}, "
            f"observed={sorted(observed.unique().tolist())}"
        )
else:
    frame.insert(1, "n_ris", expected_n_ris)
frame.to_csv(output, index=False)
