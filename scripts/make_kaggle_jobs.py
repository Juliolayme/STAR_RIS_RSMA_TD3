from __future__ import annotations

import argparse
import json
from pathlib import Path

from star_ris_rsma.config import ExperimentConfig

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/siso_n32.yaml")
p.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
p.add_argument("--test-bank")
p.add_argument("--scenario-count", type=int, default=1000)
p.add_argument("--scenario-shard-size", type=int, default=100)
p.add_argument("--output", default="kaggle_jobs.json")
a = p.parse_args()
cfg = ExperimentConfig.from_yaml(a.config)
test_bank = a.test_bank or cfg.test_bank_path
if not test_bank:
    raise SystemExit("A locked --test-bank or test_bank_path in the config is required")

jobs: list[dict[str, object]] = []
for method in ["td3", "ddpg", "ppo"]:
    for seed in a.seeds:
        train_dir = f"results/train/{method}/N{cfg.n_ris}/seed_{seed}"
        jobs.append({
            "kind": "train",
            "method": method,
            "seed": seed,
            "command": (
                f"python scripts/run_train.py --method {method} --config {a.config} "
                f"--seed {seed} --output {train_dir}"
            ),
        })
        jobs.append({
            "kind": "evaluate",
            "method": method,
            "seed": seed,
            "depends_on": train_dir,
            "command": (
                f"python scripts/run_evaluate.py --method {method} --config {a.config} "
                f"--checkpoint {train_dir}/best.pt --bank {test_bank} --seed {seed} "
                f"--output results/test/{method}/N{cfg.n_ris}/seed_{seed}.csv"
            ),
        })
for method in ["ao_sca", "ao_grid", "analytical_ris"]:
    for start in range(0, a.scenario_count, a.scenario_shard_size):
        count = min(a.scenario_shard_size, a.scenario_count - start)
        jobs.append({
            "kind": "solver",
            "method": method,
            "start": start,
            "count": count,
            "command": (
                f"python scripts/run_solver.py --method {method} --config {a.config} "
                f"--bank {test_bank} --start {start} --count {count} "
                f"--output results/solvers/N{cfg.n_ris}/{method}_{start}_{start+count}.csv"
            ),
        })
Path(a.output).write_text(json.dumps(jobs, indent=2))
print(f"Wrote {len(jobs)} jobs to {a.output}")
