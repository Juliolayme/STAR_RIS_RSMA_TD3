from __future__ import annotations

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/siso_n32.yaml")
p.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
p.add_argument("--scenario-count", type=int, default=1000)
p.add_argument("--scenario-shard-size", type=int, default=100)
p.add_argument("--output", default="kaggle_jobs.json")
a = p.parse_args()
jobs = []
for method in ["td3", "ddpg", "ppo"]:
    for seed in a.seeds:
        jobs.append({"kind":"train", "method":method, "seed":seed, "command":f"python scripts/run_train.py --method {method} --config {a.config} --seed {seed} --output results/train/{method}/seed_{seed}"})
for method in ["ao_sca", "ao_grid", "analytical_ris"]:
    for start in range(0, a.scenario_count, a.scenario_shard_size):
        count = min(a.scenario_shard_size, a.scenario_count - start)
        jobs.append({"kind":"solver", "method":method, "start":start, "count":count, "command":f"python scripts/run_solver.py --method {method} --config {a.config} --start {start} --count {count} --output results/solvers/{method}_{start}_{start+count}.csv"})
Path(a.output).write_text(json.dumps(jobs, indent=2))
print(f"Wrote {len(jobs)} jobs to {a.output}")
