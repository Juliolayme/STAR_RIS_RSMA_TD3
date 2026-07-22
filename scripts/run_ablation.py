from __future__ import annotations

import argparse
from pathlib import Path
import os
import subprocess

import pandas as pd

from star_ris_rsma.baselines.ablations import ABLATION_MODES, evaluate_ablation
from star_ris_rsma.checkpoints import load_checkpoint
from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.env import StarRisRsmaEnv
from star_ris_rsma.scenario_bank import ScenarioBank

p = argparse.ArgumentParser()
p.add_argument("--method", choices=["td3", "ddpg", "ppo"], default="td3")
p.add_argument("--config", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--bank", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--modes", nargs="+", choices=list(ABLATION_MODES), default=list(ABLATION_MODES))
p.add_argument("--output", required=True)
a = p.parse_args()

cfg = ExperimentConfig.from_yaml(a.config)
bank = ScenarioBank.load(a.bank, cfg)
env = StarRisRsmaEnv(cfg, a.seed)
agent, payload = load_checkpoint(a.checkpoint, a.method, env.observation_dim, env.action_dim, cfg, "cpu")
try:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
except (OSError, subprocess.CalledProcessError):
    git_commit = os.environ.get("GIT_COMMIT", "unknown")
rows = []
for scenario in range(len(bank)):
    obs = env.reset(channel=bank.channel(scenario))
    if a.method == "ppo":
        raw, _, _ = agent.act(obs, deterministic=True)
    else:
        raw = agent.act(obs, noise_std=0.0)
    for mode in a.modes:
        metrics = evaluate_ablation(env, raw, mode, seed=a.seed + scenario)
        rows.append({
            "method": f"{a.method}:{mode}",
            "base_method": a.method,
            "ablation": mode,
            "seed": a.seed,
            "scenario": scenario,
            "sum_rate": metrics["sum_rate"],
            "reward": metrics["reward"],
            "qos_fraction": metrics["qos_fraction"],
            "all_qos": metrics["all_qos"],
            "violation": metrics["violation"],
            "checkpoint_step": payload["step"],
            "config_hash": cfg.config_hash(),
            "git_commit": git_commit,
            "bank_checksum": bank.checksum(),
        })
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(a.output, index=False)
