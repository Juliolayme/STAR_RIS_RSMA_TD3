from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd
import torch

from star_ris_rsma.baselines import analytical_ris, ao_grid, ao_sca
from star_ris_rsma.checkpoints import load_checkpoint
from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.env import StarRisRsmaEnv
from star_ris_rsma.scenario_bank import ScenarioBank

p = argparse.ArgumentParser()
p.add_argument("--method", choices=["td3", "ddpg", "ppo", "ao_sca", "ao_grid", "analytical_ris"], required=True)
p.add_argument("--config", required=True)
p.add_argument("--bank", required=True)
p.add_argument("--checkpoint")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--count", type=int, default=100)
p.add_argument("--output", required=True)
a = p.parse_args()

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
cfg = ExperimentConfig.from_yaml(a.config)
bank = ScenarioBank.load(a.bank, cfg)
env = StarRisRsmaEnv(cfg, a.seed)
learned = a.method in {"td3", "ddpg", "ppo"}
agent = None
if learned:
    if not a.checkpoint:
        raise SystemExit("--checkpoint is required for learned methods")
    agent, _ = load_checkpoint(a.checkpoint, a.method, env.observation_dim, env.action_dim, cfg, "cpu")

rows = []
limit = min(len(bank), a.warmup + a.count)
for scenario in range(limit):
    obs = env.reset(channel=bank.channel(scenario))
    start = time.perf_counter_ns()
    if learned:
        if a.method == "ppo": raw, _, _ = agent.act(obs, deterministic=True)
        else: raw = agent.act(obs, noise_std=0.0)
        inference_ms = (time.perf_counter_ns() - start) / 1e6
        end_start = time.perf_counter_ns()
        env.evaluate_raw_action(raw)
        evaluation_ms = (time.perf_counter_ns() - end_start) / 1e6
        solve_ms = inference_ms + evaluation_ms
    else:
        if a.method == "ao_sca": ao_sca(env, seed=a.seed + scenario)
        elif a.method == "ao_grid": ao_grid(env, seed=a.seed + scenario)
        else: analytical_ris(env)
        solve_ms = (time.perf_counter_ns() - start) / 1e6
        inference_ms = float("nan")
        evaluation_ms = float("nan")
    if scenario >= a.warmup:
        rows.append({
            "method": a.method,
            "scenario": scenario - a.warmup,
            "inference_ms": inference_ms,
            "evaluation_ms": evaluation_ms,
            "solve_ms": solve_ms,
            "cpu_threads": 1,
            "seed": a.seed,
            "config_hash": cfg.config_hash(),
            "bank_checksum": bank.checksum(),
        })
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(a.output, index=False)
