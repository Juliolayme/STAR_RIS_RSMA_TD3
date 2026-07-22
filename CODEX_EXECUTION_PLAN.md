# CODEX EXECUTION PLAN — STAR-RIS–RSMA TD3

## 1. Frozen research scope

Read this file before changing code or merging results into the thesis.

The thesis remains:

> TỐI ƯU PHÂN BỔ TÀI NGUYÊN SỬ DỤNG HỌC TĂNG CƯỜNG SÂU TRONG MẠNG STAR-RIS HỖ TRỢ RSMA

The comparison design is frozen as follows:

- TD3: primary DRL method.
- DDPG and PPO: DRL baselines.
- AO-SCA: primary conventional optimization baseline.
- AO-Grid and AnalyticalRIS: supplementary references.
- NoRIS, FixedRIS, RandomRIS and Equal-Power: ablations only.
- Do not reintroduce MADDPG, CTDE, TD3-Matched or old multi-agent claims.

## 2. Physical invariants

All six methods must use `src/star_ris_rsma/env.py` and `src/star_ris_rsma/physics.py`.

- SISO BS and users.
- STAR-RIS energy-splitting mode.
- `beta_t + beta_r = 1` for every element.
- Complex amplitudes use `sqrt(beta)`, not `beta`.
- Common RSMA stream is decoded by every user.
- Common rate is the minimum common decodable rate.
- Total power and common-rate shares are simplex-normalized.
- Report sum-rate, per-user QoS fraction, all-users QoS probability and QoS violation.
- Learned evaluation must be deterministic and exploration-free.

Do not alter the channel/rate equations only to improve one algorithm.

## 3. Mandatory validation before Kaggle

```bash
python -m pip install -e .[dev]
pytest -q
python scripts/run_train.py --method td3 --config configs/smoke.yaml --seed 0 --output results/smoke/td3
python scripts/run_train.py --method ddpg --config configs/smoke.yaml --seed 0 --output results/smoke/ddpg
python scripts/run_train.py --method ppo --config configs/smoke.yaml --seed 0 --output results/smoke/ppo
python scripts/run_solver.py --method ao_sca --config configs/smoke.yaml --start 0 --count 4 --output results/smoke/ao_sca.csv
python scripts/run_solver.py --method ao_grid --config configs/smoke.yaml --start 0 --count 4 --output results/smoke/ao_grid.csv
python scripts/run_solver.py --method analytical_ris --config configs/smoke.yaml --start 0 --count 4 --output results/smoke/analytical_ris.csv
```

Acceptance conditions:

- all tests pass;
- all outputs are finite;
- power/common shares sum correctly;
- STAR energy splitting is feasible;
- AO-SCA objective history is non-decreasing within tolerance;
- AO-SCA and AO-Grid remain separate implementations;
- deterministic evaluation returns identical actions for identical observations.

## 4. Kaggle execution

### Setup

```bash
cd /kaggle/working
git clone https://github.com/Juliolayme/STAR_RIS_RSMA_TD3.git
cd STAR_RIS_RSMA_TD3
pip install -q -e .
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY
```

### Generate shard commands

```bash
python scripts/make_kaggle_jobs.py \
  --config configs/siso_n32.yaml \
  --seeds 0 1 2 3 4 5 6 7 \
  --scenario-count 1000 \
  --scenario-shard-size 100 \
  --output kaggle_jobs.json
```

This produces 24 learned shards and 10 scenario shards for each traditional solver.
Run one learned seed per Kaggle GPU session. Traditional methods may run on CPU.
Every shard must preserve its command, config, seed/range and output files.

### Merge solver shards

```bash
python scripts/merge_results.py \
  --inputs results/solvers/*.csv \
  --output results/merged/traditional.csv
```

The merger must abort on duplicate `(method, seed, scenario)` keys.

## 5. Required next code work for Codex

Complete these before full thesis results are accepted:

1. Add a locked ScenarioBank (`train/validation/test`) saved as NPZ.
2. Add checkpoint loading and deterministic evaluation for TD3/DDPG/PPO.
3. Select best checkpoints using validation only.
4. Add raw per-scenario CSV for learned methods.
5. Add CPU single-thread latency benchmark with warm-up.
6. Add paired statistical tests, 95% confidence intervals and Holm correction.
7. Add NoRIS, FixedRIS, RandomRIS and Equal-Power under a separate ablation module.
8. Add configuration files for N = 16, 32, 64, 96 and 128.
9. Add plots generated only from merged raw CSV files.
10. Record Git commit SHA and configuration hash in every result manifest.

Do not claim the current AO-SCA implementation is a global optimum or upper bound. It is a local first-order alternating solver with finite-difference gradients and monotone backtracking.

## 6. Thesis merge contract

Only update the thesis after the full locked experiment is complete.

Replace the old method story with:

> TD3 is the primary DRL method; DDPG and PPO are DRL baselines; AO-SCA is the primary conventional optimization baseline; AO-Grid and AnalyticalRIS are supplementary references.

The thesis must report sum-rate, user QoS fraction, all-users QoS probability, constraint violation, mean and 95% CI, inference/solve latency, and AO iteration/evaluation counts.

Do not reuse old MADDPG numerical values, figures, significance statements or scalability conclusions. Generate new tables and figures from this repository's locked result files.
