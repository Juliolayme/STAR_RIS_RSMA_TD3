# STAR-RIS–RSMA TD3 Resource Optimization

Code-only research repository for:

> Tối ưu phân bổ tài nguyên sử dụng học tăng cường sâu trong mạng STAR-RIS hỗ trợ RSMA

## Frozen comparison design

- **TD3**: primary DRL method.
- **DDPG, PPO**: DRL baselines.
- **AO-SCA**: primary conventional optimization baseline.
- **AO-Grid, AnalyticalRIS**: supplementary references.
- MADDPG and TD3-Matched are intentionally excluded from the main pipeline.

All methods share one SISO STAR-RIS energy-splitting environment and one RSMA rate calculator.

## Install and test

```bash
python -m pip install -e .[dev]
pytest -q
```

## Smoke commands

```bash
python scripts/run_train.py --method td3 --config configs/smoke.yaml --seed 0 --output results/smoke/td3
python scripts/run_train.py --method ddpg --config configs/smoke.yaml --seed 0 --output results/smoke/ddpg
python scripts/run_train.py --method ppo --config configs/smoke.yaml --seed 0 --output results/smoke/ppo
python scripts/run_solver.py --method ao_sca --config configs/smoke.yaml --start 0 --count 4 --output results/smoke/ao_sca.csv
python scripts/run_solver.py --method ao_grid --config configs/smoke.yaml --start 0 --count 4 --output results/smoke/ao_grid.csv
python scripts/run_solver.py --method analytical_ris --config configs/smoke.yaml --start 0 --count 4 --output results/smoke/analytical_ris.csv
```

## Kaggle sharding

```bash
python scripts/make_kaggle_jobs.py --config configs/siso_n32.yaml --output kaggle_jobs.json
```

Each Kaggle notebook should execute one entry from `kaggle_jobs.json`. Merge CSV shards with:

```bash
python scripts/merge_results.py --inputs results/solvers/*.csv --output results/merged/traditional.csv
```

Read [`CODEX_EXECUTION_PLAN.md`](CODEX_EXECUTION_PLAN.md) before extending the code or merging results into the thesis.

## Scientific cautions

- AO-SCA is a local monotone first-order SCA baseline, not a global optimum or upper bound.
- AO-Grid is a coarse search heuristic and is implemented separately from AO-SCA.
- Learned-policy evaluation must be deterministic; exploration noise is training-only.
- Do not use test scenarios for checkpoint selection or tuning.
