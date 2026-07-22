# STAR-RIS–RSMA TD3 Resource Optimization

Code-only research repository for:

> Tối ưu phân bổ tài nguyên sử dụng học tăng cường sâu trong mạng STAR-RIS hỗ trợ RSMA

## Frozen comparison design

- **TD3**: primary DRL method.
- **DDPG, PPO**: DRL baselines.
- **AO-SCA**: primary conventional optimization baseline.
- **AO-Grid, AnalyticalRIS**: supplementary references.
- **NoRIS, FixedRIS, RandomRIS, Equal-Power**: ablations.
- MADDPG, CTDE and TD3-Matched are intentionally excluded from the main pipeline.

All methods share one SISO STAR-RIS energy-splitting environment, one RSMA rate calculator and one locked train/validation/test ScenarioBank protocol.

## Install and test

```bash
python -m pip install -e .[dev]
pytest -q
```

## 1. Create locked ScenarioBanks

Run this once for each value of `N`:

```bash
python scripts/create_scenario_banks.py \
  --config configs/siso_n32.yaml \
  --output-dir artifacts/scenario_banks \
  --train-count 10000 \
  --validation-count 1000 \
  --test-count 1000
```

Available scalability configs:

```text
configs/siso_n16.yaml
configs/siso_n32.yaml
configs/siso_n64.yaml
configs/siso_n96.yaml
configs/siso_n128.yaml
```

## 2. Train and select checkpoints using validation only

```bash
python scripts/run_train.py \
  --method td3 \
  --config configs/siso_n32.yaml \
  --seed 0 \
  --output results/train/td3/N32/seed_0
```

The output contains:

- `best.pt`: checkpoint selected only on the validation bank;
- `latest.pt`: final training checkpoint;
- `training.csv`;
- `validation_raw.csv` with one row per validation scenario and evaluation step;
- `manifest.json` with config hash, Git commit and ScenarioBank checksums.

Use the same command with `--method ddpg` or `--method ppo`.

## 3. Deterministic test evaluation

```bash
python scripts/run_evaluate.py \
  --method td3 \
  --config configs/siso_n32.yaml \
  --checkpoint results/train/td3/N32/seed_0/best.pt \
  --bank artifacts/scenario_banks/N32_test.npz \
  --seed 0 \
  --output results/test/td3/N32/seed_0.csv
```

Evaluation is exploration-free and writes raw per-scenario CSV.

## 4. Conventional methods

```bash
python scripts/run_solver.py --method ao_sca --config configs/siso_n32.yaml \
  --bank artifacts/scenario_banks/N32_test.npz --start 0 --count 100 \
  --output results/solvers/N32/ao_sca_0_100.csv

python scripts/run_solver.py --method ao_grid --config configs/siso_n32.yaml \
  --bank artifacts/scenario_banks/N32_test.npz --start 0 --count 100 \
  --output results/solvers/N32/ao_grid_0_100.csv

python scripts/run_solver.py --method analytical_ris --config configs/siso_n32.yaml \
  --bank artifacts/scenario_banks/N32_test.npz --start 0 --count 100 \
  --output results/solvers/N32/analytical_ris_0_100.csv
```

AO-SCA is a local proximal first-order solver, never a global optimum or upper bound. AO-Grid is a deterministic coordinate codebook search, not random perturbation. AnalyticalRIS means analytical phase alignment with equal power/common allocation. Mathematical details are in [`docs/METHOD_IMPLEMENTATION.md`](docs/METHOD_IMPLEMENTATION.md).

## 5. Ablations

```bash
python scripts/run_ablation.py \
  --method td3 \
  --config configs/siso_n32.yaml \
  --checkpoint results/train/td3/N32/seed_0/best.pt \
  --bank artifacts/scenario_banks/N32_test.npz \
  --seed 0 \
  --output results/ablations/N32/seed_0.csv
```

Definitions:

- `no_ris`: remove the complete indirect STAR-RIS path;
- `fixed_ris`: beta = 0.5 and zero transmit/reflect phases;
- `random_ris`: sample once per scenario using a reproducible scenario seed;
- `equal_power`: override all stream powers equally while retaining the learned RIS/common allocation.

## 6. CPU single-thread latency

```bash
python scripts/benchmark_latency.py \
  --method td3 \
  --config configs/siso_n32.yaml \
  --checkpoint results/train/td3/N32/seed_0/best.pt \
  --bank artifacts/scenario_banks/N32_test.npz \
  --warmup 20 --count 500 \
  --output results/latency/td3_N32_seed0.csv
```

The script forces one Torch/OMP/MKL CPU thread and records inference, metric-evaluation and end-to-end solve latency.

## 7. Merge, statistics and plots

```bash
python scripts/merge_results.py --inputs results/test/**/*.csv results/solvers/**/*.csv \
  --output results/merged/N32_all.csv

python scripts/analyze_results.py --inputs results/merged/N32_all.csv \
  --output-dir results/statistics/N32

python scripts/plot_results.py --inputs results/merged/N32_all.csv \
  --output results/figures/N32_sum_rate.png
```

The analysis produces mean, standard deviation, 95% confidence intervals, paired t-tests, Wilcoxon tests, paired effect size and Holm-adjusted p-values. Repeated DRL seeds are averaged per method/scenario before pairing.

## Kaggle sharding

```bash
python scripts/make_kaggle_jobs.py \
  --config configs/siso_n32.yaml \
  --seeds 0 1 2 3 4 5 6 7 \
  --scenario-count 1000 \
  --scenario-shard-size 100 \
  --output kaggle_jobs_N32.json
```

Each Kaggle GPU session should run one learned seed. Conventional methods can run in CPU sessions and are sharded by non-overlapping scenario ranges.

Read [`CODEX_EXECUTION_PLAN.md`](CODEX_EXECUTION_PLAN.md) before running the full experiment or editing the thesis.
