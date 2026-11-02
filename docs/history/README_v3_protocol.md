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

> **Protocol note.** Reproduction of the audited six-method benchmark must use `configs/v3/constrained_action_n*.yaml`. The older `configs/siso_n*.yaml` files are historical/backward-compatible configurations and are not the source of `results/six_method_v1/`.

## Install and test

```bash
python -m pip install -e .[dev]
pytest -q
```

## 1. Create locked ScenarioBanks

Run this once for each value of `N`:

```bash
python scripts/create_scenario_banks.py \
  --config configs/v3/constrained_action_n32.yaml \
  --output-dir artifacts/scenario_banks \
  --train-count 10000 \
  --validation-count 1000 \
  --test-count 1000
```

Available constrained v3 scalability configs:

```text
configs/v3/constrained_action_n16.yaml
configs/v3/constrained_action_n32.yaml
configs/v3/constrained_action_n64.yaml
configs/v3/constrained_action_n96.yaml
configs/v3/constrained_action_n128.yaml
```

## 2. Train and select checkpoints using validation only

For the six-method benchmark, use the v3 training entrypoint:

```bash
python scripts/run_train_drl_v3.py \
  --method td3 \
  --config configs/v3/constrained_action_n32.yaml \
  --seed 0 \
  --output results/train/td3/N32/seed_0
```

The historical `scripts/run_train.py` entrypoint is retained for backward compatibility and must not be used to reproduce the frozen six-method benchmark.

The v3 output contains the checkpoint selected using the locked validation bank, training/validation logs and a manifest containing the effective configuration and ScenarioBank identifiers. Use the same command with `--method ddpg` or `--method ppo`.

## 3. Deterministic test evaluation

```bash
python scripts/run_evaluate.py \
  --method td3 \
  --config configs/v3/constrained_action_n32.yaml \
  --checkpoint results/train/td3/N32/seed_0/best.pt \
  --bank artifacts/scenario_banks/N32_test.npz \
  --seed 0 \
  --output results/test/td3/N32/seed_0.csv
```

Evaluation is exploration-free and writes raw per-scenario CSV.

## 4. Conventional methods

```bash
python scripts/run_solver.py --method ao_sca --config configs/v3/constrained_action_n32.yaml \
  --bank artifacts/scenario_banks/N32_test.npz --start 0 --count 100 \
  --output results/solvers/N32/ao_sca_0_100.csv

python scripts/run_solver.py --method ao_grid --config configs/v3/constrained_action_n32.yaml \
  --bank artifacts/scenario_banks/N32_test.npz --start 0 --count 100 \
  --output results/solvers/N32/ao_grid_0_100.csv

python scripts/run_solver.py --method analytical_ris --config configs/v3/constrained_action_n32.yaml \
  --bank artifacts/scenario_banks/N32_test.npz --start 0 --count 100 \
  --output results/solvers/N32/analytical_ris_0_100.csv
```

AO-SCA is a local proximal first-order solver, never a global optimum or upper bound. AO-Grid is a deterministic coordinate codebook search, not random perturbation. AnalyticalRIS is a phase-alignment heuristic with equal power/common allocation, not an analytical optimum of the full problem. Mathematical details are in [`docs/METHOD_IMPLEMENTATION.md`](docs/METHOD_IMPLEMENTATION.md).

## 5. Ablations

```bash
python scripts/run_ablation.py \
  --method td3 \
  --config configs/v3/constrained_action_n32.yaml \
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

## 6. CPU single-thread decision latency

The frozen result used in the thesis contains **exactly 100 latency samples for every method/N pair**. Reproduce that protocol with `--count 100`:

```bash
python scripts/benchmark_latency.py \
  --method td3 \
  --config configs/v3/constrained_action_n32.yaml \
  --checkpoint results/train/td3/N32/seed_0/best.pt \
  --bank artifacts/scenario_banks/N32_test.npz \
  --warmup 20 --count 100 \
  --output results/latency/td3_N32_seed0.csv
```

The benchmark forces one Torch/OMP/MKL CPU thread. The reported quantity is algorithmic decision/inference latency in the stated measurement path, not full end-to-end radio-system latency.

A larger run such as `--count 500` is allowed only as an **additional latency experiment**. It may reduce sampling uncertainty, but it is not the frozen `six_method_v1` protocol and its output must not be mixed with the published 100-sample table. The source of truth for the thesis is `results/six_method_v1/tables/TABLE_SIX_METHOD_CPU_LATENCY.csv`, where every method/N row has `count=100`.

## 7. Six-method benchmark and statistics

The full staged reproduction entrypoint is `scripts/run_drl_stage.py`, which uses the constrained v3 configurations and exactly eight DRL seeds. See [`experiments/six_method/README.md`](experiments/six_method/README.md) for the locked publication protocol.

The published paired analysis averages the eight DRL seeds within each test scenario before pairing, so the paired sample size is 1,000 scenarios rather than 8,000 flattened rows. Statistical claims must be interpreted together with the finite-seed and hyperparameter-tuning limitations documented with the result bundle.

## Kaggle sharding

```bash
python scripts/make_kaggle_jobs.py \
  --config configs/v3/constrained_action_n32.yaml \
  --seeds 0 1 2 3 4 5 6 7 \
  --scenario-count 1000 \
  --scenario-shard-size 100 \
  --output kaggle_jobs_N32.json
```

Each Kaggle GPU session should run one learned seed. Conventional methods can run in CPU sessions and are sharded by non-overlapping scenario ranges.
