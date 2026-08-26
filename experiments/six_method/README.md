# Six-method STAR-RIS–RSMA benchmark v1

Scientific trigger commit: `99318fefa53bef91fa5f105ec71ddae73fc96c39`.

## Methods

- Learned: TD3, DDPG, PPO.
- Traditional: AO-SCA, AO-Grid, AnalyticalRIS.

## Locked protocol

- `N = 16, 32, 64, 96, 128`.
- DRL seeds `0..7`.
- `100,000` environment interactions per DRL method/seed/N.
- `1,000` validation scenarios and `1,000` locked test scenarios per N.
- Identical ScenarioBank checksums across all six methods.
- QoS-dual reward shaping and feasibility-first checkpoint selection for all three DRL methods.
- Deterministic test actions.
- CPU latency measured single-threaded on one collector runner.
- Thesis/frozen reproduction uses only `configs/v3/constrained_action_n*.yaml`; legacy `configs/siso_n*.yaml` files are not part of this benchmark.

## Configuration source of truth

For each STAR-RIS size, use the corresponding constrained v3 configuration:

```text
configs/v3/constrained_action_n16.yaml
configs/v3/constrained_action_n32.yaml
configs/v3/constrained_action_n64.yaml
configs/v3/constrained_action_n96.yaml
configs/v3/constrained_action_n128.yaml
```

These configurations define the constrained physical action parameterization, locked ScenarioBank paths, QoS-dual shaping, feasibility-first validation targets, TD3 stability controls and the `0.12 -> 0.03` exploration schedule used by the six-method result bundle.

## Frozen latency protocol

The published/frozen table `results/six_method_v1/tables/TABLE_SIX_METHOD_CPU_LATENCY.csv` contains **exactly 100 single-thread CPU latency samples for every method/N pair**. Those 100-sample measurements are the values reported by the thesis.

The publication collector enforces a minimum of 100 samples so that a future rerun may contain more measurements, but this lower-bound gate must not be confused with the frozen sample count. In particular, a command using `--count 500` is an optional extended latency experiment, not a reproduction of the frozen `six_method_v1` table. Results from a 500-sample rerun must be stored and reported as a new experiment rather than mixed with the frozen 100-sample measurements.

## Split Kaggle stages

1. `td3_low_n`, `td3_high_n`.
2. `ddpg_low_n`, `ddpg_high_n`.
3. `ppo_low_n`, `ppo_high_n`.

## Publication gate

The collector refuses to publish unless all of the following hold:

- all six stage manifests exist and report the same scientific commit;
- every DRL method has exactly `5 × 8 × 1,000 = 40,000` test rows;
- every traditional baseline has exactly `5 × 1,000 = 5,000` matched test rows;
- no duplicate method/N/seed/scenario keys;
- all core metrics are finite;
- ScenarioBank checksums match at each N;
- at least 100 single-thread CPU latency samples exist per method/N;
- the generated `SIX_METHOD_AUDIT.json` verdict is `PASS`.

For the frozen published bundle, the latency coverage satisfying this gate is `100` samples for every method/N pair.

Only after this gate passes may `results/six_method_v1/` be used to update the thesis.
