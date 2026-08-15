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
- Reproduction uses only `configs/v3/constrained_action_n*.yaml`; legacy `configs/siso_n*.yaml` files are not part of this benchmark.

## Configuration source of truth

Use the corresponding constrained v3 configuration for each STAR-RIS size:

```text
configs/v3/constrained_action_n16.yaml
configs/v3/constrained_action_n32.yaml
configs/v3/constrained_action_n64.yaml
configs/v3/constrained_action_n96.yaml
configs/v3/constrained_action_n128.yaml
```

These files contain the constrained physical action parameterization, locked ScenarioBank paths, QoS-dual shaping, feasibility-first validation targets, TD3 stability controls and the exploration schedule used by the published six-method bundle.

## Frozen latency protocol

The frozen table `results/six_method_v1/tables/TABLE_SIX_METHOD_CPU_LATENCY.csv` contains **exactly 100 single-thread CPU latency samples for every method/N pair**. These 100-sample measurements are the values reported in the thesis.

The publication collector requires **at least** 100 samples so future reruns may contain more observations. That lower-bound validation rule does not redefine the frozen sample count. A command with `--count 500` therefore represents an optional extended latency experiment, not a reproduction of `six_method_v1`; its output must be stored and reported separately rather than mixed with the frozen 100-sample measurements.

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

For the frozen published bundle, every method/N latency row satisfying this gate has `count=100`.

Only after this gate passes may `results/six_method_v1/` be used to update the thesis.
