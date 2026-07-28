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

Only after this gate passes may `results/six_method_v1/` be used to update the thesis.
