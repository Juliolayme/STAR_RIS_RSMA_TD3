# Six-method STAR-RIS–RSMA benchmark

## Validated protocol

- Learned methods: TD3, DDPG, PPO.
- Traditional methods: AO-SCA, AO-Grid, AnalyticalRIS.
- N = 16, 32, 64, 96, 128.
- DRL: eight seeds and 1,000 locked test scenarios per seed/N.
- Traditional baselines: the same 1,000 locked scenarios per N.
- All DRL stages share one repository commit and the QoS-feasibility-first protocol.
- CPU latency is single-threaded and measured on the same runner for all methods.

## Best observed sum-rate by N

|   n_ris | method   |   sum_rate_mean |
|--------:|:---------|----------------:|
|      16 | ao_sca   |         15.2137 |
|      32 | ao_sca   |         16.4524 |
|      64 | ao_sca   |         17.8059 |
|      96 | ao_sca   |         18.88   |
|     128 | ao_sca   |         19.6363 |

## Fastest observed decision method by N

|   n_ris | method         |   solve_ms_median |
|--------:|:---------------|------------------:|
|      16 | analytical_ris |          0.120056 |
|      32 | analytical_ris |          0.122826 |
|      64 | analytical_ris |          0.133386 |
|      96 | analytical_ris |          0.137553 |
|     128 | analytical_ris |          0.147918 |

## Interpretation guardrails

- AO-SCA remains a local iterative baseline, not a global optimum or upper bound.
- Do not claim one DRL algorithm dominates unless quality, QoS, latency and corrected tests agree.
- Exact numerical claims must be copied from TABLE_SIX_METHOD_PERFORMANCE.csv.
- The paired tests average the eight DRL seeds per locked scenario before comparison, avoiding pseudo-replication.

DRL repository commit: `99318fefa53bef91fa5f105ec71ddae73fc96c39`.
