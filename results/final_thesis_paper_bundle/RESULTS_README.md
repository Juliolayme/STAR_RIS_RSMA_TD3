# Final thesis and paper result bundle

- Scientific algorithm commit: `89c39da461523a7f5911a302cb9415aeaa5824ce`
- Report-generation commit: `b67adee55def9dbc32c8acded09dad68c51ee49a`
- TD3: 5 N values × 8 seeds × 1,000 locked test scenarios.
- AO-SCA, AO-Grid, AnalyticalRIS: 1,000 matched scenarios per N.
- CPU latency: all methods timed sequentially in the same process per N.

## Main tables

- `tables/TABLE_FINAL_PERFORMANCE.*`
- `tables/TABLE_TD3_8SEED_CI95.*`
- `tables/TABLE_PAIRED_WILCOXON_HOLM.*`
- `tables/TABLE_CPU_LATENCY.*`
- `tables/TABLE_BASELINE_SOLVER_TIME_DESCRIPTIVE.*`

## Paper-ready figures

- `figures/fig01_training_sum_rate.png`
- `figures/fig02_training_qos_fraction.png`
- `figures/fig03_training_violation.png`
- `figures/fig04_qos_dual.png`
- `figures/fig05_validation_sum_rate.png`
- `figures/fig06_final_sum_rate.png`
- `figures/fig07_final_qos_fraction.png`
- `figures/fig08_final_all_qos.png`
- `figures/fig09_final_violation.png`
- `figures/fig10_cpu_latency.png`
- `figures/fig11_td3_speedup.png`
- `figures/fig12_quality_latency_tradeoff.png`

## Interpretation guardrails

- AO-SCA is a local iterative baseline, not a global optimum or upper bound.
- Do not call TD3 the fastest absolute method when AnalyticalRIS is faster.
- Interpret latency jointly with sum-rate and QoS quality.
