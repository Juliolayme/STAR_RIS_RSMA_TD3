# Six-method STAR-RIS–RSMA benchmark v2

## Frozen protocol

- Learned methods (unchanged canonical evidence): TD3, DDPG, PPO; 8 seeds × 1,000 locked test scenarios per N.
- Continuous traditional baseline: corrected pairwise AO, frozen at max_iter=80.
- Discrete traditional baseline: corrected AO-Grid with zero simplex level and forward/reverse RIS sweeps.
- AnalyticalRIS is unchanged.
- N = 16, 32, 64, 96, 128.
- All six methods use identical locked ScenarioBank checksums at each N.
- Holm correction is applied separately within each N over all 15 pairwise method comparisons.
- DRL seeds are averaged per locked scenario before paired tests to avoid pseudo-replication.
- CPU decision latency uses one thread, warmup=10, count=100, and all six methods are measured on the same GitHub Actions runner.

## Best observed sum-rate by N

|   n_ris | method   |   sum_rate_mean |
|--------:|:---------|----------------:|
|      16 | ao_grid  |         16.5802 |
|      32 | ao_grid  |         18.5044 |
|      64 | ao_grid  |         20.4401 |
|      96 | ao_grid  |         21.5862 |
|     128 | ao_grid  |         22.3918 |

## Fastest observed decision method by N

|   n_ris | method         |   solve_ms_median |
|--------:|:---------------|------------------:|
|      16 | analytical_ris |          0.121565 |
|      32 | analytical_ris |          0.126922 |
|      64 | analytical_ris |          0.13874  |
|      96 | analytical_ris |          0.156871 |
|     128 | analytical_ris |          0.154193 |

## Interpretation guardrails

- Corrected pairwise AO is a deterministic local continuous baseline, not a global optimum or an upper bound.
- Corrected AO-Grid is a restricted discrete heuristic, not a global optimizer.
- Exact slide values must be copied from the v2 tables generated in this artifact.

AO freeze: `corrected_pairwise_ao_v2:max_iter=80:pairwise_probe=1e-4:stationarity_tol=1e-6:post_ris_simplex_polish=40`.
AO-Grid freeze: `corrected_ao_grid_v1:rounds=2:zero_level:bidirectional_ris`.
