# Six-method STAR-RIS–RSMA benchmark

## Reviewer verdict

**PASS for thesis integration**, subject to the interpretation guardrails below. The result package is complete, reproducible, and internally consistent for comparing TD3, DDPG, PPO, AO-SCA, AO-Grid, and AnalyticalRIS under the locked protocol.

## Validated protocol

- Learned methods: TD3, DDPG, PPO.
- Traditional methods: AO-SCA, AO-Grid, AnalyticalRIS.
- STAR-RIS sizes: `N = 16, 32, 64, 96, 128`.
- DRL: eight independent seeds and 1,000 locked test scenarios per seed and N.
- Traditional baselines: the same 1,000 locked test scenarios per N.
- All DRL stages share repository commit `99318fefa53bef91fa5f105ec71ddae73fc96c39` and the QoS-feasibility-first checkpoint protocol.
- CPU latency is single-threaded and measured on the same runner for all six methods.
- The audit reports shared ScenarioBank checksums, finite core metrics, complete stage coverage, and verdict `PASS`.

## Main quantitative finding: quality–latency trade-off

AO-SCA achieves the highest mean sum-rate at every N, while TD3 provides the strongest learned quality–latency trade-off.

| N | AO-SCA sum-rate | TD3 sum-rate | TD3 / AO-SCA | TD3 QoS fraction | TD3 all-user QoS | TD3 latency (ms) | AO-SCA latency (ms) | AO-SCA / TD3 latency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 15.2137 | 10.7661 | 70.8% | 0.9893 | 0.9798 | 0.2460 | 216.4406 | 880× |
| 32 | 16.4524 | 10.7160 | 65.1% | 0.9962 | 0.9944 | 0.2575 | 370.2407 | 1,438× |
| 64 | 17.8059 | 11.7024 | 65.7% | 0.9970 | 0.9944 | 0.2848 | 603.9618 | 2,121× |
| 96 | 18.8800 | 12.0636 | 63.9% | 0.9994 | 0.9985 | 0.3112 | 890.1315 | 2,860× |
| 128 | 19.6363 | 12.4117 | 63.2% | 0.9991 | 0.9984 | 0.3420 | 1,039.0379 | 3,038× |

The defensible central conclusion is therefore:

> AO-SCA maximizes sum-rate in the evaluated setting, whereas TD3 preserves near-unity QoS reliability with sub-millisecond inference and a latency advantage of approximately three orders of magnitude.

AO-SCA is a local iterative baseline, not a global optimum or theoretical upper bound.

## Comparison among TD3, DDPG, and PPO

### TD3

TD3 is the most robust DRL method across all evaluated STAR-RIS sizes. Its mean sum-rate ranges from 10.7160 to 12.4117, mean QoS fraction from 0.9893 to 0.9994, and all-user QoS probability from 0.9798 to 0.9985. Seed-level sum-rate standard deviation remains comparatively limited, from 0.1652 to 0.9435.

### DDPG

DDPG is competitive only in the moderate case `N = 32`: sum-rate 10.2295 versus 10.7160 for TD3, slightly better QoS fraction and all-user QoS, and slightly lower CPU latency. It is therefore incorrect to claim that TD3 strictly dominates DDPG on every metric at `N = 32`.

DDPG becomes unstable as N grows. At `N = 96`, its all-user QoS probability falls to 0.1376 and mean violation rises to 0.3972. At `N = 128`, all-user QoS is 0.0040, mean violation is 0.5512, and seed-level sum-rate standard deviation is 5.0875. The convergence records also show prolonged infeasible validation states for high-N DDPG.

### PPO

PPO is comparatively stable but underfits the constrained control problem. Across N, its mean sum-rate remains approximately 2.10–2.16, QoS fraction approximately 0.61–0.66, and all-user QoS approximately 0.04–0.09. High-N validation checkpoints remain outside the required feasibility region. PPO should be reported as a negative baseline, not as a competitive solution.

## Traditional baselines

- **AO-Grid** is conservative: it obtains all-user QoS probability 1.0 and zero reported violation for all N, but sum-rate stays near 3.982. Its mean latency increases from 70.6 ms to 531.3 ms.
- **AnalyticalRIS** is the fastest raw decision rule at approximately 0.122–0.150 ms, but it has QoS fraction 0 and sum-rate near 1.982. It is not a feasible QoS-aware optimizer and must not be presented as the best practical method.
- **AO-SCA** has the strongest quality result and near-perfect QoS, but its iterative latency increases from 216.4 ms to 1,039.0 ms.

## Statistical interpretation

The paired tests use the same 1,000 locked scenarios and average the eight DRL seeds per scenario before comparison. This avoids treating seeds as duplicated scenario observations. TD3 exceeds DDPG in mean sum-rate for every N after Holm correction; the smallest difference occurs at `N = 32` with mean difference 0.4865 and Cohen's `dz = 0.5200`.

However, these are scenario-paired tests on seed-averaged policies. They should not be described as a complete hierarchical test over the population of possible training seeds. Report effect sizes and practical differences together with corrected p-values.

## Thesis-ready claims

Allowed:

1. AO-SCA provides the highest sum-rate under the evaluated protocol.
2. TD3 is the best-performing and most scalable DRL method among TD3, DDPG, and PPO.
3. TD3 maintains near-unity QoS reliability while requiring approximately 0.25–0.34 ms per decision.
4. TD3 offers a strong quality–latency trade-off, retaining about 63–71% of AO-SCA sum-rate while reducing decision latency by roughly 880–3,038 times.
5. DDPG exhibits strong seed sensitivity and loses QoS robustness at high N.
6. PPO does not reach the QoS-feasible region under the fixed 100,000-interaction budget.

Forbidden or misleading:

1. TD3 achieves the globally optimal or highest sum-rate.
2. TD3 strictly dominates DDPG on every metric for every N.
3. AnalyticalRIS is the best method merely because it has the lowest latency.
4. AO-SCA is a theoretical upper bound.
5. The paired p-values alone prove superiority over every possible training seed or deployment distribution.

## Recommended thesis figures

Use the following in the main experimental chapter:

- `fig01_six_method_sum_rate` for quality scaling.
- `fig02_six_method_qos_fraction` and `fig03_six_method_all_qos` for reliability.
- `fig05_six_method_cpu_latency` for computational cost.
- `fig06_six_method_quality_latency` as the central trade-off figure.

Use the convergence figures for TD3, DDPG, and PPO to explain stability and failure modes. Put the full Holm-corrected paired-test table and raw convergence plots in an appendix if space is limited.
