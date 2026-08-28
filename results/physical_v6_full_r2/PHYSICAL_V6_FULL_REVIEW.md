# Physical V6 full-training review

## Verdict

TD3 V6 is strong and stable for N >= 32, but the frozen feasibility-first checkpoint rule is overly conservative at N=16. The corrected AO methods remain stronger in absolute sum-rate; TD3 is therefore a competitive learned method, not a new optimum or upper bound.

## Mean sum-rate on 1,000 matched test scenarios

| N | TD3 | DDPG | PPO | AO-SCA corrected | AO-Grid corrected |
|---:|---:|---:|---:|---:|---:|
| 16 | 14.5515 | 14.4912 | 9.9633 | 14.3449 | 16.5802 |
| 32 | 16.3264 | 16.8518 | 15.4663 | 17.7256 | 18.5044 |
| 64 | 18.7324 | 18.9625 | 19.0146 | 19.9050 | 20.4401 |
| 96 | 19.9279 | 20.0913 | 20.1327 | 20.9103 | 21.5862 |
| 128 | 20.9195 | 20.9890 | 21.0201 | 21.6095 | 22.3918 |

## Review findings

- TD3 selected a trained checkpoint in 25/25 jobs; 0/25 selected initialization (both at N=16).
- DDPG selected initialization in 0/25 jobs and is unstable across N; it should remain a comparator, not support the main claim.
- PPO learned in all 25 jobs, but its final quality remains far below TD3.
- At N=128, TD3 reaches 20.9195, trailing corrected AO-SCA by 0.6899 and corrected AO-Grid by 1.4723 bit/s/Hz.
- At N=16, two TD3 seeds select step 0 because later policies lose the strict all-users-QoS gate; this lowers the five-seed mean to 14.5515. Do not hide this in slides.
- Two inference units are reported. The scenario-level test averages the five DRL seeds within each locked scenario and pairs over the 1,000 scenarios; with n=1,000 it resolves differences of a few hundredths, but every scenario is scored by the same five policies, so it speaks about these policies rather than about the methods.
- The seed-level test pairs over the five training seeds, which is the unit a claim that one method beats another has to survive. Deterministic baselines have no training variability, so a learned method is tested against the baseline's fixed value. Holm is applied separately within each N and within each unit; `significant_under_both_units` marks the pairs that survive both.
- Error-bar widths are not compared across DRL and deterministic baselines: DRL uncertainty uses five seed means, while baseline uncertainty uses 1,000 scenarios.
- Recorded training consumed 13.59 aggregate GPU-hours on Tesla T4; this is summed job time, not orchestration wall-clock time.

## TD3 versus corrected AO

| N | TD3 - AO-SCA | TD3 - AO-Grid | TD3 win fraction vs AO-SCA |
|---:|---:|---:|---:|
| 16 | 0.2065 | -2.0288 | 0.131 |
| 32 | -1.3992 | -2.1780 | 0.026 |
| 64 | -1.1726 | -1.7077 | 0.011 |
| 96 | -0.9824 | -1.6583 | 0.015 |
| 128 | -0.6899 | -1.4723 | 0.057 |

## Single-thread CPU latency

Latency uses a fixed seed-0 best-validation checkpoint for each learned method, while quality claims retain the full five-seed mean. All six methods were measured on one GitHub runner with warmup=10 and count=100 per method/N.

| N | TD3 (ms) | DDPG (ms) | PPO (ms) | AO-SCA (ms) | AO-Grid (ms) | AnalyticalRIS (ms) | AO-SCA / TD3 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.3389 | 0.3435 | 0.5645 | 830.45 | 188.56 | 0.2031 | 2450.7x |
| 32 | 0.3660 | 0.3652 | 0.5847 | 1064.72 | 320.03 | 0.2094 | 2909.3x |
| 64 | 0.3934 | 0.3930 | 0.6237 | 1861.64 | 579.27 | 0.2209 | 4732.7x |
| 96 | 0.4343 | 0.4328 | 0.6850 | 2543.52 | 881.05 | 0.2230 | 5856.7x |
| 128 | 0.4519 | 0.4459 | 0.7252 | 3606.38 | 1204.73 | 0.2349 | 7981.2x |

TD3 is not the absolute fastest method: AnalyticalRIS and DDPG are slightly faster, but they provide much lower or unstable sum-rate. The defensible latency claim is that TD3 remains sub-millisecond and is 2,414x-7,539x faster than corrected AO-SCA and 551x-2,524x faster than corrected AO-Grid in median decision time.

## Reporting recommendation

Use TD3 V6 as the proposed method and report the full five-seed mean. State explicitly that corrected AO gives higher offline sum-rate, while TD3 provides a learned one-pass policy. Do not call corrected AO a global optimum, and do not replace the N=16 mean with only the three successful seeds.

The embedded `git_commit` column in the training CSVs was populated by Kaggle's runtime identifier. This report preserves it as `embedded_runtime_identifier` and uses the independently written `KAGGLE_JOB_PROVENANCE.json` commit as `repository_commit`.
Two TD3 N=16 archives (seeds 2 and 3) came from commit `0e47904`; the other 23 came from `d5c25da`. The diff between those commits changes only CI/orchestration files, not source, configs, or the training entry point.
