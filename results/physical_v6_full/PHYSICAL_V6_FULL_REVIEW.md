# Physical V6 full-training review

## Verdict

TD3 V6 is strong and stable for N >= 32, but the frozen feasibility-first checkpoint rule is overly conservative at N=16. The corrected AO methods remain stronger in absolute sum-rate; TD3 is therefore a competitive learned method, not a new optimum or upper bound.

## Mean sum-rate on 1,000 matched test scenarios

| N | TD3 | DDPG | PPO | AO-SCA corrected | AO-Grid corrected |
|---:|---:|---:|---:|---:|---:|
| 16 | 10.2881 | 12.2811 | 6.4783 | 14.3449 | 16.5802 |
| 32 | 16.1728 | 4.5316 | 5.7864 | 17.7256 | 18.5044 |
| 64 | 18.7154 | 7.0851 | 5.7317 | 19.9050 | 20.4401 |
| 96 | 19.9279 | 14.5978 | 5.7299 | 20.9103 | 21.5862 |
| 128 | 20.9018 | 8.2911 | 5.4616 | 21.6095 | 22.3918 |

## Review findings

- TD3 selected a trained checkpoint in 23/25 jobs; 2/25 selected initialization (both at N=16).
- DDPG selected initialization in 12/25 jobs and is unstable across N; it should remain a comparator, not support the main claim.
- PPO learned in all 25 jobs, but its final quality remains far below TD3.
- At N=128, TD3 reaches 20.9018, trailing corrected AO-SCA by 0.7076 and corrected AO-Grid by 1.4900 bit/s/Hz.
- At N=16, two TD3 seeds select step 0 because later policies lose the strict all-users-QoS gate; this lowers the five-seed mean to 10.2881. Do not hide this in slides.
- All paired tests average the five DRL seeds within each locked scenario before testing. Holm is applied separately within each N over all 15 method pairs.
- Error-bar widths are not compared across DRL and deterministic baselines: DRL uncertainty uses five seed means, while baseline uncertainty uses 1,000 scenarios.
- Recorded training consumed 11.71 aggregate GPU-hours on Tesla T4; this is summed job time, not orchestration wall-clock time.

## TD3 versus corrected AO

| N | TD3 - AO-SCA | TD3 - AO-Grid | TD3 win fraction vs AO-SCA |
|---:|---:|---:|---:|
| 16 | -4.0568 | -6.2921 | 0.130 |
| 32 | -1.5527 | -2.3316 | 0.026 |
| 64 | -1.1895 | -1.7246 | 0.012 |
| 96 | -0.9824 | -1.6583 | 0.015 |
| 128 | -0.7076 | -1.4900 | 0.052 |

## Reporting recommendation

Use TD3 V6 as the proposed method and report the full five-seed mean. State explicitly that corrected AO gives higher offline sum-rate, while TD3 provides a learned one-pass policy. Do not call corrected AO a global optimum, and do not replace the N=16 mean with only the three successful seeds.

The embedded `git_commit` column in the training CSVs was populated by Kaggle's runtime identifier. This report preserves it as `embedded_runtime_identifier` and uses the independently written `KAGGLE_JOB_PROVENANCE.json` commit as `repository_commit`.
Two TD3 N=16 archives (seeds 2 and 3) came from commit `0e47904`; the other 23 came from `d5c25da`. The diff between those commits changes only CI/orchestration files, not source, configs, or the training entry point.
