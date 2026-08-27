# Physical V6 full-training evidence

This directory contains the tracked audit records and aggregated results for
the independent `physical_v6_soft_anchor` experiment.

## Fixed protocol

- RIS sizes: `N = 16, 32, 64, 96, 128`
- TD3 interaction budget: `100,000` per seed
- ScenarioBank splits: train `10,000`/seed `11001`, validation
  `1,000`/seed `22001`, test `1,000`/seed `33001`
- Action parameterization: `physical_v6_soft_anchor`
- Methods trained: TD3, DDPG, PPO
- Coverage: 5 N x 5 seeds x 3 methods = 75 GPU jobs

The 15 local `.npz` files live under `artifacts/scenario_banks/` and total
approximately 313 MB, so they are intentionally excluded from Git. They were
recovered from the still-available canonical six-method GitHub artifact, not
regenerated on Windows. The source artifact, every split checksum, and the five
frozen test-checksum comparisons are recorded in
`SCENARIO_BANK_MANIFEST.json`.

Verify the local copies before training:

```bash
python scripts/prepare_v6_scenario_banks.py --verify-existing
```

The command fails if a count, seed, disjointness check, or frozen test checksum
does not match.

## Timing evidence

Each training job now records:

- `training.csv`: elapsed seconds and interactions/second during learning;
- `validation_summary.csv`: wall time for every validation pass;
- `manifest.json`: total training-loop time, throughput, device/runtime
  versions, and peak allocated GPU memory;
- `timing.json`: UTC start/finish, training call time, all three checkpoint test
  times, and total job wall time.

These fields are intended for the thesis runtime table and the review bundle.

## Published full-run outputs

- `PHYSICAL_V6_FULL_AUDIT.json`: machine-readable coverage, checksum,
  provenance, and statistical-protocol audit;
- `PHYSICAL_V6_FULL_REVIEW.md`: scientific interpretation and reporting
  guardrails;
- `raw/DRL_V6_TEST_BEST_RAW_ALL.csv`: 75,000 best-checkpoint test rows;
- `tables/TABLE_V6_CHECKPOINT_AUDIT.csv`: initial/best/latest metrics and
  learning gain for every job;
- `tables/TABLE_V6_SIX_METHOD_PERFORMANCE.csv`: V6 DRL plus frozen corrected
  AO-SCA, AO-Grid, and AnalyticalRIS;
- `tables/TABLE_V6_SIX_METHOD_PAIRED_TESTS_HOLM.csv`: matched-scenario paired
  tests with Holm correction over all 15 pairs separately within each N;
- `tables/TABLE_V6_TRAINING_TIME.csv`: measured Tesla T4 training time and
  throughput;
- `tables/TABLE_V6_SIX_METHOD_CPU_LATENCY.csv`: single-thread mean, median,
  P95, and P99 decision latency for all six methods;
- `tables/TABLE_V6_TD3_LATENCY_SPEEDUP.csv`: TD3 median-latency ratios against
  corrected AO-SCA, corrected AO-Grid, and AnalyticalRIS;
- `raw/CPU_LATENCY_V6_RAW_ALL.csv`: 3,000 timing samples from one GitHub CPU
  runner (`warmup=10`, `count=100` per method/N).

The source GitHub Actions runs are
[TD3 33053693666](https://github.com/Juliolayme/STAR_RIS_RSMA_TD3/actions/runs/33053693666)
and
[DDPG/PPO 33061462093](https://github.com/Juliolayme/STAR_RIS_RSMA_TD3/actions/runs/33061462093).
The latency run is
[33076374485](https://github.com/Juliolayme/STAR_RIS_RSMA_TD3/actions/runs/33076374485).
The corresponding artifact IDs are recorded in the audit JSON.

Regenerate the tracked report from downloaded artifacts:

```bash
python scripts/build_physical_v6_full_report.py
```

The report builder verifies all 75 archive SHA-256 hashes, all raw/summary
metric matches, 1,000-scenario coverage, and the frozen ScenarioBank checksum
at every N before publishing output.
