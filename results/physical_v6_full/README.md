# Physical V6 full-run preparation

This directory contains the tracked audit records for the independent
`physical_v6_soft_anchor` experiment. Training outputs have not been produced
yet.

## Fixed protocol

- RIS sizes: `N = 16, 32, 64, 96, 128`
- TD3 interaction budget: `100,000` per seed
- ScenarioBank splits: train `10,000`/seed `11001`, validation
  `1,000`/seed `22001`, test `1,000`/seed `33001`
- Action parameterization: `physical_v6_soft_anchor`

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
