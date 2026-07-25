# Kaggle Run Archive

This directory is the canonical repository archive of the six final Kaggle runs.

Each run contains:

- `source/`: the notebook source and Kaggle kernel metadata pulled from the saved version;
- `output/`: the files saved by the completed Kaggle run;
- SHA-256, size, source-kernel, and retrieval metadata in `KAGGLE_RUN_MANIFEST.json`.

The authoritative implementation is under `star_ris_rsma/`, `scripts/`, and `configs/`.
The authoritative audited paper/thesis evidence is under `results/`.
Exact notebook-06 duplicates are represented by `.canonical.json` pointers to avoid
committing duplicate binary data. Files exceeding GitHub's per-file limit are represented
by `.external.json` records containing size, SHA-256, kernel reference, and retrieval command.

## Run order

1. `01_td3_low_n`
2. `02_td3_high_n`
3. `03_ao_grid`
4. `04_ao_sca`
5. `05_analytical_ris`
6. `06_final_academic_bundle`
