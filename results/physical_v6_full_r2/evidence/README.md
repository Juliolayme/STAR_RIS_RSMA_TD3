# Physical V6 r2 permanent evidence

The large binary evidence for `results/physical_v6_full_r2/` is stored in the
GitHub Release [`physical-v6-r2-evidence-9547c62`](https://github.com/Juliolayme/STAR_RIS_RSMA_TD3/releases/tag/physical-v6-r2-evidence-9547c62),
not in expiring GitHub Actions artifacts.

The release contains:

- all 15 canonical ScenarioBanks: five RIS sizes times train/validation/test;
- all 75 audited r2 training archives: TD3, DDPG and PPO, five RIS sizes and
  five seeds;
- every `initial.pt`, `best.pt`, `latest.pt`, retained candidate checkpoint,
  validation trace, test output, timing record and job provenance contained in
  those archives.

`EVIDENCE_MANIFEST.json` records the release asset SHA-256 values and the
SHA-256 of every nested training archive and ScenarioBank. `SHA256SUMS.txt`
contains the four top-level asset checksums.

## Download and verify

From the repository root:

```bash
python results/physical_v6_full_r2/evidence/download_evidence.py
```

Choose only one or more assets when the complete 2.1 GB evidence set is not
needed:

```bash
python results/physical_v6_full_r2/evidence/download_evidence.py \
  --asset scenario_banks --asset td3_training
```

The downloader streams each file and rejects it if its SHA-256 does not match
the tracked manifest. Pass `--extract` to extract only after verification.

The reviewed source snapshot is commit
`9547c6227f588a0b5c2e8ccd5ffd90c847f2234c`. The 75 GPU jobs were run from
training commit `3e96df18eab0a8b3a6a3f1006d74c31e09add2a1`; their independent per-job
provenance remains inside every nested archive.
