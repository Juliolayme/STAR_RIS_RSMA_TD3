# STAR-RIS–RSMA TD3 — Physical V6 experiment

This branch is the clean, reproducible workspace for evaluating the proposed
TD3 method with the `physical_v6_soft_anchor` action parameterization. DDPG,
PPO, corrected AO-SCA, corrected AO-Grid, and AnalyticalRIS are comparators;
TD3 remains the primary method.

Historical thesis builds, V1 results, archived Kaggle runs, and pilot outputs
are intentionally excluded from this branch. Frozen corrected baseline evidence
is retained under `results/six_method_v2/`.

## Install and test

```bash
python -m pip install -e ".[dev]"
pytest -q
```

The dependency contract supports the current Kaggle/PyTorch 2.13 images.

## Locked V6 protocol

- RIS sizes: `N = 16, 32, 64, 96, 128`
- TD3 budget: 100,000 environment interactions per seed
- Five independent TD3 seeds
- Train/validation/test ScenarioBanks: 10,000/1,000/1,000 scenarios
- Split seeds: 11001/22001/33001
- Test evaluation: 1,000 matched scenarios per N

Available V6 configs:

```text
configs/v3/pilot_v6_soft_anchor_n16.yaml
configs/v3/pilot_v6_soft_anchor_n32.yaml
configs/v3/pilot_v6_soft_anchor_n64.yaml
configs/v3/pilot_v6_soft_anchor_n96.yaml
configs/v3/pilot_v6_soft_anchor_n128.yaml
```

The corrected baseline protocol continues to use
`configs/v3/constrained_action_n32.yaml` and the equivalent config for each N.

## Verify canonical ScenarioBanks

The 15 `.npz` banks are kept locally under `artifacts/scenario_banks/` and are
not committed because they total about 313 MB. Before training, run:

```bash
python scripts/prepare_v6_scenario_banks.py --verify-existing
```

The command checks count, seed, split disjointness, and all five frozen test
checksums. Provenance and every checksum are recorded in
`results/physical_v6_full/SCENARIO_BANK_MANIFEST.json`.

The permanent V6 r2 binary evidence is indexed under
`results/physical_v6_full_r2/evidence/`. Its GitHub Release retains all 15
ScenarioBanks and all 75 audited training archives, including checkpoints,
training/validation logs, test outputs, timing and per-job provenance. Download
and SHA-256 verify it with:

```bash
python results/physical_v6_full_r2/evidence/download_evidence.py
```

## Train one TD3 job

```bash
python scripts/pilot_structure_aware_td3.py \
  --method td3 \
  --config configs/v3/pilot_v6_soft_anchor_n32.yaml \
  --seed 0 \
  --tag physical_v6_n32_100k \
  --output-root results/physical_v6_full/runs
```

Each job saves initial, best, and latest checkpoints and evaluates all three.
It also records elapsed training time, interactions/second, validation and test
times, device/runtime versions, and peak allocated GPU memory in `training.csv`,
`validation_summary.csv`, `manifest.json`, and `timing.json`.

## Corrected baseline evidence

`results/six_method_v2/` contains the frozen corrected AO-SCA, corrected
AO-Grid, AnalyticalRIS, DRL reference rows, CPU latency, tables, figures, audit,
and provenance. Its latency table contains exactly 100 latency samples for
every method/N pair and was produced with
`benchmark_latency_v2.py --warmup 10 --count 100`.

The historical V1 directory is not required: the V2 DRL raw CSV is the
byte-identical frozen input and all default V2 scripts point to that
self-contained path.
