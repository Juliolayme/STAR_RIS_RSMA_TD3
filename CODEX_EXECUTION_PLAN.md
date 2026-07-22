# CODEX EXECUTION PLAN — STAR-RIS–RSMA TD3

## 1. Frozen research scope

The thesis remains:

> TỐI ƯU PHÂN BỔ TÀI NGUYÊN SỬ DỤNG HỌC TĂNG CƯỜNG SÂU TRONG MẠNG STAR-RIS HỖ TRỢ RSMA

Frozen comparison design:

- TD3: primary DRL method.
- DDPG and PPO: DRL baselines.
- AO-SCA: primary conventional optimization baseline.
- AO-Grid and AnalyticalRIS: supplementary references.
- NoRIS, FixedRIS, RandomRIS and Equal-Power: ablations only.
- Do not reintroduce MADDPG, CTDE, TD3-Matched or old multi-agent claims.

## 2. Six research blockers completed

1. **AO-SCA corrected**: proximal first-order block surrogate is constructed and solved in feasible physical variables with monotone backtracking.
2. **AO-Grid corrected**: deterministic coordinate codebooks replace random perturbations.
3. **ScenarioBank locked**: train/validation/test NPZ banks have metadata, checksums, shape validation and disjointness checks.
4. **Checkpoint and raw CSV completed**: best checkpoint uses validation only; deterministic test evaluation writes one row per scenario.
5. **Latency/statistics/plots completed**: CPU one-thread benchmark, 95% CI, paired t-test, Wilcoxon, Holm correction, effect size and result plots are implemented.
6. **Ablation and multi-N completed**: NoRIS, FixedRIS, RandomRIS, Equal-Power and configs for N=16/32/64/96/128 are implemented.

## 3. Physical invariants

All methods must use `src/star_ris_rsma/env.py` and `src/star_ris_rsma/physics.py`.

- SISO BS and users.
- STAR-RIS energy-splitting mode.
- `beta_t + beta_r = 1` for every element.
- Complex amplitudes use `sqrt(beta)`, not `beta`.
- Common RSMA stream is decoded by every user.
- Common rate is the minimum common decodable rate.
- Total power and common-rate shares are simplex-normalized.
- Report sum-rate, per-user QoS fraction, all-users QoS probability and QoS violation.
- Learned evaluation must be deterministic and exploration-free.
- Do not alter channel/rate equations to improve one algorithm.

## 4. Mandatory validation before Kaggle

```bash
python -m pip install -e .[dev]
pytest -q
```

Create small smoke banks and run all commands in README before full experiments. Acceptance conditions:

- all tests pass;
- all outputs are finite;
- train, validation and test banks are disjoint;
- checkpoint selection reads validation only;
- test evaluation is deterministic;
- AO-SCA exact objective history is non-decreasing within tolerance;
- AO-SCA reports surrogate/backtracking metadata;
- AO-Grid reports declared finite codebooks and contains no random perturbation;
- merge aborts on duplicate `(method, seed, scenario)` keys;
- every final row includes method, seed, scenario, config hash and provenance where applicable.

## 5. Kaggle execution order

For each N in `16 32 64 96 128`:

1. Create locked ScenarioBanks once.
2. Generate Kaggle jobs.
3. Run TD3/DDPG/PPO for seeds 0-7, one learned seed per GPU session.
4. Evaluate every `best.pt` on the same locked test bank.
5. Run AO-SCA/AO-Grid/AnalyticalRIS on non-overlapping scenario shards.
6. Run TD3 ablations for all seeds.
7. Run CPU one-thread latency benchmarks.
8. Merge raw CSV and reject duplicates.
9. Generate statistics and plots only from merged raw CSV.
10. Archive configs, bank manifests, checkpoints, raw CSV, analysis CSV, figures and Git commit SHA.

## 6. Scientific reporting contract

- AO-SCA is a local proximal first-order alternating solver, not a global optimum or upper bound.
- AO-Grid is a coarse coordinate codebook search.
- AnalyticalRIS is phase alignment with equal power and equal common-rate allocation.
- Do not tune any method on the test bank.
- Do not remove a strong baseline because it outperforms TD3.
- Do not reuse old MADDPG values, figures, p-values or scalability claims.

## 7. Thesis merge contract

Only update the thesis after the locked full experiment is complete. Use this method statement:

> TD3 is the primary DRL method; DDPG and PPO are DRL baselines; AO-SCA is the primary conventional optimization baseline; AO-Grid and AnalyticalRIS are supplementary references.

Report sum-rate, user QoS fraction, all-users QoS probability, constraint violation, mean and 95% CI, inference/solve latency, AO iterations/evaluations and paired tests with Holm correction.
