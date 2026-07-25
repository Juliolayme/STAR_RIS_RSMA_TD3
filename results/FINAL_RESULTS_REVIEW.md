# Final Results — Senior Reviewer Assessment

**Decision:** Accept for thesis integration with mandatory interpretation caveats.

## Evidence integrity

- Five STAR-RIS sizes: N = 16, 32, 64, 96, 128.
- TD3: 8 independent training seeds per N and 1,000 locked test scenarios per seed.
- Deterministic baselines: 1,000 matched scenarios per N.
- Raw coverage, finite-value, checksum/provenance, table, figure, and CPU-timing audits passed.
- Scientific algorithm commit: `89c39da461523a7f5911a302cb9415aeaa5824ce`.
- Report-generation commit: `b67adee55def9dbc32c8acded09dad68c51ee49a`.

## Thesis-ready findings

- TD3 mean QoS fraction ranges from **0.9892** to **0.9994**.
- TD3 all-users-QoS probability ranges from **0.9798** to **0.9985**.
- TD3 mean sum-rate changes from **10.7661** at N=16 to **12.4117** at N=128.
- Against AO-Grid, TD3 sum-rate improvement ranges from **169.1%** to **211.7%**.
- Against AnalyticalRIS, TD3 sum-rate improvement ranges from **440.7%** to **526.2%**.
- AO-SCA exceeds TD3 sum-rate by **4.4476–7.2246**; this quality gap must be reported.
- TD3 is **884×–3690×** faster than AO-SCA and **299×–1748×** faster than AO-Grid in the declared single-process CPU benchmark.
- AnalyticalRIS is approximately **2.07×–2.30×** faster than TD3, but it fails QoS in the reported experiment.

## Statistical interpretation

- Paired t-test with Holm correction: **15/15** comparisons significant at alpha=0.05.
- Exact Wilcoxon with Holm correction: **0/15** comparisons significant at alpha=0.05.
- Therefore, do not write an unconditional “statistically significant under both tests” claim.
- Report effect sizes, paired mean differences, both corrected tests, and the eight-seed limitation together.

## Mandatory writing guardrails

- Position TD3 as a **QoS-reliable, low-latency compromise**, not the maximum-sum-rate method.
- State explicitly that AO-SCA is a local iterative baseline, not a global optimum or upper bound.
- Do not call TD3 the fastest absolute method because AnalyticalRIS is faster.
- Explain that unbounded Student-t intervals can cross [0,1] for bounded metrics; preserve raw values and use clipped display or bounded/bootstrap intervals in the thesis.
- For a Q1 paper, justify weak baseline behavior and add ablations/stronger learning baselines before making broad superiority claims.

## Repository contents

- `final_thesis_paper_bundle/`: figures, tables, raw evidence, and reproducibility manifest.
- `FINAL_THESIS_PAPER_BUNDLE.zip`: submission-ready archive downloaded from Kaggle.
- `FINAL_RESULTS_AUDIT.json`: machine-readable audit and reviewer decision.
