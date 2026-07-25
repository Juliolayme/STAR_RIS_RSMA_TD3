# Codex Repository Guide for Thesis Writing

## Source of truth

- Physical model and environment: `star_ris_rsma/`
- Experiment entrypoints and result generation: `scripts/`
- Declared experiment configurations: `configs/`
- Original final notebook sources: `kaggle_notebooks/`
- Saved Kaggle notebook source and output: `kaggle_runs/`
- Audited tables, figures, raw evidence, and provenance: `results/`

## Required consistency rule

Derive every equation, variable definition, constraint, state/action description,
reward term, neural-network detail, training setting, baseline description, and
evaluation protocol from the checked-in code/configuration. Do not introduce a
theoretical assumption that is absent from the implementation without labeling it
as a proposed extension.

## Final scientific positioning

- TD3 is a QoS-reliable, low-latency quality/latency compromise.
- AO-SCA has higher sum-rate in the final experiment and is a local iterative baseline,
not a global optimum or upper bound.
- TD3 is much faster than AO-SCA and AO-Grid, but AnalyticalRIS is faster than TD3 and
fails QoS in this experiment.
- Report both paired t-Holm and exact Wilcoxon-Holm results; do not claim significance
under both tests.
- Use `results/FINAL_RESULTS_REVIEW.md` and `results/FINAL_RESULTS_AUDIT.json` as mandatory
reviewer guardrails when drafting the thesis and paper.

## Reproducibility anchors

- Scientific algorithm commit: `89c39da461523a7f5911a302cb9415aeaa5824ce`
- Report-generation commit: `b67adee55def9dbc32c8acded09dad68c51ee49a`
- Five STAR-RIS sizes: 16, 32, 64, 96, 128
- TD3: eight independent training seeds per size
- Evaluation: 1,000 locked test scenarios per seed/size
