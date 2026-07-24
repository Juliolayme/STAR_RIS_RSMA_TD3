# Kaggle final thesis/paper execution plan

This file is the execution contract for Codex. The objective is to rerun the
complete STAR-RIS-assisted RSMA study from one pinned repository commit and
produce a submission-ready evidence bundle for the thesis and paper.

## 1. Non-negotiable scientific rules

1. Do not change the channel model, RSMA rate equations, reward, TD3
   hyperparameters, baseline algorithms, scenario counts, seeds, or timing
   boundaries.
2. Notebooks 01-05 clone the repository and check out the pinned scientific
   algorithm commit `89c39da461523a7f5911a302cb9415aeaa5824ce`.
   Notebook 06 checks out the pinned report-code commit
   `bc7f88955060cfd156322a2a9cba0716dc5be52e`, which adds only the Kaggle
   orchestration/report modules on top of that scientific commit.
3. TD3 uses five values of `N = 16, 32, 64, 96, 128`, eight training seeds
   `0..7`, and 1,000 locked test scenarios per seed.
4. AO-Grid, AO-SCA, and AnalyticalRIS each use the same 1,000 locked test
   scenarios for every N.
5. The latency benchmark runs all four methods sequentially in the same process,
   with one PyTorch/BLAS CPU thread, on the same scenario row.
6. Keep all raw CSV files, manifests, logs, figures, and tables. Never replace a
   failed or missing run with synthetic numbers.
7. Use Kaggle Internet ON and accelerator `NvidiaTeslaT4`. The baseline
   notebooks do not mathematically require a GPU, but the same Kaggle execution
   profile is retained for operational consistency.
8. Maximum timeout for every notebook is 43,200 seconds (12 hours).

## 2. Notebook order

The notebooks are under `kaggle_notebooks/`.

| Order | Notebook | Output stage ID | Purpose |
|---:|---|---|---|
| 1 | `01_td3_low_n_training.ipynb` | `td3_low_n` | TD3, N=16/32/64, seeds 0-7, evaluation and convergence plots |
| 2 | `02_td3_high_n_training.ipynb` | `td3_high_n` | TD3, N=96/128, seeds 0-7, evaluation and convergence plots |
| 3 | `03_ao_grid_evaluation.ipynb` | `ao_grid` | AO-Grid on 5×1,000 locked scenarios |
| 4 | `04_ao_sca_evaluation.ipynb` | `ao_sca` | AO-SCA on 5×1,000 locked scenarios |
| 5 | `05_analytical_ris_evaluation.ipynb` | `analytical_ris` | AnalyticalRIS on 5×1,000 locked scenarios |
| 6 | `06_final_academic_report_and_benchmark.ipynb` | final bundle | Validation, statistics, figures, tables and fair CPU latency benchmark |

Notebooks 01-05 are independent after cloning the pinned repository commit.
Notebook 06 must receive the outputs of notebooks 01-05 as Kaggle kernel
sources.

## 3. Expected outputs

### TD3 notebooks

Each TD3 notebook must retain:

- `final_td3_v3/N{N}/seed_{seed}/train/best.pt`
- `training.csv`
- `validation_raw.csv`
- `validation_summary.csv`
- `best_validation.json`
- `manifest.json`
- `test.csv`
- `train.log`, `evaluate.log`, `audit.log`
- stage convergence figures in PNG and PDF
- `STAGE_MANIFEST.json`

`latest.pt` is intentionally removed after successful evaluation to reduce
Kaggle output size. `best.pt` is retained for the final latency benchmark.

### Baseline notebooks

Each baseline notebook must retain:

- all chunk-level raw CSV and logs
- one merged `{METHOD}_RAW_ALL.csv`
- one `{METHOD}_SUMMARY.csv`
- method-specific PNG and PDF figures
- `STAGE_MANIFEST.json`

### Final notebook

The final notebook must produce:

- `FINAL_THESIS_PAPER_BUNDLE/`
- `FINAL_THESIS_PAPER_BUNDLE.zip`
- 300-DPI PNG figures and vector PDF figures
- CSV, Markdown and LaTeX tables
- merged TD3/baseline/latency raw CSV files
- eight-seed Student-t 95% confidence intervals
- seed-level paired t-tests and Wilcoxon signed-rank tests
- Holm-adjusted p-values and Cohen's dz
- complete CPU latency mean, median, standard deviation, p95 and p99
- `REPRODUCIBILITY_MANIFEST.json`
- `RESULTS_README.md`

## 4. Codex CLI procedure

### 4.1 Prerequisites

Use the current official Kaggle CLI:

```bash
python -m pip install --upgrade kaggle
kaggle auth login
```

Alternatively set `KAGGLE_API_TOKEN` or install the Kaggle credential file
before continuing.

Set the account name:

```bash
export KAGGLE_USERNAME="REPLACE_WITH_KAGGLE_USERNAME"
```

Clone the repository branch containing the notebooks:

```bash
git clone --branch agent/td3-qos-scalability-v2 \
  https://github.com/Juliolayme/STAR_RIS_RSMA_TD3.git
cd STAR_RIS_RSMA_TD3
```

### 4.2 Create local Kaggle kernel folders

Codex should execute the following Python script from the repository root. It
copies each notebook into a separate publish folder and creates
`kernel-metadata.json`.

```bash
python - <<'PY'
import json
import os
import shutil
from pathlib import Path

username = os.environ["KAGGLE_USERNAME"]
source = Path("kaggle_notebooks")
publish = Path(".kaggle_publish")
publish.mkdir(exist_ok=True)

items = [
    ("01_td3_low_n_training.ipynb", "star-ris-td3-low-n-final", "STAR-RIS TD3 low-N final"),
    ("02_td3_high_n_training.ipynb", "star-ris-td3-high-n-final", "STAR-RIS TD3 high-N final"),
    ("03_ao_grid_evaluation.ipynb", "star-ris-ao-grid-final", "STAR-RIS AO-Grid final"),
    ("04_ao_sca_evaluation.ipynb", "star-ris-ao-sca-final", "STAR-RIS AO-SCA final"),
    ("05_analytical_ris_evaluation.ipynb", "star-ris-analytical-ris-final", "STAR-RIS AnalyticalRIS final"),
    ("06_final_academic_report_and_benchmark.ipynb", "star-ris-final-academic-bundle", "STAR-RIS final academic bundle"),
]

stage_kernel_sources = [
    f"{username}/star-ris-td3-low-n-final",
    f"{username}/star-ris-td3-high-n-final",
    f"{username}/star-ris-ao-grid-final",
    f"{username}/star-ris-ao-sca-final",
    f"{username}/star-ris-analytical-ris-final",
]

for notebook, slug, title in items:
    target = publish / slug
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copy2(source / notebook, target / notebook)

    is_final = notebook.startswith("06_")
    metadata = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": notebook,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": stage_kernel_sources if is_final else [],
        "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
PY
```

### 4.3 Push and run notebooks 01-05

Push each notebook with a T4 and a 12-hour timeout:

```bash
kaggle kernels push -p .kaggle_publish/star-ris-td3-low-n-final \
  --accelerator NvidiaTeslaT4 --timeout 43200

kaggle kernels push -p .kaggle_publish/star-ris-td3-high-n-final \
  --accelerator NvidiaTeslaT4 --timeout 43200

kaggle kernels push -p .kaggle_publish/star-ris-ao-grid-final \
  --accelerator NvidiaTeslaT4 --timeout 43200

kaggle kernels push -p .kaggle_publish/star-ris-ao-sca-final \
  --accelerator NvidiaTeslaT4 --timeout 43200

kaggle kernels push -p .kaggle_publish/star-ris-analytical-ris-final \
  --accelerator NvidiaTeslaT4 --timeout 43200
```

Do not start notebook 06 until all five source notebooks report successful
completion:

```bash
kaggle kernels status "$KAGGLE_USERNAME/star-ris-td3-low-n-final"
kaggle kernels status "$KAGGLE_USERNAME/star-ris-td3-high-n-final"
kaggle kernels status "$KAGGLE_USERNAME/star-ris-ao-grid-final"
kaggle kernels status "$KAGGLE_USERNAME/star-ris-ao-sca-final"
kaggle kernels status "$KAGGLE_USERNAME/star-ris-analytical-ris-final"
```

If a notebook fails:

1. Download its output/logs.
2. Fix only infrastructure or notebook orchestration errors.
3. Do not modify scientific settings.
4. Push the same notebook again; completed seed/chunk outputs are resume-safe
   only within an interactive session, so a failed committed Kaggle version may
   need to rerun that stage.

### 4.4 Run notebook 06

After all five source notebooks succeed, push notebook 06. Its
`kernel_sources` field attaches their saved outputs automatically.

```bash
kaggle kernels push -p .kaggle_publish/star-ris-final-academic-bundle \
  --accelerator NvidiaTeslaT4 --timeout 43200

kaggle kernels status "$KAGGLE_USERNAME/star-ris-final-academic-bundle"
```

### 4.5 Download the final evidence bundle

```bash
mkdir -p kaggle_downloads/final
kaggle kernels output \
  "$KAGGLE_USERNAME/star-ris-final-academic-bundle" \
  --path kaggle_downloads/final \
  --force
```

The required deliverable is:

```text
kaggle_downloads/final/FINAL_THESIS_PAPER_BUNDLE.zip
```

Also retain the uncompressed `FINAL_THESIS_PAPER_BUNDLE/` directory when
available.

## 5. Mandatory validation before accepting the result

Codex must reject the final result if any check below fails:

- exactly 40 TD3 test runs: 5 N × 8 seeds
- exactly 1,000 unique scenarios in every TD3 test run
- exactly 1,000 unique scenarios per N for each deterministic baseline
- exact N coverage `{16,32,64,96,128}`
- seed coverage `{0,1,2,3,4,5,6,7}`
- no NaN or Inf in quality or latency metrics
- one bank checksum per N, matching TD3 and all baselines
- all five stage manifests use the scientific algorithm commit
  `89c39da461523a7f5911a302cb9415aeaa5824ce`
- notebook 06 checks out report-code commit
  `bc7f88955060cfd156322a2a9cba0716dc5be52e`
- all four methods exist in the final quality and latency tables
- latency raw output contains 5,000 rows
- latency summary contains mean, median, std, p95 and p99
- CPU benchmark metadata reports one PyTorch thread and one BLAS thread
- no claim that TD3 is faster than AnalyticalRIS when the measured ratio is
  below one
- no claim that AO-SCA is a global optimum or upper bound

## 6. Safe thesis/paper interpretation

The intended conclusion is:

> TD3 amortizes the online optimization into a neural forward pass and
> substantially reduces CPU decision latency relative to iterative AO-SCA and
> AO-Grid while maintaining high QoS satisfaction and useful sum-rate.
> AnalyticalRIS can have lower latency than TD3, but its quality and QoS
> performance must be reported jointly; therefore TD3 is a favorable
> quality-computation trade-off, not necessarily the fastest absolute method.

Do not compare absolute latency scaling across N without noting that different
Kaggle sessions or hardware allocations can alter cross-run timing. Notebook 06
solves this for the primary benchmark by timing all methods for a given N in the
same process and session.
