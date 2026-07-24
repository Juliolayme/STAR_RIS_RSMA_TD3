from __future__ import annotations

"""Main entrypoint used by Kaggle notebook 06 to build the evidence bundle."""

import datetime as dt
import json
import shutil
from pathlib import Path

from kaggle_final_report_common import (
    ALGORITHM_COMMIT,
    FIGURE_DIR,
    FINAL_ROOT,
    N_VALUES,
    RAW_DIR,
    REPO_URL,
    SEEDS,
    current_repository_commit,
    discover_stage_roots,
    prepare_output_directories,
    validate_stage_manifests,
    write_table_formats,
)
from kaggle_final_report_latency import plot_latency, run_fair_cpu_latency
from kaggle_final_report_quality import (
    build_baseline_timing_table,
    build_performance_tables,
    load_baseline_outputs,
    load_td3_outputs,
    paired_seed_level_tests,
    plot_final_quality,
    plot_training_curves,
    validate_cross_method_checksums,
)


def write_results_readme(report_commit: str) -> None:
    """Write a concise map of submission-ready figures and tables."""
    figure_files = sorted(path.name for path in FIGURE_DIR.glob("*.png"))
    lines = [
        "# Final thesis and paper result bundle",
        "",
        f"- Scientific algorithm commit: `{ALGORITHM_COMMIT}`",
        f"- Report-generation commit: `{report_commit}`",
        "- TD3: 5 N values × 8 seeds × 1,000 locked test scenarios.",
        "- AO-SCA, AO-Grid, AnalyticalRIS: 1,000 matched scenarios per N.",
        "- CPU latency: all methods timed sequentially in the same process per N.",
        "",
        "## Main tables",
        "",
        "- `tables/TABLE_FINAL_PERFORMANCE.*`",
        "- `tables/TABLE_TD3_8SEED_CI95.*`",
        "- `tables/TABLE_PAIRED_WILCOXON_HOLM.*`",
        "- `tables/TABLE_CPU_LATENCY.*`",
        "- `tables/TABLE_BASELINE_SOLVER_TIME_DESCRIPTIVE.*`",
        "",
        "## Paper-ready figures",
        "",
        *[f"- `figures/{name}`" for name in figure_files],
        "",
        "## Interpretation guardrails",
        "",
        "- AO-SCA is a local iterative baseline, not a global optimum or upper bound.",
        "- Do not call TD3 the fastest absolute method when AnalyticalRIS is faster.",
        "- Interpret latency jointly with sum-rate and QoS quality.",
    ]
    (FINAL_ROOT / "RESULTS_README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Validate all stages, compute statistics, benchmark latency, and archive."""
    prepare_output_directories()
    report_commit = current_repository_commit()
    stage_roots = discover_stage_roots()
    stage_manifests = validate_stage_manifests(stage_roots)

    print("Attached stages:")
    for stage_id, root in sorted(stage_roots.items()):
        print(f"  {stage_id}: {root}")

    td3_test, td3_training, td3_validation = load_td3_outputs(stage_roots)
    baseline_raw = load_baseline_outputs(stage_roots)
    validate_cross_method_checksums(td3_test, baseline_raw)

    # Preserve the exact raw evidence used by all tables and figures.
    td3_test.to_csv(RAW_DIR / "TD3_TEST_RAW_ALL.csv", index=False)
    td3_training.to_csv(RAW_DIR / "TD3_TRAINING_RAW_ALL.csv", index=False)
    td3_validation.to_csv(RAW_DIR / "TD3_VALIDATION_RAW_ALL.csv", index=False)
    baseline_raw.to_csv(RAW_DIR / "BASELINES_RAW_ALL.csv", index=False)

    td3_summary, final_table = build_performance_tables(td3_test, baseline_raw)
    baseline_timing = build_baseline_timing_table(baseline_raw)
    statistical_tests = paired_seed_level_tests(td3_test, baseline_raw)
    write_table_formats(td3_summary, "TABLE_TD3_8SEED_CI95")
    write_table_formats(final_table, "TABLE_FINAL_PERFORMANCE")
    write_table_formats(statistical_tests, "TABLE_PAIRED_WILCOXON_HOLM")
    write_table_formats(baseline_timing, "TABLE_BASELINE_SOLVER_TIME_DESCRIPTIVE")
    plot_training_curves(td3_training, td3_validation)
    plot_final_quality(final_table)

    latency_raw, latency_summary, latency_metadata = run_fair_cpu_latency(stage_roots)
    latency_raw.to_csv(RAW_DIR / "LATENCY_RAW_ALL.csv", index=False)
    write_table_formats(latency_summary, "TABLE_CPU_LATENCY")
    plot_latency(latency_summary, final_table)

    reproducibility = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": REPO_URL,
        "scientific_algorithm_commit": ALGORITHM_COMMIT,
        "report_generation_commit": report_commit,
        "stage_manifests": stage_manifests,
        "n_values": list(N_VALUES),
        "td3_seeds": list(SEEDS),
        "test_scenarios_per_n": 1000,
        "latency_scenarios_per_n": 1000,
        "latency_metadata": latency_metadata,
        "latency_protocol": {
            "same_process_per_n": True,
            "torch_threads": 1,
            "blas_threads": 1,
            "actor_repeats": 100,
            "decode_repeats": 100,
            "end_to_end_repeats": 20,
            "actor_warmup": 500,
            "solver_warmup_scenarios": 2,
            "excluded": [
                "channel generation",
                "ScenarioBank loading",
                "environment reset",
            ],
        },
    }
    (FINAL_ROOT / "REPRODUCIBILITY_MANIFEST.json").write_text(
        json.dumps(reproducibility, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_results_readme(report_commit)

    archive = shutil.make_archive(
        "/kaggle/working/FINAL_THESIS_PAPER_BUNDLE",
        "zip",
        root_dir=FINAL_ROOT,
    )
    print("Final directory:", FINAL_ROOT)
    print("Final archive:", archive)
    print("PNG figures:", len(list(FIGURE_DIR.glob("*.png"))))


if __name__ == "__main__":
    main()
