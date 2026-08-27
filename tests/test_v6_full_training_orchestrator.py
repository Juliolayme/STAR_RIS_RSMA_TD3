from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR = runpy.run_path(
    ROOT / "scripts" / "orchestrate_kaggle_v6_full.py"
)
ASSIGNMENTS = ORCHESTRATOR["ASSIGNMENTS"]
MAX_ACTIVE_PER_ACCOUNT = ORCHESTRATOR["MAX_ACTIVE_PER_ACCOUNT"]
N_VALUES = ORCHESTRATOR["N_VALUES"]
REQUIRED_ARCHIVE_FILES = ORCHESTRATOR["REQUIRED_ARCHIVE_FILES"]
SEEDS = ORCHESTRATOR["SEEDS"]
Job = ORCHESTRATOR["Job"]
METHODS = ORCHESTRATOR["METHODS"]
runner_source = ORCHESTRATOR["runner_source"]


def test_orchestrator_covers_all_25_jobs_across_three_accounts() -> None:
    combinations = [item for jobs in ASSIGNMENTS.values() for item in jobs]
    expected = {(n_ris, seed) for n_ris in N_VALUES for seed in SEEDS}
    assert len(ASSIGNMENTS) == 3
    assert sorted(len(jobs) for jobs in ASSIGNMENTS.values()) == [8, 8, 9]
    assert len(combinations) == 25
    assert set(combinations) == expected
    assert MAX_ACTIVE_PER_ACCOUNT == 2


def test_orchestrator_requires_reviewer_evidence() -> None:
    assert {
        "initial.pt",
        "best.pt",
        "latest.pt",
        "summary.json",
        "timing.json",
        "test_initial_raw.csv",
        "test_best_raw.csv",
        "test_latest_raw.csv",
        "SCENARIO_BANK_VERIFICATION.json",
        "KAGGLE_JOB_PROVENANCE.json",
    }.issubset(REQUIRED_ARCHIVE_FILES)


def test_generated_kaggle_runner_is_valid_python() -> None:
    source = runner_source(Job("td3", 128, 4), "a" * 40)
    compile(source, "generated_v6_runner.py", "exec")
    assert "physical_v6_n128_seed4" in source
    assert '"--n-ris", "128", "--verify-existing"' in source


def test_td3_names_are_unchanged_so_finished_kernels_stay_adoptable() -> None:
    job = Job("td3", 128, 4)
    assert job.slug == "star-ris-td3-v6-full-n128-seed-4"
    assert job.tag == "physical_v6_n128_100k"
    assert job.archive_name == "physical_v6_n128_seed4.zip"
    assert job.output_root == "/kaggle/working/physical_v6_full"


def test_comparator_methods_get_disjoint_kernel_and_archive_names() -> None:
    assert set(METHODS) == {"td3", "ddpg", "ppo"}
    names = {
        method: Job("td3", 32, 1, method)
        for method in METHODS
    }
    assert len({job.slug for job in names.values()}) == 3
    assert len({job.archive_name for job in names.values()}) == 3
    assert len({job.output_root for job in names.values()}) == 3
    assert names["ddpg"].archive_name == "physical_v6_ddpg_n32_seed1.zip"
    assert names["ppo"].slug == "star-ris-ppo-v6-full-n32-seed-1"


def test_generated_runner_trains_the_requested_method() -> None:
    for method in METHODS:
        source = runner_source(Job("ddpg", 96, 2, method), "c" * 40)
        compile(source, f"generated_{method}_runner.py", "exec")
        assert f'"--method", "{method}"' in source
        assert f'"method": "{method}"' in source
        assert '"--n-ris", "96", "--verify-existing"' in source
