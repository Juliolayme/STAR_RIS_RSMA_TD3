from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "train-td3-v6-full-25jobs.yml"
REUSABLE = ROOT / ".github" / "workflows" / "_kaggle-v6-single-job.yml"


def test_v6_full_workflow_covers_25_unique_jobs_with_per_account_limit() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    combinations: list[tuple[int, int]] = []
    expected_sizes = {"account-td3": 9, "account-ddpg": 8, "account-ppo": 8}
    for account, expected_size in expected_sizes.items():
        job = jobs[account]
        assert job["strategy"]["fail-fast"] is False
        assert job["strategy"]["max-parallel"] == 2
        entries = job["strategy"]["matrix"]["include"]
        assert len(entries) == expected_size
        combinations.extend((int(item["n_ris"]), int(item["seed"])) for item in entries)

    expected = {
        (n_ris, seed)
        for n_ris in (16, 32, 64, 96, 128)
        for seed in range(5)
    }
    assert len(combinations) == 25
    assert set(combinations) == expected


def test_reusable_v6_job_locks_training_and_output_evidence() -> None:
    text = REUSABLE.read_text(encoding="utf-8")
    for required in (
        '"--method", "td3"',
        '"--train-count", "10000"',
        '"--validation-count", "1000"',
        '"--test-count", "1000"',
        '"--verify-existing"',
        '"initial.pt", "best.pt", "latest.pt"',
        '"timing.json"',
        '"test_initial_raw.csv"',
        '"test_best_raw.csv"',
        '"test_latest_raw.csv"',
        "actions/upload-artifact@v4",
    ):
        assert required in text
