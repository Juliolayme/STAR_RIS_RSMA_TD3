from __future__ import annotations

from star_ris_rsma import experiment


def test_git_commit_does_not_mislabel_kaggle_runtime_id(monkeypatch) -> None:
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_ID", "runtime-id-not-a-commit")
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda *args, **kwargs: "abc123\n")

    assert experiment._git_commit() == "abc123"


def test_explicit_git_commit_still_takes_precedence(monkeypatch) -> None:
    monkeypatch.setenv("GIT_COMMIT", "explicit-commit")
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_ID", "runtime-id-not-a-commit")

    assert experiment._git_commit() == "explicit-commit"
