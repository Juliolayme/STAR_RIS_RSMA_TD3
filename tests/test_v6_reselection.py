"""Re-selecting the reported checkpoint must not require retraining."""

from pathlib import Path
import json
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESELECT = runpy.run_path(ROOT / "scripts" / "reselect_v6_checkpoint.py")
select = RESELECT["select"]
resolve_checkpoint = RESELECT["resolve_checkpoint"]
candidate_rows = RESELECT["candidate_rows"]


FRONTIER = [
    {"eval_step": 1000, "checkpoint": "candidate_step1000.pt", "mean_sum_rate": 9.0, "mean_violation": 0.0},
    {"eval_step": 3000, "checkpoint": "candidate_step3000.pt", "mean_sum_rate": 12.0, "mean_violation": 0.010},
    {"eval_step": 4000, "checkpoint": "candidate_step4000.pt", "mean_sum_rate": 15.0, "mean_violation": 0.030},
]


@pytest.mark.parametrize(
    ("tolerance", "expected"),
    [(0.0, 1000), (0.009, 1000), (0.010, 3000), (0.029, 3000), (0.030, 4000), (1.0, 4000)],
)
def test_tolerance_moves_the_selection_along_the_frontier(tolerance, expected) -> None:
    assert select(FRONTIER, tolerance)["eval_step"] == expected


def test_selection_fails_loudly_when_nothing_clears_the_tolerance() -> None:
    strict = [dict(FRONTIER[2])]
    with pytest.raises(RuntimeError, match="frontier starts at"):
        select(strict, 0.001)


def test_missing_candidate_index_is_reported_rather_than_guessed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="retraining"):
        candidate_rows(tmp_path)


def _run_dir(tmp_path: Path, *, files: tuple[str, ...], best_step: int) -> Path:
    for name in files:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "best_validation.json").write_text(json.dumps({"eval_step": best_step}))
    return tmp_path


def test_steps_without_a_candidate_file_fall_back_to_the_named_checkpoints(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, files=("initial.pt", "best.pt", "latest.pt"), best_step=2500)
    summary = {"train_steps": 4000}
    for step, expected in ((0, "initial.pt"), (4000, "latest.pt"), (2500, "best.pt")):
        entry = {"eval_step": step, "checkpoint": f"candidate_step{step}.pt"}
        assert resolve_checkpoint(run, entry, summary).name == expected


def test_an_unrecoverable_step_is_an_error_not_a_silent_substitution(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, files=("initial.pt", "best.pt", "latest.pt"), best_step=2500)
    entry = {"eval_step": 1234, "checkpoint": "candidate_step1234.pt"}
    with pytest.raises(RuntimeError, match="no stored checkpoint"):
        resolve_checkpoint(run, entry, {"train_steps": 4000})
