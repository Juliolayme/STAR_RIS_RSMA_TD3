from __future__ import annotations

import runpy
from pathlib import Path

import pandas as pd

import star_ris_rsma.latency as latency_module


_ORIGINAL_SUMMARIZE_LATENCY = latency_module.summarize_latency


def summarize_latency_by_n(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize a merged latency frame without mixing ScenarioBanks across N.

    ``validate_latency_frame`` intentionally requires scenario IDs and the bank
    checksum to be unique within one locked ScenarioBank.  Aggregate artifacts
    contain one such bank for each N, so validation and summarization must be
    performed independently for every N before concatenating the summaries.
    """
    if frame.empty:
        raise ValueError("Cannot summarize an empty latency frame")

    summaries = [
        _ORIGINAL_SUMMARIZE_LATENCY(group.copy())
        for _, group in frame.groupby("n_ris", sort=True)
    ]
    return pd.concat(summaries, ignore_index=True)


def main() -> None:
    # summarize_latency_benchmark imports the function by name. Replace it only
    # for this aggregate entrypoint while retaining strict single-bank checks.
    latency_module.summarize_latency = summarize_latency_by_n
    script = Path(__file__).with_name("summarize_latency_benchmark.py")
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
