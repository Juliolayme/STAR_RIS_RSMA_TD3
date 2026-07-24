from __future__ import annotations

import pandas as pd

from .latency import summarize_latency


def summarize_latency_by_n(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize a merged latency frame one locked ScenarioBank at a time.

    Scenario identifiers and bank checksums are only unique within one RIS size.
    Validate and summarize each ``n_ris`` group independently, then concatenate
    the resulting one-row summaries.
    """
    if frame.empty:
        raise ValueError("Cannot summarize an empty latency frame")
    if "n_ris" not in frame.columns:
        raise ValueError("Latency frame missing n_ris")

    summaries = [
        summarize_latency(group.copy())
        for _, group in frame.groupby("n_ris", sort=True)
    ]
    return pd.concat(summaries, ignore_index=True)
