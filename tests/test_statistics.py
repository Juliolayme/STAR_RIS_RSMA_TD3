import numpy as np
import pandas as pd

from star_ris_rsma.statistics import holm_adjust, paired_comparisons, summarize


def test_holm_adjust_is_bounded_and_monotone_in_sorted_order():
    raw = [0.01, 0.04, 0.03]
    adjusted = holm_adjust(raw)
    assert all(0 <= value <= 1 for value in adjusted)
    order = np.argsort(raw)
    assert np.all(np.diff(np.asarray(adjusted)[order]) >= -1e-12)


def test_summary_and_paired_results():
    df = pd.DataFrame({
        "method": ["a"] * 4 + ["b"] * 4,
        "scenario": list(range(4)) * 2,
        "sum_rate": [2.0, 3.2, 4.1, 5.4, 1.1, 2.0, 3.3, 4.0],
    })
    summary = summarize(df)
    paired = paired_comparisons(df)
    assert set(summary.method) == {"a", "b"}
    assert paired.iloc[0]["n_pairs"] == 4
    assert paired.iloc[0]["mean_difference"] > 0
