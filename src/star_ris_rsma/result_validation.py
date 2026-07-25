from __future__ import annotations

import numpy as np
import pandas as pd


CORE_METRICS = ("sum_rate", "qos_fraction", "all_qos", "violation")
_BOOL_MAP = {
    "true": 1.0,
    "false": 0.0,
    "1": 1.0,
    "0": 0.0,
}


def coerce_core_metrics(
    frame: pd.DataFrame,
    *,
    context: str = "metrics",
    require_finite: bool = False,
) -> pd.DataFrame:
    """Convert the core result metrics to a homogeneous float64 frame.

    CSV readers may infer ``all_qos`` as bool, integer, or string while the
    remaining metrics are floats. A mixed bool/float selection can become an
    object ndarray, which is not accepted by ``numpy.isfinite``. This helper is
    the single conversion boundary used by result audits and aggregators.
    """
    missing = [column for column in CORE_METRICS if column not in frame.columns]
    if missing:
        raise ValueError(f"{context}: missing metric columns {missing}")

    numeric = pd.DataFrame(index=frame.index)
    for column in ("sum_rate", "qos_fraction", "violation"):
        numeric[column] = pd.to_numeric(frame[column], errors="coerce").astype(
            np.float64
        )

    all_qos = pd.to_numeric(frame["all_qos"], errors="coerce")
    unresolved = all_qos.isna()
    if unresolved.any():
        mapped = (
            frame.loc[unresolved, "all_qos"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(_BOOL_MAP)
        )
        all_qos.loc[unresolved] = mapped
    numeric["all_qos"] = all_qos.astype(np.float64)
    numeric = numeric.loc[:, list(CORE_METRICS)]

    if require_finite:
        values = numeric.to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(values)
        if not finite.all():
            bad_rows, bad_columns = np.where(~finite)
            examples = [
                {
                    "row": int(row),
                    "column": CORE_METRICS[int(column)],
                    "raw_value": repr(
                        frame.iloc[int(row)][CORE_METRICS[int(column)]]
                    ),
                }
                for row, column in zip(bad_rows[:10], bad_columns[:10])
            ]
            raise ValueError(
                f"{context}: non-finite or non-numeric core metrics {examples}"
            )

    return numeric


def replace_core_metrics(
    frame: pd.DataFrame,
    *,
    context: str = "metrics",
    require_finite: bool = True,
) -> pd.DataFrame:
    """Return a copy whose core metric columns are explicitly float64."""
    result = frame.copy()
    numeric = coerce_core_metrics(
        result,
        context=context,
        require_finite=require_finite,
    )
    for column in CORE_METRICS:
        result[column] = numeric[column].to_numpy(dtype=np.float64, copy=False)
    return result
