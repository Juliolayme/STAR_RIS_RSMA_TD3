from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from star_ris_rsma.result_validation import (
    CORE_METRICS,
    coerce_core_metrics,
    replace_core_metrics,
)


def mixed_metric_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sum_rate": [3.1, "3.2", 3.3],
            "qos_fraction": [1.0, "0.75", 0.5],
            "all_qos": [True, "false", "1"],
            "violation": [0.0, "0.01", 0.02],
        }
    )


def test_coerce_core_metrics_handles_bool_and_string_bool() -> None:
    numeric = coerce_core_metrics(
        mixed_metric_frame(),
        require_finite=True,
    )

    assert tuple(numeric.columns) == CORE_METRICS
    assert all(dtype == np.dtype("float64") for dtype in numeric.dtypes)
    assert np.isfinite(numeric.to_numpy(dtype=np.float64)).all()
    assert numeric["all_qos"].tolist() == [1.0, 0.0, 1.0]


def test_replace_core_metrics_preserves_metadata_columns() -> None:
    frame = mixed_metric_frame()
    frame["scenario"] = [0, 1, 2]

    converted = replace_core_metrics(frame, require_finite=True)

    assert converted["scenario"].tolist() == [0, 1, 2]
    assert all(converted[column].dtype == np.dtype("float64") for column in CORE_METRICS)


def test_coerce_core_metrics_leaves_invalid_value_as_nan_when_allowed() -> None:
    frame = mixed_metric_frame().iloc[[0]].copy()
    frame.loc[frame.index[0], "sum_rate"] = "not-a-number"

    numeric = coerce_core_metrics(frame, require_finite=False)

    assert np.isnan(numeric.iloc[0]["sum_rate"])


def test_coerce_core_metrics_rejects_invalid_value_when_required() -> None:
    frame = mixed_metric_frame().iloc[[0]].copy()
    frame.loc[frame.index[0], "sum_rate"] = "not-a-number"

    with pytest.raises(ValueError, match="non-finite or non-numeric"):
        coerce_core_metrics(frame, require_finite=True)


def test_coerce_core_metrics_requires_all_metrics() -> None:
    with pytest.raises(ValueError, match="missing metric columns"):
        coerce_core_metrics(pd.DataFrame({"sum_rate": [1.0]}))
