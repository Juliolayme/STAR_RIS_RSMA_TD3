from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.audit_pilot_output import coerce_metric_frame


def test_coerce_metric_frame_handles_bool_and_string_bool() -> None:
    frame = pd.DataFrame(
        {
            "sum_rate": [3.1, "3.2", 3.3],
            "qos_fraction": [1.0, "0.75", 0.5],
            "all_qos": [True, "false", "1"],
            "violation": [0.0, "0.01", 0.02],
        }
    )

    numeric = coerce_metric_frame(frame)

    assert all(dtype == np.dtype("float64") for dtype in numeric.dtypes)
    assert np.isfinite(numeric.to_numpy(dtype=np.float64)).all()
    assert numeric["all_qos"].tolist() == [1.0, 0.0, 1.0]


def test_coerce_metric_frame_leaves_invalid_value_as_nan() -> None:
    frame = pd.DataFrame(
        {
            "sum_rate": ["not-a-number"],
            "qos_fraction": [1.0],
            "all_qos": [True],
            "violation": [0.0],
        }
    )

    numeric = coerce_metric_frame(frame)

    assert np.isnan(numeric.loc[0, "sum_rate"])


def test_coerce_metric_frame_requires_all_metrics() -> None:
    with pytest.raises(ValueError, match="Missing metric columns"):
        coerce_metric_frame(pd.DataFrame({"sum_rate": [1.0]}))
