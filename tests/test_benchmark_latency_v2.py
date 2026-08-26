from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_latency_v2.py"
SPEC = importlib.util.spec_from_file_location("benchmark_latency_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_checkpoint_tolerance_accepts_cross_runner_float_drift() -> None:
    observed = 4.437100985241393
    canonical = 4.437086501005909

    assert np.isclose(
        observed,
        canonical,
        rtol=MODULE.CHECKPOINT_VERIFY_RTOL,
        atol=MODULE.CHECKPOINT_VERIFY_ATOL,
    )


def test_checkpoint_tolerance_rejects_material_metric_drift() -> None:
    assert not np.isclose(
        4.438,
        4.437086501005909,
        rtol=MODULE.CHECKPOINT_VERIFY_RTOL,
        atol=MODULE.CHECKPOINT_VERIFY_ATOL,
    )
