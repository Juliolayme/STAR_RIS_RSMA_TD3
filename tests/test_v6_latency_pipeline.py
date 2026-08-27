from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_physical_v6_latency.py"
SPEC = importlib.util.spec_from_file_location("summarize_physical_v6_latency", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_latency() -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    for n_ris in MODULE.N_VALUES:
        bank = f"bank-{n_ris}"
        for method in MODULE.METHODS:
            learned = method in MODULE.LEARNED
            checkpoint_sha = f"sha-{method}-{n_ris}" if learned else "not_applicable"
            checkpoint_step = 100000 if learned else -1
            if learned:
                checkpoints.append(
                    {
                        "method": method,
                        "n_ris": n_ris,
                        "checkpoint_sha256": checkpoint_sha,
                        "checkpoint_step": checkpoint_step,
                    }
                )
            for scenario in range(100):
                rows.append(
                    {
                        "method": method,
                        "n_ris": n_ris,
                        "scenario": scenario,
                        "source_scenario": scenario + 10,
                        "inference_ms": 0.1 if learned else np.nan,
                        "evaluation_ms": 0.2 if learned else np.nan,
                        "solve_ms": 1.0 + scenario / 1000.0,
                        "cpu_threads": 1,
                        "seed": 0,
                        "bank_checksum": bank,
                        "latency_protocol": MODULE.PROTOCOL,
                        "checkpoint_step": checkpoint_step,
                        "checkpoint_sha256": checkpoint_sha,
                        "repository_commit": "commit",
                        "runner_os": "runner",
                        "cpu_model": "cpu",
                        "python_version": "3.11",
                        "torch_version": "2.10",
                    }
                )
    return pd.DataFrame(rows), {"audit": "PASS", "checkpoints": checkpoints}


def test_v6_latency_validation_accepts_complete_matched_protocol() -> None:
    raw, index = synthetic_latency()

    MODULE.validate(raw, index)
    summary = MODULE.summarize(raw)

    assert len(raw) == 3000
    assert len(summary) == 30
    assert set(summary["count"]) == {100}


def test_v6_latency_validation_rejects_cross_method_bank_mismatch() -> None:
    raw, index = synthetic_latency()
    mask = (raw.method == "td3") & (raw.n_ris == 128)
    raw.loc[mask, "bank_checksum"] = "wrong-bank"

    with pytest.raises(RuntimeError, match="do not share ScenarioBank"):
        MODULE.validate(raw, index)


def test_v6_latency_workflow_uses_frozen_banks_and_audited_artifacts() -> None:
    text = (ROOT / ".github" / "workflows" / "finalize-physical-v6-latency.yml").read_text(
        encoding="utf-8"
    )

    assert "run-id: 33053693666" in text
    assert text.count("run-id: 33061462093") == 2
    assert "run-id: 30422028560" in text
    assert '--bank "$bank"' in text
    assert "--warmup 10 --count 100" in text
    assert "PHYSICAL_V6_SIX_METHOD_LATENCY" in text
