from __future__ import annotations

import runpy
from pathlib import Path

import star_ris_rsma.latency as latency_module
from star_ris_rsma.latency_aggregation import summarize_latency_by_n


def main() -> None:
    # summarize_latency_benchmark imports this symbol by name. Replace it only
    # for the aggregate entrypoint while preserving strict single-bank checks
    # in the original latency module.
    latency_module.summarize_latency = summarize_latency_by_n
    script = Path(__file__).with_name("summarize_latency_benchmark.py")
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
