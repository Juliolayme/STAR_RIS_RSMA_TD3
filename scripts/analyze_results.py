from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from star_ris_rsma.statistics import write_analysis

p = argparse.ArgumentParser()
p.add_argument("--inputs", nargs="+", required=True)
p.add_argument("--metric", default="sum_rate")
p.add_argument("--output-dir", required=True)
a = p.parse_args()
df = pd.concat([pd.read_csv(path) for path in a.inputs], ignore_index=True)
summary, comparisons = write_analysis(df, Path(a.output_dir), a.metric)
print(f"Wrote {summary} and {comparisons}")
