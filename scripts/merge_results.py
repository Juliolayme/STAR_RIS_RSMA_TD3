from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--inputs", nargs="+", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
frames = [pd.read_csv(x) for x in a.inputs]
df = pd.concat(frames, ignore_index=True)
keys = [k for k in ["method", "seed", "scenario", "step"] if k in df.columns]
if keys:
    duplicated = df.duplicated(keys, keep=False)
    if duplicated.any():
        raise SystemExit(f"Duplicate result keys detected: {df.loc[duplicated, keys].head().to_dict('records')}")
df = df.sort_values(keys) if keys else df
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.output, index=False)
print(df.groupby("method")["sum_rate"].agg(["mean", "std", "count"]) if "sum_rate" in df else df.tail())
