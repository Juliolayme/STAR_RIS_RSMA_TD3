from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--inputs", nargs="+", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()

paths: list[str] = []
for pattern in a.inputs:
    matches = sorted(glob.glob(pattern, recursive=True))
    paths.extend(matches or [pattern])
paths = list(dict.fromkeys(paths))
if not paths:
    raise SystemExit("No input CSV files matched")
frames = [pd.read_csv(path) for path in paths]
df = pd.concat(frames, ignore_index=True)
keys = [key for key in ["method", "seed", "scenario", "step", "eval_step"] if key in df.columns]
if keys:
    duplicated = df.duplicated(keys, keep=False)
    if duplicated.any():
        raise SystemExit(
            f"Duplicate result keys detected: {df.loc[duplicated, keys].head().to_dict('records')}"
        )
    df = df.sort_values(keys)
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.output, index=False)
print(df.groupby("method")["sum_rate"].agg(["mean", "std", "count"]) if "sum_rate" in df else df.tail())
