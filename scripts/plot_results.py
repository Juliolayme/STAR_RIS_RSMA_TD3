from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from star_ris_rsma.statistics import summarize

p = argparse.ArgumentParser()
p.add_argument("--inputs", nargs="+", required=True)
p.add_argument("--metric", default="sum_rate")
p.add_argument("--output", required=True)
a = p.parse_args()
df = pd.concat([pd.read_csv(path) for path in a.inputs], ignore_index=True)
summary = summarize(df, a.metric).sort_values("mean")
errors = [summary["mean"] - summary["ci95_low"], summary["ci95_high"] - summary["mean"]]
fig, ax = plt.subplots(figsize=(8, max(4, 0.55 * len(summary))))
ax.barh(summary["method"], summary["mean"], xerr=errors, capsize=3)
ax.set_xlabel(f"{a.metric} (mean and 95% CI)")
ax.set_ylabel("Method")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.output, dpi=200)
