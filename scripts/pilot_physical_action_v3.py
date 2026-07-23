from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


N_VALUES = (16, 128)
SEEDS = (0, 1, 2)
METRICS = ("sum_rate", "qos_fraction", "all_qos", "violation")


def infer_n(path: Path) -> int:
    match = re.search(r"N(16|128)", path.as_posix())
    if not match:
        raise RuntimeError(f"Cannot infer N from {path}")
    return int(match.group(1))


def load_v3(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    manifests = list(root.rglob("manifest.json"))
    if len(manifests) != len(N_VALUES) * len(SEEDS):
        raise RuntimeError(f"Expected 6 v3 manifests, found {len(manifests)}")
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        if config.get("action_parameterization") != "physical_v3":
            raise RuntimeError(f"Non-v3 manifest found: {path}")

    for path in sorted(root.rglob("test.csv")):
        frame = pd.read_csv(path)
        n_ris = infer_n(path)
        frame["n_ris"] = n_ris
        frame["method"] = "td3_v3_physical"
        frame["source_path"] = path.as_posix()
        frames.append(frame)
    if not frames:
        raise RuntimeError("No v3 test CSV files found")
    result = pd.concat(frames, ignore_index=True)
    for n_ris in N_VALUES:
        for seed in SEEDS:
            group = result[(result.n_ris == n_ris) & (result.seed == seed)]
            if len(group) != 1000 or group.scenario.nunique() != 1000:
                raise RuntimeError(
                    f"Incomplete v3 coverage N={n_ris}, seed={seed}: "
                    f"rows={len(group)}, scenarios={group.scenario.nunique()}"
                )
    if not np.isfinite(result[list(METRICS)].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite v3 metric")
    return result


def load_legacy(root: Path) -> pd.DataFrame:
    path = next(iter(root.rglob("MERGED_RAW_TEST.csv")), None)
    if path is None:
        raise RuntimeError("MERGED_RAW_TEST.csv not found")
    frame = pd.read_csv(path)
    frame = frame[(frame.method == "td3_v2_fixed") & (frame.n_ris.isin(N_VALUES))].copy()
    expected = len(N_VALUES) * 8 * 1000
    if len(frame) != expected:
        raise RuntimeError(f"Legacy TD3 coverage mismatch: {len(frame)} != {expected}")
    frame["method"] = "td3_v2_legacy_action"
    return frame


def load_traditional(root: Path) -> pd.DataFrame:
    path = next(iter(root.rglob("TRADITIONAL_RAW_TEST.csv")), None)
    if path is None:
        raise RuntimeError("TRADITIONAL_RAW_TEST.csv not found")
    frame = pd.read_csv(path)
    frame = frame[frame.n_ris.isin(N_VALUES)].copy()
    expected = 3 * len(N_VALUES) * 1000
    if len(frame) != expected:
        raise RuntimeError(f"Traditional coverage mismatch: {len(frame)} != {expected}")
    return frame


def collapse(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["method", "n_ris", "scenario"], as_index=False)[list(METRICS)].mean()


def paired(v3: pd.DataFrame, other: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n_ris in N_VALUES:
        a = v3[v3.n_ris == n_ris].set_index("scenario")
        for method in sorted(other.method.unique()):
            b = other[(other.method == method) & (other.n_ris == n_ris)].set_index("scenario")
            common = a.index.intersection(b.index)
            if len(common) != 1000:
                raise RuntimeError(f"Paired coverage mismatch for {method}, N={n_ris}")
            for metric in METRICS:
                delta = a.loc[common, metric].to_numpy(dtype=float) - b.loc[common, metric].to_numpy(dtype=float)
                sem = stats.sem(delta)
                ci = stats.t.interval(0.95, len(delta) - 1, loc=float(delta.mean()), scale=float(sem))
                t_result = stats.ttest_1samp(delta, 0.0)
                if np.allclose(delta, 0.0, atol=1e-15, rtol=0.0):
                    wilcoxon_p = 1.0
                else:
                    wilcoxon_p = float(stats.wilcoxon(delta).pvalue)
                rows.append({
                    "baseline": method,
                    "n_ris": n_ris,
                    "metric": metric,
                    "mean_delta_v3_minus_baseline": float(delta.mean()),
                    "ci95_low": float(ci[0]),
                    "ci95_high": float(ci[1]),
                    "paired_t_p": float(t_result.pvalue),
                    "wilcoxon_p": wilcoxon_p,
                    "cohen_dz": float(delta.mean() / delta.std(ddof=1)) if delta.std(ddof=1) > 0 else 0.0,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--traditional-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    v3_raw = load_v3(args.v3_root)
    legacy_raw = load_legacy(args.legacy_root)
    traditional_raw = load_traditional(args.traditional_root)
    v3 = collapse(v3_raw)
    comparison = collapse(pd.concat([legacy_raw, traditional_raw], ignore_index=True))
    all_collapsed = pd.concat([v3, comparison], ignore_index=True)

    summary = (
        all_collapsed.groupby(["method", "n_ris"], as_index=False)[list(METRICS)]
        .mean()
        .sort_values(["n_ris", "method"])
    )
    effects = pd.DataFrame(paired(v3, comparison))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    v3_raw.to_csv(args.output_dir / "V3_RAW_TEST.csv", index=False)
    summary.to_csv(args.output_dir / "V3_PILOT_SUMMARY.csv", index=False)
    effects.to_csv(args.output_dir / "V3_PAIRED_PILOT.csv", index=False)
    domain = {
        "legacy_actor_output": "[-1, 1]",
        "legacy_max_single_power_share_k4": float(np.exp(1) / (np.exp(1) + 4 * np.exp(-1))),
        "legacy_beta_range": [float(1 / (1 + np.exp(1))), float(1 / (1 + np.exp(-1)))],
        "legacy_phase_range_radians": [-float(np.pi * np.tanh(1)), float(np.pi * np.tanh(1))],
        "physical_v3_power_domain": "full 5-stream simplex including vertices",
        "physical_v3_common_domain": "full 4-user simplex including vertices",
        "physical_v3_beta_range": [0.0, 1.0],
        "physical_v3_phase_range_radians": [-float(np.pi), float(np.pi)],
    }
    (args.output_dir / "ACTION_DOMAIN_AUDIT.json").write_text(json.dumps(domain, indent=2), encoding="utf-8")

    lines = [
        "# TD3 physical-action v3 pilot",
        "",
        "This is a 3-seed pilot at N=16 and N=128. It is not final thesis evidence.",
        "The purpose is to test whether removing the legacy action-domain restriction closes the solver gap.",
        "",
        "## Scenario-mean performance",
        "",
        "| Method | N | Sum-rate | QoS fraction | All-QoS | Violation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.method} | {int(row.n_ris)} | {row.sum_rate:.5f} | "
            f"{row.qos_fraction:.5f} | {row.all_qos:.5f} | {row.violation:.5f} |"
        )
    lines.extend([
        "",
        "## Key paired deltas: v3 minus baseline",
        "",
        "| Baseline | N | Metric | Mean delta | 95% CI |",
        "|---|---:|---|---:|---|",
    ])
    for row in effects.itertuples(index=False):
        if row.metric in {"sum_rate", "qos_fraction", "all_qos", "violation"}:
            lines.append(
                f"| {row.baseline} | {int(row.n_ris)} | {row.metric} | "
                f"{row.mean_delta_v3_minus_baseline:.5f} | "
                f"[{row.ci95_low:.5f}, {row.ci95_high:.5f}] |"
            )
    (args.output_dir / "V3_PILOT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.output_dir / "V3_PILOT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
