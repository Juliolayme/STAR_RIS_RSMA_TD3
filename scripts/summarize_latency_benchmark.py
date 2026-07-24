from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from star_ris_rsma.latency import summarize_latency, validate_latency_frame
from star_ris_rsma.result_validation import CORE_METRICS


N_VALUES = (16, 32, 64, 96, 128)
METHODS = ("td3", "ao_sca", "ao_grid", "analytical_ris")
EXPECTED_SCENARIOS = 1000


def read_single(root: Path, name: str) -> Path:
    paths = sorted(root.rglob(name))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {name} below {root}, found {len(paths)}")
    return paths[0]


def load_latency(root: Path) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    raw_paths = sorted(root.rglob("LATENCY_RAW.csv"))
    metadata_paths = sorted(root.rglob("LATENCY_METADATA.json"))
    manifest_paths = sorted(
        path for path in root.rglob("manifest.json") if path.parent.name == "results"
    )
    if len(raw_paths) != len(N_VALUES):
        raise RuntimeError(f"Expected {len(N_VALUES)} latency CSV files, found {len(raw_paths)}")
    if len(metadata_paths) != len(N_VALUES):
        raise RuntimeError(f"Expected {len(N_VALUES)} latency metadata files, found {len(metadata_paths)}")
    if len(manifest_paths) != len(N_VALUES):
        raise RuntimeError(f"Expected {len(N_VALUES)} result manifests, found {len(manifest_paths)}")

    frames = []
    for path in raw_paths:
        frame = pd.read_csv(path)
        validate_latency_frame(frame)
        frames.append(frame)
    metadata = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
    return pd.concat(frames, ignore_index=True), metadata, manifests


def validate_protocol(
    raw: pd.DataFrame,
    metadata: list[dict[str, object]],
    manifests: list[dict[str, object]],
    final_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    errors: list[str] = []
    final_raw = pd.read_csv(read_single(final_root, "V3_MERGED_RAW_TEST.csv"))
    final_ci = pd.read_csv(read_single(final_root, "V3_CI95_SUMMARY.csv"))
    final_protocol = json.loads(read_single(final_root, "V3_PROTOCOL.json").read_text(encoding="utf-8"))

    if tuple(int(x) for x in final_protocol.get("n_values", [])) != N_VALUES:
        errors.append("Final quality artifact has unexpected N values")
    if tuple(int(x) for x in final_protocol.get("seeds", [])) != tuple(range(8)):
        errors.append("Final quality artifact does not contain seeds 0-7")
    if int(final_protocol.get("test_scenarios_per_n", -1)) != EXPECTED_SCENARIOS:
        errors.append("Final quality artifact has unexpected scenario count")

    if set(pd.to_numeric(raw["n_ris"], errors="raise").astype(int)) != set(N_VALUES):
        errors.append("Latency output does not cover all N values")
    for n_ris in N_VALUES:
        group = raw[pd.to_numeric(raw["n_ris"], errors="raise").astype(int) == n_ris]
        scenarios = sorted(pd.to_numeric(group["scenario"], errors="raise").astype(int).unique())
        if len(group) != EXPECTED_SCENARIOS or scenarios != list(range(EXPECTED_SCENARIOS)):
            errors.append(f"N={n_ris}: incomplete latency scenarios")
        if set(pd.to_numeric(group["seed"], errors="raise").astype(int)) != {0}:
            errors.append(f"N={n_ris}: latency must use fixed seed 0 checkpoint")
        bench_checks = group["bank_checksum"].astype(str).unique()
        quality_checks = final_raw[
            pd.to_numeric(final_raw["n_ris"], errors="raise").astype(int) == n_ris
        ]["bank_checksum"].astype(str).unique()
        if len(bench_checks) != 1 or len(quality_checks) != 1 or bench_checks[0] != quality_checks[0]:
            errors.append(f"N={n_ris}: locked test-bank checksum mismatch")

    metadata_by_n = {int(item["n_ris"]): item for item in metadata}
    if set(metadata_by_n) != set(N_VALUES):
        errors.append("Latency metadata does not cover all N values exactly once")
    for n_ris, item in metadata_by_n.items():
        if int(item.get("torch_num_threads", -1)) != 1:
            errors.append(f"N={n_ris}: PyTorch was not single-threaded")
        if item.get("timing_boundary") != "ready locked channel -> returned decision/result":
            errors.append(f"N={n_ris}: unexpected timing boundary")
        if int(item.get("scenarios", -1)) != EXPECTED_SCENARIOS:
            errors.append(f"N={n_ris}: metadata scenario count mismatch")

    manifest_by_n: dict[int, dict[str, object]] = {}
    for payload in manifests:
        config = payload.get("config", {}) or {}
        n_ris = int(config.get("n_ris", -1))
        if n_ris in manifest_by_n:
            errors.append(f"N={n_ris}: duplicate timing checkpoint manifest")
        manifest_by_n[n_ris] = payload
        if int(payload.get("seed", -1)) != 0:
            errors.append(f"N={n_ris}: timing checkpoint is not seed 0")
        if config.get("action_parameterization") != "physical_v3":
            errors.append(f"N={n_ris}: timing checkpoint is not physical_v3")
        if config.get("qos_dual_enabled") is not True:
            errors.append(f"N={n_ris}: adaptive QoS dual is not enabled")
        if payload.get("training_protocol") != "td3_qos_scalability_v3_constrained":
            errors.append(f"N={n_ris}: unexpected training protocol")
    if set(manifest_by_n) != set(N_VALUES):
        errors.append("Timing checkpoint manifests do not cover all N values")

    latency_values = raw[[column for column in raw.columns if column.endswith("_ms")]]
    if not np.isfinite(latency_values.to_numpy(dtype=np.float64)).all():
        errors.append("Non-finite latency detected")

    if errors:
        raise RuntimeError("Latency benchmark validation failed:\n" + "\n".join(errors))
    return final_raw, final_ci, final_protocol


def build_tradeoff(raw: pd.DataFrame, final_ci: pd.DataFrame) -> pd.DataFrame:
    latency = summarize_latency(raw).set_index("n_ris")
    rows: list[dict[str, object]] = []
    for n_ris in N_VALUES:
        group = raw[pd.to_numeric(raw["n_ris"], errors="raise").astype(int) == n_ris]
        ci_n = final_ci[pd.to_numeric(final_ci["n_ris"], errors="raise").astype(int) == n_ris].set_index("metric")
        row: dict[str, object] = {
            "n_ris": n_ris,
            "td3_sum_rate_final_8seed_mean": float(ci_n.loc["sum_rate", "mean"]),
            "td3_sum_rate_ci95_low": float(ci_n.loc["sum_rate", "ci95_low"]),
            "td3_sum_rate_ci95_high": float(ci_n.loc["sum_rate", "ci95_high"]),
            "td3_qos_fraction_final_8seed_mean": float(ci_n.loc["qos_fraction", "mean"]),
            "td3_all_qos_final_8seed_mean": float(ci_n.loc["all_qos", "mean"]),
            "td3_violation_final_8seed_mean": float(ci_n.loc["violation", "mean"]),
        }
        for method in ("ao_sca", "ao_grid", "analytical_ris"):
            for metric in CORE_METRICS:
                row[f"{method}_{metric}_mean"] = float(
                    pd.to_numeric(group[f"{method}_{metric}"], errors="raise").mean()
                )
        for column, value in latency.loc[n_ris].items():
            if column not in {"seed", "scenarios", "bank_checksum"}:
                row[column] = value
        ao_sum_rate = float(row["ao_sca_sum_rate_mean"])
        row["td3_percent_of_ao_sca_sum_rate"] = (
            100.0 * float(row["td3_sum_rate_final_8seed_mean"]) / ao_sum_rate
        )
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(value: object, digits: int = 3) -> str:
    number = float(value)
    return f"{number:.{digits}f}"


def write_report(tradeoff: pd.DataFrame, metadata: list[dict[str, object]], output: Path) -> None:
    lines = [
        "# Fair CPU inference-latency benchmark",
        "",
        "- Same hosted CPU process, one PyTorch/BLAS thread, and the same locked channel for all methods within each N.",
        "- Timed boundary starts after the channel is ready; channel generation and environment reset are excluded.",
        "- TD3 end-to-end time includes actor inference, physical action decoding, and one final metric evaluation.",
        "- AO-SCA, AO-Grid, and AnalyticalRIS times include the complete solver call and returned metrics.",
        "- Latency uses one fixed seed-0 checkpoint per N; final TD3 quality uses the retained eight-seed mean and 95% CI.",
        "- Training/offline learning time is excluded. AO-SCA is local, not a global optimum or upper bound.",
        "",
        "## End-to-end latency",
        "",
        "| N | TD3 mean / p95 ms | AO-SCA mean / p95 ms | AO-SCA ÷ TD3 | AO-Grid mean / p95 ms | AO-Grid ÷ TD3 | AnalyticalRIS mean / p95 ms | AnalyticalRIS ÷ TD3 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tradeoff.itertuples(index=False):
        lines.append(
            f"| {int(row.n_ris)} | {fmt(row.td3_end_to_end_ms_mean)} / {fmt(row.td3_end_to_end_ms_p95)} | "
            f"{fmt(row.ao_sca_end_to_end_ms_mean)} / {fmt(row.ao_sca_end_to_end_ms_p95)} | "
            f"{fmt(row.ao_sca_over_td3_ratio_of_means, 1)}× | "
            f"{fmt(row.ao_grid_end_to_end_ms_mean)} / {fmt(row.ao_grid_end_to_end_ms_p95)} | "
            f"{fmt(row.ao_grid_over_td3_ratio_of_means, 1)}× | "
            f"{fmt(row.analytical_ris_end_to_end_ms_mean)} / {fmt(row.analytical_ris_end_to_end_ms_p95)} | "
            f"{fmt(row.analytical_ris_over_td3_ratio_of_means, 2)}× |"
        )

    lines.extend([
        "",
        "## Quality-latency trade-off",
        "",
        "| N | TD3 sum-rate, eight-seed mean [95% CI] | AO-SCA | TD3 / AO-SCA | AO-Grid | AnalyticalRIS | TD3 QoS fraction | TD3 All-QoS | TD3 violation |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in tradeoff.itertuples(index=False):
        lines.append(
            f"| {int(row.n_ris)} | {fmt(row.td3_sum_rate_final_8seed_mean)} "
            f"[{fmt(row.td3_sum_rate_ci95_low)}, {fmt(row.td3_sum_rate_ci95_high)}] | "
            f"{fmt(row.ao_sca_sum_rate_mean)} | {fmt(row.td3_percent_of_ao_sca_sum_rate, 1)}% | "
            f"{fmt(row.ao_grid_sum_rate_mean)} | {fmt(row.analytical_ris_sum_rate_mean)} | "
            f"{fmt(row.td3_qos_fraction_final_8seed_mean, 5)} | "
            f"{fmt(row.td3_all_qos_final_8seed_mean, 5)} | "
            f"{fmt(row.td3_violation_final_8seed_mean, 5)} |"
        )

    faster_sca = bool((tradeoff["ao_sca_over_td3_ratio_of_means"] > 1.0).all())
    faster_grid = bool((tradeoff["ao_grid_over_td3_ratio_of_means"] > 1.0).all())
    analytical_ratios = tradeoff["analytical_ris_over_td3_ratio_of_means"].to_numpy(dtype=float)
    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        f"- TD3 faster than AO-SCA at every N: **{faster_sca}**.",
        f"- TD3 faster than AO-Grid at every N: **{faster_grid}**.",
        f"- AnalyticalRIS / TD3 latency ratio range: **{analytical_ratios.min():.3f}× to {analytical_ratios.max():.3f}×**. Do not claim TD3 is the fastest method if this ratio is below one.",
        "- The defensible claim is an online-computation trade-off: TD3 amortizes optimization into a neural forward pass, substantially reducing latency relative to iterative AO while retaining high QoS and useful sum-rate.",
        "- AnalyticalRIS is a low-complexity heuristic; its latency must be interpreted together with its much lower sum-rate and QoS performance.",
        "- CPU models may differ across N matrix jobs. Speed ratios are valid within each N because all methods are measured sequentially in the same process and on the same pinned CPU.",
        "",
        "## Runner metadata",
        "",
    ])
    for item in sorted(metadata, key=lambda value: int(value["n_ris"])):
        lines.append(
            f"- N={int(item['n_ris'])}: {item.get('processor', 'unknown')}; "
            f"affinity={item.get('affinity', [])}; torch threads={item.get('torch_num_threads')}"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency-root", type=Path, required=True)
    parser.add_argument("--final-statistics-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw, metadata, manifests = load_latency(args.latency_root)
    _, final_ci, final_protocol = validate_protocol(
        raw, metadata, manifests, args.final_statistics_root
    )
    summary = summarize_latency(raw)
    tradeoff = build_tradeoff(raw, final_ci)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.sort_values(["n_ris", "scenario"]).to_csv(
        args.output_dir / "LATENCY_RAW_ALL.csv", index=False
    )
    summary.to_csv(args.output_dir / "LATENCY_SUMMARY.csv", index=False)
    tradeoff.to_csv(args.output_dir / "LATENCY_TRADEOFF.csv", index=False)
    (args.output_dir / "LATENCY_METADATA_ALL.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "LATENCY_PROTOCOL.json").write_text(
        json.dumps({
            "n_values": N_VALUES,
            "timing_checkpoint_seed": 0,
            "scenarios_per_n": EXPECTED_SCENARIOS,
            "methods": METHODS,
            "quality_source": "retained final TD3 v3 eight-seed artifact",
            "quality_protocol": final_protocol,
            "latency_unit": "wall-clock milliseconds per ready-channel decision/result",
            "latency_statistics": ["mean", "median", "p95", "p99"],
            "speedup_definition": "solver end-to-end latency divided by TD3 end-to-end latency",
            "training_time_included": False,
            "ao_sca_claim_boundary": "local method; not global optimum or upper bound",
        }, indent=2),
        encoding="utf-8",
    )
    write_report(tradeoff, metadata, args.output_dir / "LATENCY_REPORT.md")
    print((args.output_dir / "LATENCY_REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
