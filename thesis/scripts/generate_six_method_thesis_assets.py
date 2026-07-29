from __future__ import annotations

"""Generate thesis-ready LaTeX tables from the audited six-method bundle."""

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "results" / "six_method_v1"
DEFAULT_OUTPUT = REPO_ROOT / "thesis" / "generated"
METHOD_LABELS = {
    "td3": "TD3",
    "ddpg": "DDPG",
    "ppo": "PPO",
    "ao_sca": "AO-SCA",
    "ao_grid": "AO-Grid",
    "analytical_ris": "AnalyticalRIS",
}
METHOD_ORDER = ["td3", "ddpg", "ppo", "ao_sca", "ao_grid", "analytical_ris"]
ROW_END = r" \\\\"


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


def fmt_violation(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0,0000"
    if abs(value) < 1e-5:
        mantissa, exponent = f"{value:.2e}".split("e")
        exponent_int = int(exponent)
        return rf"${mantissa.replace('.', ',')}\times 10^{{{exponent_int}}}$"
    return fmt(value, 4)


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    order = {name: index for index, name in enumerate(METHOD_ORDER)}
    result["_order"] = result["method"].map(order)
    return result.sort_values(["n_ris", "_order"])


def write_performance(frame: pd.DataFrame, output: Path) -> None:
    lines = [
        "% Generated from TABLE_SIX_METHOD_PERFORMANCE.csv. Do not edit manually.",
        r"\begin{landscape}",
        r"\begin{longtable}{r l r r r r r}",
        r"\caption{Hiệu năng của sáu phương pháp trên tập kiểm thử khóa.}",
        r"\label{tab:six-method-performance}\\",
        r"\toprule",
        r"$N$ & \textbf{Phương pháp} & \textbf{Số seed} & \textbf{Tổng tốc độ} & \textbf{Tỷ lệ QoS} & \textbf{Toàn bộ UE đạt QoS} & \textbf{Mức vi phạm} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{7}{c}{\tablename\ \thetable\ -- tiếp theo}\\",
        r"\toprule",
        r"$N$ & \textbf{Phương pháp} & \textbf{Số seed} & \textbf{Tổng tốc độ} & \textbf{Tỷ lệ QoS} & \textbf{Toàn bộ UE đạt QoS} & \textbf{Mức vi phạm} \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    previous_n = None
    for row in ordered(frame).itertuples(index=False):
        n_ris = int(row.n_ris)
        if previous_n is not None and n_ris != previous_n:
            lines.append(r"\addlinespace")
        cells = [
            str(n_ris),
            METHOD_LABELS[str(row.method)],
            str(int(row.seeds)),
            fmt(row.sum_rate_mean),
            fmt(row.qos_fraction_mean),
            fmt(row.all_qos_mean),
            fmt_violation(row.violation_mean),
        ]
        lines.append(" & ".join(cells) + ROW_END)
        previous_n = n_ris
    lines += [r"\end{longtable}", r"\end{landscape}"]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latency(frame: pd.DataFrame, output: Path) -> None:
    lines = [
        "% Generated from TABLE_SIX_METHOD_CPU_LATENCY.csv. Do not edit manually.",
        r"\begin{landscape}",
        r"\begin{longtable}{r l r r r r r}",
        r"\caption{Độ trễ ra quyết định của sáu phương pháp trên cùng một CPU runner.}",
        r"\label{tab:six-method-latency}\\",
        r"\toprule",
        r"$N$ & \textbf{Phương pháp} & \textbf{Số mẫu} & \textbf{Trung bình (ms)} & \textbf{Độ lệch chuẩn} & \textbf{Trung vị (ms)} & \textbf{Khoảng min--max (ms)} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{7}{c}{\tablename\ \thetable\ -- tiếp theo}\\",
        r"\toprule",
        r"$N$ & \textbf{Phương pháp} & \textbf{Số mẫu} & \textbf{Trung bình (ms)} & \textbf{Độ lệch chuẩn} & \textbf{Trung vị (ms)} & \textbf{Khoảng min--max (ms)} \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    previous_n = None
    for row in ordered(frame).itertuples(index=False):
        n_ris = int(row.n_ris)
        if previous_n is not None and n_ris != previous_n:
            lines.append(r"\addlinespace")
        cells = [
            str(n_ris),
            METHOD_LABELS[str(row.method)],
            str(int(row.count)),
            fmt(row.solve_ms_mean, 3),
            fmt(row.solve_ms_std, 3),
            fmt(row.solve_ms_median, 3),
            f"{fmt(row.min, 3)}--{fmt(row.max, 3)}",
        ]
        lines.append(" & ".join(cells) + ROW_END)
        previous_n = n_ris
    lines += [r"\end{longtable}", r"\end{landscape}"]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_td3_tests(frame: pd.DataFrame, output: Path) -> None:
    subset = frame[(frame["method_a"] == "td3") | (frame["method_b"] == "td3")]
    rows: list[dict[str, object]] = []
    for item in subset.to_dict("records"):
        td3_is_a = item["method_a"] == "td3"
        rows.append(
            {
                "n_ris": int(item["n_ris"]),
                "comparator": item["method_b"] if td3_is_a else item["method_a"],
                "difference": float(item["mean_difference_a_minus_b"]) * (1.0 if td3_is_a else -1.0),
                "effect": float(item["cohen_dz"]) * (1.0 if td3_is_a else -1.0),
                "t_sig": bool(item["paired_t_holm_significant_0_05"]),
                "w_sig": bool(item["wilcoxon_holm_significant_0_05"]),
            }
        )
    comparator_order = {name: index for index, name in enumerate(METHOD_ORDER[1:])}
    rows.sort(key=lambda item: (item["n_ris"], comparator_order[item["comparator"]]))
    lines = [
        "% Generated from TABLE_SIX_METHOD_PAIRED_TESTS_HOLM.csv. Do not edit manually.",
        r"\begin{landscape}",
        r"\begin{longtable}{r l r r c c}",
        r"\caption{So sánh ghép cặp tổng tốc độ giữa TD3 và năm phương pháp còn lại sau hiệu chỉnh Holm.}",
        r"\label{tab:td3-paired-tests-holm}\\",
        r"\toprule",
        r"$N$ & \textbf{Phương pháp đối chiếu} & $\Delta=\overline{R}_{\mathrm{TD3}}-\overline{R}_{b}$ & \textbf{Cohen's $d_z$} & \textbf{$t$-Holm} & \textbf{Wilcoxon-Holm} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{6}{c}{\tablename\ \thetable\ -- tiếp theo}\\",
        r"\toprule",
        r"$N$ & \textbf{Phương pháp đối chiếu} & $\Delta=\overline{R}_{\mathrm{TD3}}-\overline{R}_{b}$ & \textbf{Cohen's $d_z$} & \textbf{$t$-Holm} & \textbf{Wilcoxon-Holm} \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    previous_n = None
    for item in rows:
        if previous_n is not None and item["n_ris"] != previous_n:
            lines.append(r"\addlinespace")
        cells = [
            str(item["n_ris"]),
            METHOD_LABELS[str(item["comparator"])],
            fmt(float(item["difference"])),
            fmt(float(item["effect"])),
            "Có" if item["t_sig"] else "Không",
            "Có" if item["w_sig"] else "Không",
        ]
        lines.append(" & ".join(cells) + ROW_END)
        previous_n = item["n_ris"]
    lines += [r"\end{longtable}", r"\end{landscape}"]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    audit = json.loads((args.results / "SIX_METHOD_AUDIT.json").read_text(encoding="utf-8"))
    if audit.get("verdict") != "PASS":
        raise SystemExit(f"Audit verdict is not PASS: {audit.get('verdict')!r}")
    if not audit.get("shared_scenario_banks", False):
        raise SystemExit("ScenarioBank checksums are not shared")

    table_dir = args.results / "tables"
    performance = pd.read_csv(table_dir / "TABLE_SIX_METHOD_PERFORMANCE.csv")
    latency = pd.read_csv(table_dir / "TABLE_SIX_METHOD_CPU_LATENCY.csv")
    paired = pd.read_csv(table_dir / "TABLE_SIX_METHOD_PAIRED_TESTS_HOLM.csv")

    args.output.mkdir(parents=True, exist_ok=True)
    write_performance(performance, args.output / "table_six_method_performance.tex")
    write_latency(latency, args.output / "table_six_method_latency.tex")
    write_td3_tests(paired, args.output / "table_td3_paired_tests_holm.tex")

    manifest = {
        "audit_verdict": audit["verdict"],
        "scientific_commit": audit.get("drl_repository_commit"),
        "source_directory": str(table_dir),
        "generated_files": [
            "table_six_method_performance.tex",
            "table_six_method_latency.tex",
            "table_td3_paired_tests_holm.tex",
        ],
    }
    (args.output / "six_method_assets_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
