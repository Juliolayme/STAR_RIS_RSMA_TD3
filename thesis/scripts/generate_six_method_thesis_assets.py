from __future__ import annotations

"""Generate thesis-ready LaTeX tables from the audited six-method bundle.

The script refuses to run unless the published audit verdict is PASS. It does
not change or recompute scientific results; it only formats the canonical CSV
files for the thesis.
"""

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


def decimal_comma(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


def compact_decimal(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0,0000"
    if abs(value) < 1e-5:
        return f"{value:.2e}".replace(".", ",").replace("e", r"\times 10^{") + "}"
    return decimal_comma(value, 4)


def latency_decimal(value: float) -> str:
    return decimal_comma(value, 3)


def tex_header(caption: str, label: str, columns: str, headings: str) -> list[str]:
    return [
        r"\begin{landscape}",
        rf"\begin{{longtable}}{{{columns}}}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}\\",
        r"  \toprule",
        f"  {headings} \\",
        r"  \midrule",
        r"  \endfirsthead",
        rf"  \multicolumn{{{len(columns.split())}}}{{c}}{{\tablename\ \thetable\ -- tiếp theo}}\\",
        r"  \toprule",
        f"  {headings} \\",
        r"  \midrule",
        r"  \endhead",
        r"  \bottomrule",
        r"  \endlastfoot",
    ]


def write_performance(frame: pd.DataFrame, output: Path) -> None:
    frame = frame.copy()
    frame["method_order"] = frame["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    frame = frame.sort_values(["n_ris", "method_order"])
    lines = [
        "% Generated from TABLE_SIX_METHOD_PERFORMANCE.csv. Do not edit manually.",
        r"\begin{landscape}",
        r"\begin{longtable}{r l r r r r r}",
        r"  \caption{Hiệu năng của sáu phương pháp trên tập kiểm thử khóa.}",
        r"  \label{tab:six-method-performance}\\",
        r"  \toprule",
        r"  $N$ & \textbf{Phương pháp} & \textbf{Số seed} & \textbf{Tổng tốc độ} & \textbf{Tỷ lệ QoS} & \textbf{Toàn bộ UE đạt QoS} & \textbf{Mức vi phạm} \\",
        r"  \midrule",
        r"  \endfirsthead",
        r"  \multicolumn{7}{c}{\tablename\ \thetable\ -- tiếp theo}\\",
        r"  \toprule",
        r"  $N$ & \textbf{Phương pháp} & \textbf{Số seed} & \textbf{Tổng tốc độ} & \textbf{Tỷ lệ QoS} & \textbf{Toàn bộ UE đạt QoS} & \textbf{Mức vi phạm} \\",
        r"  \midrule",
        r"  \endhead",
        r"  \bottomrule",
        r"  \endlastfoot",
    ]
    last_n: int | None = None
    for row in frame.itertuples(index=False):
        n_ris = int(row.n_ris)
        if last_n is not None and n_ris != last_n:
            lines.append(r"  \addlinespace")
        lines.append(
            "  "
            + " & ".join(
                [
                    str(n_ris),
                    METHOD_LABELS[str(row.method)],
                    str(int(row.seeds)),
                    decimal_comma(row.sum_rate_mean),
                    decimal_comma(row.qos_fraction_mean),
                    decimal_comma(row.all_qos_mean),
                    compact_decimal(row.violation_mean),
                ]
            )
            + r" \\" 
        )
        last_n = n_ris
    lines.extend(
        [
            r"\end{longtable}",
            r"\end{landscape}",
            "",
            r"\noindent\textit{Cách đọc bảng:} Đối với TD3, DDPG và PPO, các giá trị là trung bình của tám kết quả theo seed, trong đó mỗi seed được đánh giá trên 1.000 kịch bản kiểm thử khóa. Đối với các phương pháp truyền thống, các giá trị được tổng hợp trực tiếp trên cùng 1.000 kịch bản. Vì đơn vị bất định khác nhau, không nên so sánh trực tiếp độ rộng khoảng tin cậy của hai nhóm như thể chúng có cùng nguồn biến thiên.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latency(frame: pd.DataFrame, output: Path) -> None:
    frame = frame.copy()
    frame["method_order"] = frame["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    frame = frame.sort_values(["n_ris", "method_order"])
    lines = [
        "% Generated from TABLE_SIX_METHOD_CPU_LATENCY.csv. Do not edit manually.",
        r"\begin{landscape}",
        r"\begin{longtable}{r l r r r r r}",
        r"  \caption{Độ trễ ra quyết định của sáu phương pháp trên cùng một CPU runner.}",
        r"  \label{tab:six-method-latency}\\",
        r"  \toprule",
        r"  $N$ & \textbf{Phương pháp} & \textbf{Số mẫu} & \textbf{Trung bình (ms)} & \textbf{Độ lệch chuẩn} & \textbf{Trung vị (ms)} & \textbf{Khoảng min--max (ms)} \\",
        r"  \midrule",
        r"  \endfirsthead",
        r"  \multicolumn{7}{c}{\tablename\ \thetable\ -- tiếp theo}\\",
        r"  \toprule",
        r"  $N$ & \textbf{Phương pháp} & \textbf{Số mẫu} & \textbf{Trung bình (ms)} & \textbf{Độ lệch chuẩn} & \textbf{Trung vị (ms)} & \textbf{Khoảng min--max (ms)} \\",
        r"  \midrule",
        r"  \endhead",
        r"  \bottomrule",
        r"  \endlastfoot",
    ]
    last_n: int | None = None
    for row in frame.itertuples(index=False):
        n_ris = int(row.n_ris)
        if last_n is not None and n_ris != last_n:
            lines.append(r"  \addlinespace")
        span = f"{latency_decimal(row.min)}--{latency_decimal(row.max)}"
        lines.append(
            "  "
            + " & ".join(
                [
                    str(n_ris),
                    METHOD_LABELS[str(row.method)],
                    str(int(row.count)),
                    latency_decimal(row.solve_ms_mean),
                    latency_decimal(row.solve_ms_std),
                    latency_decimal(row.solve_ms_median),
                    span,
                ]
            )
            + r" \\" 
        )
        last_n = n_ris
    lines.extend(
        [
            r"\end{longtable}",
            r"\end{landscape}",
            "",
            r"\noindent\textit{Lưu ý:} Các số đo được thực hiện đơn luồng trên cùng một CPU runner. Tỷ lệ tăng tốc chỉ có ý nghĩa trong nền tảng đã công bố và phải được diễn giải cùng chất lượng nghiệm và QoS.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_td3_tests(frame: pd.DataFrame, output: Path) -> None:
    subset = frame[(frame["method_a"] == "td3") | (frame["method_b"] == "td3")].copy()
    rows: list[dict[str, object]] = []
    for item in subset.to_dict("records"):
        if item["method_a"] == "td3":
            comparator = str(item["method_b"])
            difference = float(item["mean_difference_a_minus_b"])
            effect = float(item["cohen_dz"])
        else:
            comparator = str(item["method_a"])
            difference = -float(item["mean_difference_a_minus_b"])
            effect = -float(item["cohen_dz"])
        rows.append(
            {
                "n_ris": int(item["n_ris"]),
                "comparator": comparator,
                "difference": difference,
                "effect": effect,
                "t_sig": bool(item["paired_t_holm_significant_0_05"]),
                "w_sig": bool(item["wilcoxon_holm_significant_0_05"]),
            }
        )
    order = {m: i for i, m in enumerate(["ddpg", "ppo", "ao_sca", "ao_grid", "analytical_ris"])}
    rows.sort(key=lambda item: (int(item["n_ris"]), order[str(item["comparator"])]))
    lines = [
        "% Generated from TABLE_SIX_METHOD_PAIRED_TESTS_HOLM.csv. Do not edit manually.",
        r"\begin{landscape}",
        r"\begin{longtable}{r l r r c c}",
        r"  \caption{So sánh ghép cặp tổng tốc độ giữa TD3 và năm phương pháp còn lại sau hiệu chỉnh Holm.}",
        r"  \label{tab:td3-paired-tests-holm}\\",
        r"  \toprule",
        r"  $N$ & \textbf{Phương pháp đối chiếu} & $\Delta=\overline{R}_{\mathrm{TD3}}-\overline{R}_{b}$ & \textbf{Cohen's $d_z$} & \textbf{$t$-Holm} & \textbf{Wilcoxon-Holm} \\",
        r"  \midrule",
        r"  \endfirsthead",
        r"  \multicolumn{6}{c}{\tablename\ \thetable\ -- tiếp theo}\\",
        r"  \toprule",
        r"  $N$ & \textbf{Phương pháp đối chiếu} & $\Delta=\overline{R}_{\mathrm{TD3}}-\overline{R}_{b}$ & \textbf{Cohen's $d_z$} & \textbf{$t$-Holm} & \textbf{Wilcoxon-Holm} \\",
        r"  \midrule",
        r"  \endhead",
        r"  \bottomrule",
        r"  \endlastfoot",
    ]
    last_n: int | None = None
    for item in rows:
        n_ris = int(item["n_ris"])
        if last_n is not None and n_ris != last_n:
            lines.append(r"  \addlinespace")
        lines.append(
            "  "
            + " & ".join(
                [
                    str(n_ris),
                    METHOD_LABELS[str(item["comparator"])],
                    decimal_comma(float(item["difference"])),
                    decimal_comma(float(item["effect"])),
                    "Có" if item["t_sig"] else "Không",
                    "Có" if item["w_sig"] else "Không",
                ]
            )
            + r" \\" 
        )
        last_n = n_ris
    lines.extend(
        [
            r"\end{longtable}",
            r"\end{landscape}",
            "",
            r"\noindent Trong bảng, ``Có'' nghĩa là bác bỏ giả thuyết không ở mức ý nghĩa $\alpha=0{,}05$ sau hiệu chỉnh Holm. Dấu âm khi so với AO-SCA cho thấy AO-SCA có tổng tốc độ trung bình cao hơn TD3; dấu dương trong các so sánh còn lại cho thấy TD3 có tổng tốc độ trung bình cao hơn phương pháp đối chiếu.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    audit_path = args.results / "SIX_METHOD_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("verdict") != "PASS":
        raise SystemExit(f"Refusing to generate thesis assets: audit verdict={audit.get('verdict')!r}")
    if not audit.get("shared_scenario_banks", False):
        raise SystemExit("Refusing to generate thesis assets: ScenarioBanks are not shared")

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
        "source_tables": {
            "performance": str(table_dir / "TABLE_SIX_METHOD_PERFORMANCE.csv"),
            "latency": str(table_dir / "TABLE_SIX_METHOD_CPU_LATENCY.csv"),
            "paired_tests": str(table_dir / "TABLE_SIX_METHOD_PAIRED_TESTS_HOLM.csv"),
        },
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
