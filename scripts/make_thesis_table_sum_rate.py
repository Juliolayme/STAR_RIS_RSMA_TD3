"""Sinh Bảng 4.3 (tổng tốc độ, trung bình ± độ lệch chuẩn) cho luận văn từ CSV đã khóa.

DRL: độ lệch chuẩn giữa 5 trung bình theo hạt giống; AO/AnalyticalRIS: độ lệch chuẩn giữa
1.000 kịch bản kiểm thử. Chạy: python scripts/make_thesis_table_sum_rate.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/physical_v6_full_r2/tables/TABLE_V6_SIX_METHOD_PERFORMANCE.csv"
OUT = ROOT / "thesis_latex/figures/bang_ch4_tong_toc_do.tex"
ORDER = ["td3", "ddpg", "ppo", "ao_sca", "ao_grid", "analytical_ris"]
HEAD = ["TD3", "DDPG", "PPO", "AO--SCA", "AO--Grid", "AnalyticalRIS"]


def vn(x: float, d: int) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def main() -> None:
    t = pd.read_csv(SRC).set_index(["method", "n_ris"])
    rows = []
    for n in sorted(t.index.get_level_values("n_ris").unique()):
        cells = []
        for m in ORDER:
            r = t.loc[(m, n)]
            sd = r.sum_rate_std
            if sd < 0.0005:
                cells.append(f"{vn(r.sum_rate_mean, 3)} (SD $<$ 0,0005)")
            else:
                cells.append(f"{vn(r.sum_rate_mean, 3)} $\\pm$ {vn(sd, 3)}")
        rows.append(f"${n}$ & " + " & ".join(cells) + r" \\")
    tex = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Tổng tốc độ trên 1.000 kịch bản kiểm thử, trung bình $\pm$ độ lệch chuẩn (bit/s/Hz)}",
        r"\label{tab:ch4-tong-toc-do}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        "$N$ & " + " & ".join(f"\\textbf{{{h}}}" for h in HEAD) + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"\parbox{\linewidth}{\scriptsize Ghi chú: với TD3, DDPG và PPO, độ lệch chuẩn tính giữa năm"
        r" trung bình theo hạt giống huấn luyện; với AO--SCA, AO--Grid và AnalyticalRIS (thuật"
        r" toán xác định), độ lệch chuẩn tính giữa $1.000$ kịch bản kiểm thử. Hai đại lượng"
        r" phản ánh hai nguồn bất định khác nhau nên không so sánh trực tiếp với nhau. Độ lệch"
        r" chuẩn mô tả độ phân tán, không phải khoảng tin cậy. Riêng các ô AnalyticalRIS khi"
        r" $N \ge 32$, độ lệch chuẩn được ghi trong ngoặc vì nhỏ hơn 0,0005.}",
        r"\tablesource{Tác giả tổng hợp từ kết quả kiểm thử đã khóa (2026)}",
        r"\end{table}",
        "",
    ])
    OUT.write_text(tex, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
