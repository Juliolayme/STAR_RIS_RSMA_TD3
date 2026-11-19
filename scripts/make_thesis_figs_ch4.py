"""Sinh Hình 4.1 (đường cong xác thực) và Hình 4.2 (tổng tốc độ theo N) của luận văn từ CSV đã khóa.

Hình 4.1: đường đậm là trung bình 5 hạt giống; dải mờ là khoảng nhỏ nhất – lớn nhất giữa các hạt giống.
Hình 4.2: thanh sai số của TD3, DDPG, PPO là ± độ lệch chuẩn giữa 5 trung bình theo hạt giống
(cột sum_rate_std); các bộ giải xác định không có thanh sai số theo hạt giống.
Chạy: python scripts/make_thesis_figs_ch4.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results/physical_v6_full_r2/tables"
FIG = ROOT / "thesis_latex/figures"
N_COLORS = {16: "cUEc", 32: "cUEa", 64: "cUEb", 96: "cUEd", 128: "cChung"}


def coords(pairs):
    return " ".join(f"({x:g},{y:.4f})" for x, y in pairs)


def validation_figure() -> str:
    d = pd.read_csv(TAB / "TABLE_V6_VALIDATION_CURVES.csv")
    g = d.groupby(["method", "n_ris", "eval_step"]).mean_sum_rate.agg(["mean", "min", "max"]).reset_index()
    common = (r"width=6.0cm, height=6.2cm, xlabel={Bước huấn luyện}, xmin=0, xmax=100000, ymin=3, ymax=23, "
              r"scaled x ticks=false, xtick={0,50000,100000}, xticklabels={$0$,$50\text{k}$,$100\text{k}$}, "
              r"grid=both, grid style={gray!25}, tick label style={font=\footnotesize}, "
              r"label style={font=\small}, title style={font=\small\bfseries}")
    out = [r"% Sinh tu dong boi scripts/make_thesis_figs_ch4.py tu TABLE_V6_VALIDATION_CURVES.csv.",
           r"% Duong dam: trung binh 5 hat giong; dai mo: nho nhat - lon nhat giua cac hat giong.",
           r"\begin{tikzpicture}[font=\small]"]
    for k, (m, title) in enumerate([("td3", "TD3"), ("ddpg", "DDPG"), ("ppo", "PPO")]):
        opts = f"name={m}, title={{{title}}}, {common}"
        if k == 0:
            opts += r", ylabel={Tổng tốc độ trên tập xác thực (bit/s/Hz)}"
        else:
            prev = ["td3", "ddpg"][k - 1]
            opts += f", yticklabel=\\empty, at={{({prev}.east)}}, anchor=west, xshift=0.8cm"
        if k == 1:
            opts += (r", legend style={font=\small,at={(0.5,-0.26)},anchor=north,draw=gray!50,legend columns=6,"
                     r"/tikz/every even column/.append style={column sep=8pt}}, legend cell align=left")
        out.append(f"\\begin{{axis}}[{opts}]")
        for n, col in N_COLORS.items():
            s = g[(g.method == m) & (g.n_ris == n)].sort_values("eval_step")
            band = list(zip(s.eval_step, s["min"])) + list(zip(s.eval_step[::-1], s["max"][::-1]))
            out.append(f"\\addplot[fill={col},fill opacity=0.18,draw=none,forget plot] coordinates {{{coords(band)}}} -- cycle;")
            out.append(f"\\addplot[very thick,{col}] coordinates {{{coords(zip(s.eval_step, s['mean']))}}};")
            if k == 1:
                out.append(f"\\addlegendentry{{$N={n}$}}")
        if k == 1:
            out.append(r"\addlegendimage{area legend,fill=gray,fill opacity=0.18,draw=none}")
            out.append(r"\addlegendentry{dải 5 hạt giống}")
        out.append(r"\end{axis}")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out) + "\n"


def sum_rate_figure() -> str:
    t = pd.read_csv(TAB / "TABLE_V6_SIX_METHOD_PERFORMANCE.csv")
    spec = [("td3", "cUEc", "*", "TD3", True), ("ddpg", "cUEa", "square*", "DDPG", True),
            ("ppo", "cUEb", "triangle*", "PPO", True), ("ao_sca", "cUEd", "diamond*", "AO--SCA", False),
            ("ao_grid", "cChung", "pentagon*", "AO--Grid", False), ("analytical_ris", "cKenh", "o", "AnalyticalRIS", False)]
    out = [r"% Sinh tu dong boi scripts/make_thesis_figs_ch4.py tu TABLE_V6_SIX_METHOD_PERFORMANCE.csv.",
           r"% Thanh sai so (TD3, DDPG, PPO): +- do lech chuan giua 5 trung binh theo hat giong.",
           r"\begin{tikzpicture}[font=\small]", r"\begin{axis}[",
           r"  width=13cm, height=8cm,", r"  xlabel={Số phần tử STAR--RIS $N$},",
           r"  ylabel={Tổng tốc độ trung bình (bit/s/Hz)},", r"  xtick={16,32,64,96,128}, xmin=8, xmax=136,",
           r"  ymin=0, ymax=24,", r"  grid=both, grid style={gray!25},",
           r"  tick label style={font=\small}, label style={font=\small},",
           r"  legend style={font=\small,at={(0.5,-0.18)},anchor=north,draw=gray!50,",
           r"                legend columns=3,/tikz/every even column/.append style={column sep=10pt}},",
           r"  legend cell align=left,", r"]"]
    for m, col, mark, lab, err in spec:
        s = t[t.method == m].sort_values("n_ris")
        if err:
            c = " ".join(f"({n},{v:.6f}) +- (0,{e:.6f})" for n, v, e in zip(s.n_ris, s.sum_rate_mean, s.sum_rate_std))
            opt = f"very thick,{col},mark={mark},mark size=2pt,error bars/.cd,y dir=both,y explicit,error bar style={{line width=0.8pt,{col}}},error mark options={{rotate=90,mark size=3pt,{col}}}"
        else:
            c = " ".join(f"({n},{v:.6f})" for n, v in zip(s.n_ris, s.sum_rate_mean))
            opt = f"very thick,{col},mark={mark},mark size=2pt"
        out.append(f"\\addplot[{opt}] coordinates {{{c}}};")
        out.append(f"\\addlegendentry{{{lab}}}")
    out += [r"\end{axis}", r"\end{tikzpicture}"]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    (FIG / "fig_ch4_duong_cong_xac_thuc.tex").write_text(validation_figure(), encoding="utf-8")
    (FIG / "fig_ch4_tong_toc_do.tex").write_text(sum_rate_figure(), encoding="utf-8")
    print("wrote fig_ch4_duong_cong_xac_thuc.tex, fig_ch4_tong_toc_do.tex")
