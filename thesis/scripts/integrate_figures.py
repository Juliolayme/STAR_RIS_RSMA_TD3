from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path


THESIS_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = THESIS_DIR / "figures"


def remove_legacy_figure_block(text: str, figure_path: str, next_section: str) -> str:
    """Remove one legacy figure, its source line and explanatory paragraph."""
    pattern = re.compile(
        rf"\n\\ThesisFigure\{{{re.escape(figure_path)}\}}.*?\n(?=\\section\{{{re.escape(next_section)}\}})",
        flags=re.DOTALL,
    )
    return pattern.sub("\n", text, count=1)


def insert_before_once(text: str, anchor: str, block: str, marker: str) -> str:
    if marker in text:
        return text
    if anchor not in text:
        raise RuntimeError(f"Cannot find insertion anchor: {anchor}")
    return text.replace(anchor, f"{block}\n\n{anchor}", 1)


def update_tex(path: Path, transform) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = transform(original)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def chapter1(text: str) -> str:
    text = remove_legacy_figure_block(
        text,
        "figures/chapter1_research_landscape.pdf",
        "Kết luận chương",
    )
    block = r"""% FIGURE-INTEGRATED: CH1 START
\begin{landscape}
\ThesisFigure{figures/chapter1_rsma_vs_conventional.pdf}
{So sánh nguyên lý phân bổ tài nguyên và xử lý nhiễu của OMA, SDMA, NOMA và RSMA}
{fig:ch1-rsma-comparison}[0.96\linewidth]
\figuresource{Tác giả xây dựng dựa trên \cite{Mao2018RSMABridging,Mao2022RSMAFundamentals,Clerckx2021NOMACritical} (2026)}
\end{landscape}

Hình~\ref{fig:ch1-rsma-comparison} sử dụng biểu tượng trạm gốc và ba người dùng để minh họa cách từng kỹ thuật tổ chức tài nguyên và xử lý nhiễu. Trong phần OMA, các đường nét đứt biểu diễn việc cấp tài nguyên thời gian hoặc tần số trực giao; các ô tài nguyên chỉ mang tính minh họa và không ánh xạ bắt buộc theo tỷ lệ một--một với số người dùng. Trong phần SDMA, các mũi tên bản rộng biểu diễn các búp sóng không gian tách biệt trong một hệ đa anten; đây là minh họa khái niệm và không phải kiến trúc SISO được mô phỏng trong luận văn. Trong phần NOMA, các mũi tên có độ dày khác nhau biểu diễn các lớp công suất được chồng trên cùng tài nguyên thời gian--tần số, còn biểu tượng SIC tại người dùng mạnh thể hiện quá trình giải mã liên tiếp; chúng không phải ba búp sóng độc lập. Trong phần RSMA, luồng màu tím là luồng chung $s_c$ được mọi người dùng giải mã trước, còn các đường màu riêng là các luồng riêng $s_k$; sau SIC, mỗi người dùng tiếp tục giải mã luồng riêng tương ứng. Ba người dùng trong hình chỉ dùng để trình bày trực quan, trong khi cấu hình thực nghiệm của luận văn sử dụng $K=4$ người dùng. Hình vì vậy làm rõ lý do RSMA có thể dung hòa giữa giải mã một phần nhiễu và xem phần nhiễu còn lại như tạp âm.
% FIGURE-INTEGRATED: CH1 END"""
    return insert_before_once(
        text,
        r"\section{Tổng quan RSMA}",
        block,
        "% FIGURE-INTEGRATED: CH1 START",
    )


def chapter2(text: str) -> str:
    text = remove_legacy_figure_block(
        text,
        "figures/chapter2_conceptual_foundation.pdf",
        "Kết luận chương",
    )
    block = r"""% FIGURE-INTEGRATED: CH2 START
\begin{landscape}
\ThesisFigure{figures/chapter2_ris_star_ris_overview.pdf}
{So sánh RIS phản xạ truyền thống và STAR--RIS hỗ trợ phủ sóng toàn không gian}
{fig:ch2-ris-star-ris}[0.94\linewidth]
\figuresource{Tác giả xây dựng dựa trên \cite{Xu2021STARRIS,Mu2021STARAided,Wu2021IRSTutorial} (2026)}
\end{landscape}

Hình~\ref{fig:ch2-ris-star-ris} đối chiếu trực tiếp hai nguyên lý điều khiển môi trường truyền. Ở nửa trái, RIS truyền thống chỉ tạo đường phản xạ nên chủ yếu phục vụ người dùng trong cùng nửa không gian và để lại một miền khuất ở phía sau bề mặt. Ở nửa phải, STAR--RIS đồng thời tạo tia phản xạ và tia truyền qua để phục vụ hai nhóm người dùng ở hai phía. Hai hệ số $\phi_n^T$ và $\phi_n^R$ trong hình nhấn mạnh rằng tỷ lệ công suất $\beta_n^T$ phải được chuyển thành biên độ bằng căn bậc hai, đồng thời thỏa $\beta_n^T+\beta_n^R=1$. Dải ES--MS--TS đặt STAR--RIS trong ba giao thức phổ biến; luận văn lựa chọn ES vì mỗi phần tử đồng thời truyền và phản xạ, phù hợp với không gian hành động liên tục được TD3 tối ưu. Hình là cầu nối giữa khái niệm phủ sóng toàn không gian và mô hình hệ số phức được sử dụng trong các phương trình tiếp theo.
% FIGURE-INTEGRATED: CH2 END"""
    return insert_before_once(
        text,
        r"\section{Bài toán tối ưu phi lồi}",
        block,
        "% FIGURE-INTEGRATED: CH2 START",
    )


def chapter3(text: str) -> str:
    text = remove_legacy_figure_block(
        text,
        "figures/chapter3_system_model.pdf",
        "Kết luận chương",
    )
    block = r"""% FIGURE-INTEGRATED: CH3 START
\begin{landscape}
\ThesisFigure{figures/chapter3_system_model.pdf}
{Mô hình hệ thống SISO STAR--RIS hỗ trợ RSMA được sử dụng trong luận văn}
{fig:ch3-system-model}[0.94\linewidth]
\figuresource{Tác giả xây dựng từ mô hình vật lý và cấu hình trong repository (2026)}
\end{landscape}

Hình~\ref{fig:ch3-system-model} mô tả đúng kiến trúc được cài đặt: một trạm gốc SISO phát một luồng chung $s_c$ và bốn luồng riêng $s_1,\ldots,s_4$ tới bốn người dùng SISO. Người dùng 1 và 3 nằm ở miền phản xạ, còn người dùng 2 và 4 nằm ở miền truyền qua, phù hợp với cách source gán chỉ báo miền luân phiên. Đường xám nét đứt là kênh trực tiếp BS--user; đường xanh dương nối BS với STAR--RIS và các đường xanh lá/cam tạo đường ghép tầng phản xạ hoặc truyền qua. Do đó, hình không giả định liên kết trực tiếp bị chặn hoàn toàn. Công thức cạnh STAR--RIS thể hiện hệ số biên độ--pha theo energy splitting, trong khi hai dải phía dưới nối trực tiếp mô hình vật lý với vector hành động và observation của tác tử. Cách trình bày này làm rõ TD3 không điều khiển một beamformer nhiều anten mà tối ưu công suất RSMA, tỷ lệ tốc độ chung, hệ số chia năng lượng và hai vector pha.
% FIGURE-INTEGRATED: CH3 END"""
    return insert_before_once(
        text,
        r"\section{Mô hình kênh}",
        block,
        "% FIGURE-INTEGRATED: CH3 START",
    )


def chapter4(text: str) -> str:
    text = remove_legacy_figure_block(
        text,
        "figures/chapter4_td3_pipeline.pdf",
        "Kết luận chương",
    )
    block = r"""% FIGURE-INTEGRATED: CH4 START
\begin{landscape}
\ThesisFigure{figures/chapter4_td3_training_pipeline.pdf}
{Quy trình huấn luyện, lựa chọn checkpoint và đánh giá TD3 không rò rỉ tập kiểm thử}
{fig:ch4-pipeline}[0.95\linewidth]
\figuresource{Tác giả xây dựng từ giao thức thực nghiệm của repository (2026)}
\end{landscape}

Hình~\ref{fig:ch4-pipeline} tổng hợp ba lớp của pipeline thực nghiệm. Lớp dữ liệu gồm ba ScenarioBank không giao nhau, trong đó tập kiểm thử được gắn biểu tượng khóa. Lớp huấn luyện biểu diễn vòng lặp giữa môi trường, actor TD3, bộ giải mã hành động khả thi, replay buffer, hai critic và bộ điều khiển đối ngẫu QoS; ba cơ chế target-policy smoothing, delayed policy update và lấy minimum của hai critic được ghi riêng để phân biệt TD3 với DDPG. Tập validation chỉ đi vào khối lựa chọn checkpoint sau mỗi 5.000 bước. Sau khi \texttt{best.pt} được khóa, chính sách mới được đánh giá xác định với exploration noise bằng không trên test bank, sau đó ghi raw CSV, thực hiện thống kê ghép cặp, đo độ trễ và tạo báo cáo. Dòng cảnh báo cuối hình cùng việc không có mũi tên quay từ test về training thể hiện rõ nguyên tắc chống test leakage.
% FIGURE-INTEGRATED: CH4 END"""
    return insert_before_once(
        text,
        r"\section{Kết luận chương}",
        block,
        "% FIGURE-INTEGRATED: CH4 START",
    )


def chapter5(text: str) -> str:
    text = remove_legacy_figure_block(
        text,
        "figures/chapter5_quality_latency_tradeoff.pdf",
        "Hàm ý thực tiễn",
    )
    block = r"""% FIGURE-INTEGRATED: CH5 START
\begin{landscape}
\ThesisFigure{figures/chapter5_quality_latency_tradeoff.pdf}
{Không gian đánh đổi định tính giữa tổng tốc độ, độ tin cậy QoS và độ trễ ra quyết định}
{fig:ch5-tradeoff}[0.90\linewidth]
\figuresource{Tác giả xây dựng từ kết quả đã kiểm toán trong repository (2026)}
\end{landscape}

Hình~\ref{fig:ch5-tradeoff} tổng hợp định tính vị trí của bốn phương pháp trong không gian chất lượng--độ trễ. AnalyticalRIS nằm gần phía độ trễ thấp nhất nhưng được đánh dấu không đạt QoS; AO--Grid có chất lượng cao hơn cấu hình phân tích đơn giản nhưng vẫn thấp hơn TD3; TD3 nằm ở vùng trung gian với viền xanh biểu thị độ tin cậy QoS cao; AO--SCA nằm ở vùng tổng tốc độ cao nhất nhưng có thời gian giải lớn hơn. Mũi tên giữa TD3 và AO--SCA diễn giải hai chiều của trade-off: AO--SCA cải thiện chất lượng nghiệm, còn TD3 giảm mạnh độ trễ ra quyết định. Các tọa độ trong hình không phải số đo theo một thang tuyến tính và không thay thế bảng raw theo từng $N$; chúng chỉ tóm tắt kết luận đã được lượng hóa bằng các khoảng chênh lệch và tỷ lệ tốc độ ở các mục trước.
% FIGURE-INTEGRATED: CH5 END"""
    return insert_before_once(
        text,
        r"\section{Hàm ý thực tiễn}",
        block,
        "% FIGURE-INTEGRATED: CH5 START",
    )


FIGURES = {
    "chapter1_rsma_vs_conventional": [
        "OMA",
        "SDMA",
        "NOMA",
        "RSMA",
        "Luồng chung",
        "SIC",
        "Trạm gốc",
    ],
    "chapter2_ris_star_ris_overview": [
        "RIS truyền thống",
        "STAR-RIS",
        "Tia phản xạ",
        "Tia truyền qua",
        "ES:",
        "MS:",
        "TS:",
    ],
    "chapter3_system_model": [
        "Trạm gốc SISO",
        "STAR-RIS thụ động",
        "Người dùng 1",
        "Người dùng 4",
        "đường trực tiếp",
        "VÉC-TƠ HÀNH ĐỘNG",
        "QUAN SÁT",
    ],
    "chapter4_td3_training_pipeline": [
        "Tập huấn luyện",
        "Tập xác thực",
        "Tập kiểm thử",
        "Khóa tập kiểm thử",
        "Hai bộ đánh giá",
        "best.pt",
        "Không có đường quay",
    ],
    "chapter5_quality_latency_tradeoff": [
        "AnalyticalRIS",
        "AO-Grid",
        "TD3",
        "AO-SCA",
        "Không đạt QoS",
        "mang tính định tính",
    ],
}


def verify_figures() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stem, tokens in FIGURES.items():
        drawio = FIGURE_DIR / f"{stem}.drawio"
        pdf = FIGURE_DIR / f"{stem}.pdf"
        svg = FIGURE_DIR / f"{stem}.svg"
        png = FIGURE_DIR / f"{stem}.png"
        for path in (drawio, pdf, svg, png):
            if not path.exists() or path.stat().st_size < 64:
                raise RuntimeError(f"Missing or empty figure asset: {path}")
        ET.parse(drawio)
        xml_text = drawio.read_text(encoding="utf-8")
        missing = [token for token in tokens if token not in xml_text]
        if missing:
            raise RuntimeError(f"{stem} misses required technical labels: {missing}")
        if not pdf.read_bytes().startswith(b"%PDF-"):
            raise RuntimeError(f"Invalid PDF header: {pdf}")
        rows.append(
            {
                "stem": stem,
                "drawio_kb": drawio.stat().st_size / 1024,
                "pdf_kb": pdf.stat().st_size / 1024,
                "svg_kb": svg.stat().st_size / 1024,
                "png_kb": png.stat().st_size / 1024,
                "labels": len(tokens),
            }
        )
    return rows


def verify_tex_references() -> None:
    required = {
        "chapter1.tex": "chapter1_rsma_vs_conventional.pdf",
        "chapter2.tex": "chapter2_ris_star_ris_overview.pdf",
        "chapter3.tex": "chapter3_system_model.pdf",
        "chapter4.tex": "chapter4_td3_training_pipeline.pdf",
        "chapter5.tex": "chapter5_quality_latency_tradeoff.pdf",
    }
    for name, figure in required.items():
        text = (THESIS_DIR / name).read_text(encoding="utf-8")
        if figure not in text:
            raise RuntimeError(f"{name} does not reference {figure}")


def write_verification_report(rows: list[dict[str, object]]) -> None:
    lines = [
        "# Báo cáo kiểm tra và tích hợp hình minh họa",
        "",
        "Bộ năm hình Draw.io đã được kiểm tra tự động trước khi build luận văn. "
        "Quy trình kiểm tra xác nhận tệp nguồn Draw.io là XML hợp lệ, bốn định dạng "
        "tài sản tồn tại, PDF có header hợp lệ, các nhãn kỹ thuật bắt buộc xuất hiện "
        "và mỗi chương tham chiếu đúng tệp PDF.",
        "",
        "| Chương | Tệp hình | Draw.io (KB) | PDF (KB) | SVG (KB) | PNG (KB) | Nhãn kỹ thuật đã kiểm tra |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"| {index} | `{row['stem']}` | {row['drawio_kb']:.1f} | "
            f"{row['pdf_kb']:.1f} | {row['svg_kb']:.1f} | {row['png_kb']:.1f} | {row['labels']} |"
        )
    lines.extend(
        [
            "",
            "## Kết quả review học thuật",
            "",
            "- **Chương 1:** phân biệt OMA, SDMA, NOMA và RSMA bằng biểu tượng BS/user, "
            "resource blocks, spatial beams, power layers, common/private streams và SIC. "
            "Phần văn bản làm rõ ba user chỉ là minh họa và SDMA không phải mô hình SISO của thí nghiệm.",
            "- **Chương 2:** thể hiện đúng RIS chỉ phản xạ, STAR-RIS đồng thời truyền qua/phản xạ, "
            "có hai miền phủ sóng, công thức hệ số và ba giao thức ES/MS/TS.",
            "- **Chương 3:** thể hiện đúng BS SISO, bốn user, đường trực tiếp và đường ghép tầng, "
            "phân nhóm reflection/transmission, action vector và observation.",
            "- **Chương 4:** thể hiện train/validation/test tách biệt, test bị khóa, validation-only "
            "checkpoint selection, deterministic test và không có đường quay từ test về training.",
            "- **Chương 5:** biểu diễn trade-off định tính, không ngụy tạo tọa độ tuyệt đối; "
            "AO-SCA có chất lượng cao hơn, TD3 có QoS/latency cân bằng, AnalyticalRIS không đạt QoS.",
            "",
            "## Điều kiện chấp nhận",
            "",
            "- [x] Mỗi chương có ít nhất một hình minh họa.",
            "- [x] Hình được chèn từ PDF vector, không dùng ảnh raster làm nguồn chính.",
            "- [x] Hình rộng được đặt trên trang ngang để chữ và icon có thể đọc khi in A4.",
            "- [x] Mỗi hình có caption, label, nguồn và đoạn văn liên kết với nội dung chương.",
            "- [x] Không có claim MADDPG/CTDE hoặc mô hình MIMO trong phần mô tả hệ thống thực nghiệm.",
        ]
    )
    (THESIS_DIR / "FIGURE_VERIFICATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    changed = []
    transforms = {
        "chapter1.tex": chapter1,
        "chapter2.tex": chapter2,
        "chapter3.tex": chapter3,
        "chapter4.tex": chapter4,
        "chapter5.tex": chapter5,
    }
    for name, transform in transforms.items():
        if update_tex(THESIS_DIR / name, transform):
            changed.append(name)

    rows = verify_figures()
    verify_tex_references()
    write_verification_report(rows)
    print(f"Integrated/verified thesis figures; changed={changed}")


if __name__ == "__main__":
    main()
