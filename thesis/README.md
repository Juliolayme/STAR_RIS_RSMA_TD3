# Luận văn LaTeX – STAR-RIS–RSMA TD3

## Build cục bộ

Yêu cầu: XeLaTeX, latexmk, biber, TeX Live Vietnamese, latex-extra và science.

```bash
cd thesis
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

PDF sinh tại `thesis/main.pdf`.

## Build trên GitHub Actions

Workflow `.github/workflows/build-thesis.yml` chạy khi có thay đổi LaTeX/Markdown trên nhánh `thesis/latex-full-v1`.

Workflow:
1. cài TeX Live và biber;
2. build bằng XeLaTeX;
3. kiểm tra PDF bằng `pdfinfo`;
4. upload artifact;
5. commit bản PDF vào `thesis/build/Nguyen_Duy_Thanh_STAR_RIS_RSMA_TD3_Thesis.pdf` trên cùng nhánh.

## Font

LaTeX ưu tiên Times New Roman khi hệ thống có font. CI dùng TeX Gyre Termes làm fallback tương thích Times. Không commit font vào repository.

## Hình minh họa

Nếu file Draw.io chưa tồn tại, bản PDF hiển thị placeholder và vẫn build. Tạo hình theo `DRAWIO_FIGURE_SPEC.md`, xuất SVG/PDF vào `thesis/figures/`.

## Thông tin cần xác nhận trước khi nộp

- “Luận văn thạc sĩ” hay “Đề án tốt nghiệp thạc sĩ”.
- Tháng/năm chính thức.
- Logo UTH chính thức.
- Cách ghi tên hai người hướng dẫn.
- Bảng kết quả per-N và baseline DDPG/PPO nếu hoàn tất.
