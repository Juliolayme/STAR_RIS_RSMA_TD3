# Gộp code chuẩn và hình vẽ vào nhánh `thesis/latex-review`

Gói này chứa mọi thứ cần đưa vào nhánh nộp. Giải nén đè lên bản làm việc của
nhánh `thesis/latex-review`, kiểm tra rồi commit.

## Gói này có gì

| Đường dẫn | Nguồn | Vì sao cần |
|---|---|---|
| `src/` | `experiment/physical-v6-full` | Nhánh `latex-review` đang mang code cũ, `action.py` chỉ có `legacy_v1` và `physical_v3`, **không có** `physical_v6_soft_anchor`. Không có nó thì không tái tạo được số nào ở Chương 4. |
| `configs/` | cùng nguồn | Bổ sung `configs/v3/pilot_v6_soft_anchor_n{16,32,64,96,128}.yaml`, đủ cả năm kích thước `N` như công thức (3.25) |
| `scripts/` | cùng nguồn | Thêm các script dựng bảng, hình và kiểm định cho lần chạy v6 |
| `results/physical_v6_full_r2/` | cùng nguồn | Bộ số liệu mà Bảng 4.2–4.5 trích dẫn, kèm `README.md` ánh xạ từng bảng/hình |
| `thesis_latex/figures/tikz/` | mới | Hình TikZ cho Chương 1–3, cả `.tex` lẫn `.pdf` đã biên dịch |

## Cách áp dụng

```bash
git clone https://github.com/Juliolayme/STAR_RIS_RSMA_TD3.git
cd STAR_RIS_RSMA_TD3
git checkout thesis/latex-review

# tạo nhánh phụ để đối chiếu trước khi gộp thẳng
git checkout -b thesis/latex-review-v6

# giải nén gói này vào thư mục gốc của repo
unzip -o /duong/dan/toi/overlay_latex_review.zip -d .

git status
git diff --stat
```

Xem kỹ `git diff` ở `src/star_ris_rsma/action.py`. Phải thấy xuất hiện
`physical_v6_soft_anchor`, `weighted_reference_phase`,
`STRUCTURED_SOFTMAX_TEMPERATURE = 10.0` và `PHASE_RESIDUAL_SCALE = 0.25`.

```bash
git add -A
git commit -m "Đưa code v6 và hình TikZ vào nhánh luận văn

- src/, configs/, scripts/ lấy từ experiment/physical-v6-full (fde2fe7)
- results/physical_v6_full_r2/ là bộ số liệu Chương 4 trích dẫn
- thesis_latex/figures/tikz/ là hình Chương 1-3"

git push origin thesis/latex-review-v6
```

Nếu muốn ghi thẳng vào `thesis/latex-review`, thay hai lệnh cuối bằng:

```bash
git checkout thesis/latex-review
git merge thesis/latex-review-v6
git push origin thesis/latex-review
```

## Dung lượng

Tệp `results/physical_v6_full_r2/raw/DRL_V6_TEST_BEST_RAW_ALL.csv` nặng khoảng
24 MB. GitHub nhận được nhưng sẽ cảnh báo nếu vượt 50 MB. Nếu muốn nhẹ hơn, có
thể bỏ thư mục `raw/` khỏi commit — mọi bảng và hình trong luận văn đều dựng từ
`tables/`, không cần `raw/`. Khi đó thêm dòng sau vào `.gitignore`:

```
results/physical_v6_full_r2/raw/
```

## Ba việc còn lại cần bạn tự làm

1. **Nguồn LaTeX của luận văn chưa có trên nhánh này.** `git ls-tree` ở
   `thesis/latex-review` chỉ trả về đúng một tệp là `thesis_latex/README.md`.
   Toàn bộ `main.tex`, các chương và thư mục hình chưa được commit. Kiểm tra
   `.gitignore` rồi `git add` phần còn thiếu.

2. **Phụ lục B** có bảng ánh xạ nội dung luận văn với tệp mã nguồn. Bổ sung tên
   nhánh và commit `fde2fe7` vào đó để người đọc không mở nhầm nhánh.

3. **Tiêu đề Bảng 4.5** cần ghi rõ là giá trị trung vị. Xem phần cảnh báo trong
   `results/physical_v6_full_r2/README.md`.
