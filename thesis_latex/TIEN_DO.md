# Tình trạng bản dựng lại

Nguồn dựng lại: `Luan_van_STAR_RIS_RSMA_DRL_ban_review.pdf` (102 trang, biên dịch
01/09/2026). Khung định dạng lấy từ nhánh `thesis/revision-review-v1`.

## Đã hoàn thành

Toàn bộ luận văn, **106 trang**, biên dịch sạch: không lỗi ký tự thiếu, không
tham chiếu hỏng, không trích dẫn thiếu, không dòng tràn lề.

| Phần | Nội dung |
|---|---|
| Mở đầu | 8 mục |
| Chương 1 | 8 mục, Bảng 1.1, bốn hình |
| Chương 2 | 10 mục, 36 công thức, bốn hình |
| Chương 3 | 13 mục, 25 công thức, sáu hình, Bảng 3.1, Thuật toán 1 |
| Chương 4 | 15 mục, năm bảng, năm hình |
| Kết luận | 2 mục |
| Phụ lục A | Danh mục ký hiệu, số chiều trạng thái và hành động |
| Phụ lục B | Quy trình tái lập và ánh xạ mã nguồn |

Tổng cộng **19 hình** và **10 bảng**.

## Quyết định đã áp dụng

- Font cài cứng `Times New Roman` theo quy định trường.
- Bỏ nhãn "Ý nghĩa vật lý"; nội dung gộp vào đoạn "Trong đó". Phần diễn giải
  dài tách thành đoạn riêng, không nhãn.
- Chú thích hình và bảng dùng dấu hai chấm, khớp bản PDF gốc.
- Trích dẫn dùng khóa BibTeX, không dùng số cứng.
- Hạn chế từ tiếng Anh trong văn bản; mọi thuật ngữ Việt hóa đều mở ngoặc ghi
  từ tiếng Anh ở lần xuất hiện đầu.
- Mọi từ viết tắt được ghi đầy đủ ở lần xuất hiện đầu.
- Bảng và hình của Chương 4 sinh tự động từ `results/physical_v6_full_r2/`.

## Sửa lỗi so với bản PDF gốc

- `references.bib` thiếu mục PPO của Schulman, tương ứng trích dẫn [14].
- Dấu thập phân không nhất quán giữa dấu chấm và dấu phẩy trong công thức.
- Mục hiệu chỉnh pha ghi "Công thức (3.13)" trong khi nội dung nói về cả hai.
- Tiêu đề Bảng 4.5 không nêu rõ là giá trị trung vị.
- Chú giải Hình 4.1 và 4.2 đặt trong khung, che mất đường biểu diễn.
- Danh mục từ viết tắt có 4 mục không dùng và thiếu 5 mục đang dùng.
- Phụ lục B bổ sung tên nhánh và mã phiên bản của mã nguồn.

## Biên dịch

```bash
cd thesis_latex
xelatex main && biber main && xelatex main && xelatex main
```

Máy biên dịch cần có font Times New Roman.
