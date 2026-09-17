# Tiến độ dựng lại luận văn

Nguồn dựng lại: `Luan_van_STAR_RIS_RSMA_DRL_ban_review.pdf` (102 trang, biên dịch
01/09/2026). Khung định dạng lấy từ nhánh `thesis/revision-review-v1`.

## Đã xong

- `main.tex`, `uththesis.sty`, `frontmatter.tex`, `references.bib`
- Chương 1 đầy đủ 8 mục, kèm Bảng 1.1 và bốn hình
- Chương 2 đầy đủ 10 mục, 36 công thức đánh số, bốn hình
- Chương 3 đầy đủ 13 mục, 25 công thức, sáu hình, Bảng 3.1 và Thuật toán 1
- Chương 4 đầy đủ 15 mục, năm bảng và năm hình. Bảng 4.2-4.5 và toàn bộ hình
  được sinh tự động từ results/physical_v6_full_r2/tables/ nên không gõ tay
- Mọi hình dùng bản TikZ mới vẽ lại

## Chưa xong

- Mở đầu (đang là chỗ giữ chỗ)
- Kết luận và ba phụ lục
- Các dòng `\input` tương ứng trong `main.tex` đang được chú thích

## Quyết định đã áp dụng

- Font cài cứng `Times New Roman` theo quy định trường.
- Nhãn "Ý nghĩa vật lý" đã bỏ; nội dung gộp vào đoạn "Trong đó". Khi phần diễn
  giải dài thì tách thành đoạn mới nhưng không thêm nhãn.
- Chú thích hình và bảng dùng dấu hai chấm, khớp bản PDF đang nộp. Khung
  `uththesis.sty` gốc dùng dấu chấm nên đã sửa `labelsep`.
- Thêm gói `icomma` để dấu phẩy thập phân trong công thức không bị giãn.
- Trích dẫn dùng khóa BibTeX thay vì số cứng, nên chèn thêm tài liệu sẽ không
  làm sai số thứ tự.
- Hạn chế từ tiếng Anh trong văn bản tiếng Việt: panel thành khung, beamforming
  thành tạo búp sóng, Replay Buffer thành bộ nhớ phát lại, checkpoint thành điểm
  lưu mô hình, mini-batch thành lô nhỏ, episode thành giai đoạn, off-policy và
  on-policy thành học ngoài chính sách và học theo chính sách hiện hành.

## Sửa lỗi so với bản PDF

- `references.bib` thiếu mục PPO của Schulman và cộng sự, tương ứng trích dẫn
  [14]. Đã bổ sung `Schulman2017PPO`.
- Dấu thập phân trong công thức không nhất quán giữa dấu chấm và dấu phẩy.
  Đã thống nhất về dấu phẩy.
- Ở mục về hiệu chỉnh pha, bản PDF ghi "Công thức (3.13)" trong khi nội dung
  nói về cả hai công thức. Đã sửa thành "Hai công thức (3.12) và (3.13)".

## Biên dịch

```bash
cd thesis_latex
xelatex main && biber main && xelatex main && xelatex main
```

Máy biên dịch cần có font Times New Roman. Nếu không có, thay dòng
`\setmainfont{Times New Roman}` trong `uththesis.sty` bằng `Tinos` hoặc
`TeX Gyre Termes` để kiểm tra bố cục.
