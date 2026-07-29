# Nhánh luận văn 4 chương

Nhánh làm việc: `thesis/restructure-4chap-v2`

Nhánh này được tạo từ `thesis/latex-full-v1` để tái cấu trúc luận văn theo yêu cầu mới.

## Cấu trúc đã chốt

- Không có Tóm tắt tiếng Việt.
- Không có Abstract tiếng Anh.
- Giữ nguyên tên đề tài đã được trường duyệt; trang bìa là ngoại lệ đối với quy tắc khai báo từ viết tắt.
- Trong nội dung, lần đầu xuất hiện viết theo dạng: thuật ngữ tiếng Việt (thuật ngữ đầy đủ tiếng Anh, viết tắt). Các lần sau chỉ dùng từ viết tắt.
- Mở đầu + 4 chương + Kết luận.

### Chương 1

Cơ sở lý thuyết và các công nghệ sử dụng trong luận văn. Giải thích trực quan, dễ hiểu, hạn chế công thức.

### Chương 2

Mô hình hệ thống và bài toán tối ưu.

### Chương 3

Xây dựng thuật toán TD3. DDPG và PPO được mô tả ở mức phương pháp so sánh; kết quả chỉ được đưa vào khi có dữ liệu khóa và kiểm toán.

### Chương 4

Mô phỏng và đánh giá TD3, DDPG, PPO, AO-SCA, AO-Grid và AnalyticalRIS sau khi đủ kết quả. Không tạo số liệu giả hoặc dùng kết quả chưa khóa.

## Thư mục hình đề xuất

```text
thesis/figures/
├── drawio/chapter1/
├── drawio/chapter2/
├── drawio/chapter3/
├── drawio/chapter4/
├── svg/
├── pdf/
├── png/
└── results/
```

Tên và prompt chi tiết nằm trong:

```text
thesis/DRAWIO_PROMPTS_THESIS_4_CHAPTERS.md
```

## Quy trình làm hình

1. Clone và checkout nhánh này.
2. Tạo từng hình theo prompt.
3. Xuất đủ `.drawio`, `.svg`, `.pdf`, `.png`.
4. Đặt file đúng thư mục và đúng tên quy ước.
5. Tạo `thesis/FIGURE_REVIEW_4_CHAPTERS.md` để ghi kết quả verify.
6. Commit và push lại đúng nhánh `thesis/restructure-4chap-v2`.

## Lệnh clone/checkout

```bash
git clone https://github.com/Juliolayme/STAR_RIS_RSMA_TD3.git
cd STAR_RIS_RSMA_TD3
git checkout thesis/restructure-4chap-v2
```

Nếu repository đã được clone:

```bash
git fetch origin
git checkout thesis/restructure-4chap-v2
git pull origin thesis/restructure-4chap-v2
```

## Quy tắc khoa học

- Không đưa MADDPG hoặc CTDE trở lại luận văn.
- Mô hình thực nghiệm là SISO STAR-RIS energy splitting hỗ trợ RSMA.
- AO-SCA là solver cục bộ, không phải nghiệm tối ưu toàn cục.
- Không kết luận TD3 tốt nhất trên mọi chỉ số.
- Kết quả Chương 4 chỉ lấy từ dữ liệu thật, cùng ScenarioBank và đã được kiểm toán.
