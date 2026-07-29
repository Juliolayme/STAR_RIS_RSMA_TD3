# Nhánh luận văn 4 chương

Nhánh làm việc: `thesis/restructure-4chap-v2`

Nhánh này được tạo từ `thesis/latex-full-v1` để tái cấu trúc luận văn theo yêu cầu mới.

## Trạng thái hiện tại

- Đã đồng bộ toàn bộ source code và kết quả benchmark sáu phương pháp từ `main` thông qua merge commit `b1f1f2437b6361b8d2e38b55678da4185a815470`.
- Benchmark dùng scientific commit `99318fefa53bef91fa5f105ec71ddae73fc96c39`.
- Kết quả đã được publish trên `main` tại commit `5dce28bd5b27f2ac8eb59d4045cb8ee88076bdf8`.
- `results/six_method_v1/SIX_METHOD_AUDIT.json` có verdict `PASS`.
- Đã có đủ TD3, DDPG, PPO, AO-SCA, AO-Grid và AnalyticalRIS trên cùng ScenarioBank.
- Đã tạo các bảng LaTeX, bảng siêu tham số và phần giới hạn thống kê trong `thesis/generated/`.

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

Xây dựng thuật toán TD3; mô tả DDPG và PPO làm phương pháp so sánh. Bảng siêu tham số dùng file:

```latex
\input{generated/table_drl_hyperparameters}
```

### Chương 4

Mô phỏng và đánh giá TD3, DDPG, PPO, AO-SCA, AO-Grid và AnalyticalRIS bằng dữ liệu đã khóa và kiểm toán.

Các tài sản đã chuẩn bị:

```latex
\input{generated/table_six_method_performance}
\input{generated/table_six_method_latency}
\input{generated/table_td3_paired_tests_holm}
\input{generated/statistical_limitations}
```

## Thư mục hình

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

Review độc lập ba hình Chương 4 nằm trong:

```text
thesis/FIGURE_REVIEW_SENIOR.md
```

Kết luận hiện tại:

- Hình 4.1: đạt.
- Hình 4.3: đạt.
- Hình 4.2: đạt về dữ liệu và kích thước nhưng còn nhãn trục tiếng Anh; cần vẽ lại bốn panel từ `TABLE_SIX_METHOD_PERFORMANCE.csv` để đáp ứng yêu cầu toàn bộ text tiếng Việt.

## Quy trình làm hình

1. Clone và checkout nhánh này.
2. Tạo từng hình theo prompt.
3. Xuất đủ `.drawio`, `.svg`, `.pdf`, `.png`.
4. Đặt file đúng thư mục và đúng tên quy ước.
5. Cập nhật `thesis/FIGURE_REVIEW_4_CHAPTERS.md` sau mỗi lần sửa.
6. Commit và push lại đúng nhánh `thesis/restructure-4chap-v2`.

## Tái tạo bảng Chương 4

Chạy từ thư mục gốc repository:

```bash
python thesis/scripts/generate_six_method_thesis_assets.py
```

Script chỉ sinh bảng khi audit verdict là `PASS` và ScenarioBank được xác nhận dùng chung.

## Lệnh clone/checkout

```bash
git clone https://github.com/Juliolayme/STAR_RIS_RSMA_TD3.git
cd STAR_RIS_RSMA_TD3
git checkout thesis/restructure-4chap-v2
```

Repository đã được clone:

```bash
git fetch origin
git checkout thesis/restructure-4chap-v2
git pull origin thesis/restructure-4chap-v2
```

## Quy tắc khoa học

- Không đưa MADDPG hoặc CTDE trở lại luận văn.
- Mô hình thực nghiệm là SISO STAR-RIS energy splitting hỗ trợ RSMA.
- AO-SCA là solver cục bộ, không phải nghiệm tối ưu toàn cục.
- Có thể kết luận TD3 là phương pháp DRL tốt nhất trong ba implementation được đánh giá.
- Không kết luận TD3 tốt nhất trong cả sáu phương pháp hoặc có tổng tốc độ cao nhất.
- Không kết luận TD3 vượt DDPG trên mọi chỉ số tại mọi N.
- Không gọi TD3 là phương pháp nhanh nhất tuyệt đối vì AnalyticalRIS có độ trễ thấp hơn nhưng không đạt QoS.
- Kết quả Chương 4 chỉ lấy từ dữ liệu thật, cùng ScenarioBank và đã được kiểm toán.
