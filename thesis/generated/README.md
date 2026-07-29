# Tài sản LaTeX sinh từ benchmark sáu phương pháp

Thư mục này chứa các bảng và đoạn nội dung được chuẩn bị cho bản luận văn 4 chương.

## Nguồn dữ liệu

- `results/six_method_v1/SIX_METHOD_AUDIT.json`
- `results/six_method_v1/SIX_METHOD_REVIEW.md`
- `results/six_method_v1/tables/TABLE_SIX_METHOD_PERFORMANCE.csv`
- `results/six_method_v1/tables/TABLE_SIX_METHOD_CPU_LATENCY.csv`
- `results/six_method_v1/tables/TABLE_SIX_METHOD_PAIRED_TESTS_HOLM.csv`

Scientific commit của benchmark: `99318fefa53bef91fa5f105ec71ddae73fc96c39`.

Commit publish kết quả lên `main`: `5dce28bd5b27f2ac8eb59d4045cb8ee88076bdf8`.

## Các file đã tạo

- `table_drl_hyperparameters.tex`: thiết lập chung và siêu tham số riêng của TD3, DDPG, PPO.
- `table_six_method_performance.tex`: bảng hiệu năng đầy đủ của sáu phương pháp tại năm giá trị N.
- `table_six_method_latency.tex`: bảng độ trễ CPU của sáu phương pháp.
- `table_td3_paired_tests_holm.tex`: các so sánh ghép cặp có TD3 sau hiệu chỉnh Holm.
- `statistical_limitations.tex`: phần giới hạn thống kê và phạm vi diễn giải dùng cho Chương 4.

## Cách tái tạo các bảng số liệu

Chạy từ thư mục gốc của repository:

```bash
python thesis/scripts/generate_six_method_thesis_assets.py
```

Script chỉ chạy khi `SIX_METHOD_AUDIT.json` có verdict `PASS` và xác nhận các phương pháp dùng chung ScenarioBank.

## Vị trí chèn trong luận văn 4 chương

### Chương 3 — Xây dựng thuật toán

```latex
\input{generated/table_drl_hyperparameters}
```

Đặt sau phần mô tả cấu hình TD3, DDPG và PPO.

### Chương 4 — Mô phỏng và đánh giá

```latex
\input{generated/table_six_method_performance}
\input{generated/table_six_method_latency}
\input{generated/table_td3_paired_tests_holm}
\input{generated/statistical_limitations}
```

Không nên đặt ba bảng dài liên tiếp trong thân chương. Bảng hiệu năng và bảng độ trễ dùng trong nội dung chính; bảng kiểm định đầy đủ có thể chuyển sang phụ lục, còn thân chương chỉ giữ các hàng so sánh TD3.

## Quy tắc diễn giải

- Có thể viết: TD3 là phương pháp DRL tốt nhất và có khả năng mở rộng tốt nhất trong ba implementation được đánh giá.
- Phải viết: AO-SCA đạt tổng tốc độ cao nhất trong thiết lập thí nghiệm.
- Không được viết: TD3 đạt tổng tốc độ cao nhất trong sáu phương pháp.
- Không được viết: TD3 vượt DDPG trên mọi chỉ số tại mọi N.
- Không được gọi AO-SCA là nghiệm tối ưu toàn cục hoặc cận trên lý thuyết.
- Không được gọi TD3 là phương pháp nhanh nhất tuyệt đối vì AnalyticalRIS nhanh hơn nhưng không đáp ứng QoS.
