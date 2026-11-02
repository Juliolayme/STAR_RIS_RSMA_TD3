# Bộ kết quả dùng cho luận văn

**Đây là bộ kết quả duy nhất được trích dẫn trong luận văn.** Mọi số liệu ở
Chương 4 đều lấy từ thư mục này.

## Nguồn

- Nhánh sinh dữ liệu: `experiment/physical-v6-full`, commit `fde2fe7798eaf080886b121cb7f66492b4c9bc48`
- Tham số hóa hành động: `physical_v6_soft_anchor`
- Cấu hình: `configs/v3/pilot_v6_soft_anchor_n{16,32,64,96,128}.yaml`

## Giao thức

- Kích thước STAR–RIS: `N = 16, 32, 64, 96, 128`
- Ngân sách huấn luyện: `100.000` tương tác mỗi hạt giống
- Số hạt giống: 5 cho mỗi thuật toán và mỗi `N`
- Ngân hàng kịch bản: huấn luyện `10.000` (seed 11001), xác thực `1.000`
  (seed 22001), kiểm thử `1.000` (seed 33001)
- Thuật toán: TD3, DDPG, PPO, AO–SCA, AO–Grid, AnalyticalRIS

## Ánh xạ sang bảng và hình trong luận văn (bản `thesis-v6`)

Các hình trong luận văn được vẽ lại bằng TikZ/PGFPlots trong `thesis_latex/figures/`, với số liệu lấy từ các tệp dưới đây.

| Nội dung luận văn | Tệp trong thư mục này |
|---|---|
| Bảng 4.2 — Độ lệch chuẩn giữa năm hạt giống | `tables/TABLE_V6_SIX_METHOD_PERFORMANCE.csv`, cột `sum_rate_std` |
| Bảng 4.3 — Tổng tốc độ, trung bình ± độ lệch chuẩn | cùng tệp, cột `sum_rate_mean`, `sum_rate_std` (sinh bằng `scripts/make_thesis_table_sum_rate.py`) |
| Bảng 4.4 — Xác suất toàn bộ UE đạt QoS | cùng tệp, cột `all_qos_mean` |
| Bảng 4.5 — Độ trễ ra quyết định (trung vị) | `tables/TABLE_V6_SIX_METHOD_CPU_LATENCY.csv`, cột `solve_ms_median` |
| Hình 4.1 — Diễn biến trên tập xác thực | `tables/TABLE_V6_VALIDATION_CURVES.csv` |
| Hình 4.2 — Tổng tốc độ theo N | `tables/TABLE_V6_SIX_METHOD_PERFORMANCE.csv` |
| Hình 4.3 — Xác suất đạt QoS | cùng tệp, cột `all_qos_mean` |
| Hình 4.4, 4.5 — Độ trễ và đánh đổi tốc độ–độ trễ | `tables/TABLE_V6_SIX_METHOD_CPU_LATENCY.csv` |
| Mục 4.12 — Kiểm định thống kê | `tables/TABLE_V6_SIX_METHOD_PAIRED_TESTS_HOLM.csv` |
| Mục 3.5.3 — Điểm lưu được chọn | `tables/TABLE_V6_CHECKPOINT_AUDIT.csv` |
| Dữ liệu thô AO–SCA, AO–Grid, AnalyticalRIS | `../six_method_v2/raw/TRADITIONAL_TEST_RAW_ALL.csv` |

Luận văn báo cáo độ trễ bằng **trung vị** (ví dụ AO–SCA tại `N = 16`: trung vị `830,45` ms, trung bình `755,34` ms).

## Cảnh báo

Thư mục `results/physical_v6_full/` (không có hậu tố `_r2`) trên nhánh
`experiment/physical-v6-full` là một lần chạy **trước đó** và cho số liệu khác
hẳn — ví dụ DDPG tại `N = 32` chỉ đạt `4,53` thay vì `16,85`. Không trích dẫn
thư mục đó.
