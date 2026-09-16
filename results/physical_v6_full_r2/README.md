# Bộ kết quả dùng cho luận văn

**Đây là bộ kết quả duy nhất được trích dẫn trong luận văn.** Mọi số liệu ở
Chương 4 đều lấy từ thư mục này.

## Nguồn

- Nhánh sinh dữ liệu: `experiment/physical-v6-full`, commit `fde2fe7`
- Tham số hóa hành động: `physical_v6_soft_anchor`
- Cấu hình: `configs/v3/pilot_v6_soft_anchor_n{16,32,64,96,128}.yaml`

## Giao thức

- Kích thước STAR–RIS: `N = 16, 32, 64, 96, 128`
- Ngân sách huấn luyện: `100.000` tương tác mỗi hạt giống
- Số hạt giống: 5 cho mỗi thuật toán và mỗi `N`
- Ngân hàng kịch bản: huấn luyện `10.000` (seed 11001), xác thực `1.000`
  (seed 22001), kiểm thử `1.000` (seed 33001)
- Thuật toán: TD3, DDPG, PPO, AO–SCA, AO–Grid, AnalyticalRIS

## Ánh xạ sang bảng và hình trong luận văn

| Nội dung luận văn | Tệp trong thư mục này |
|---|---|
| Bảng 4.3 — Tổng tốc độ trung bình | `tables/TABLE_V6_SIX_METHOD_PERFORMANCE.csv`, cột `sum_rate_mean` |
| Bảng 4.2 — Độ lệch chuẩn giữa các hạt giống | cùng tệp, cột `sum_rate_std` |
| Bảng 4.4 — Xác suất toàn bộ UE đạt QoS | cùng tệp, cột `all_qos_mean` |
| Bảng 4.5 — Độ trễ ra quyết định | `tables/TABLE_V6_SIX_METHOD_CPU_LATENCY.csv`, **cột `solve_ms_median`** |
| Hình 4.1 — Tổng tốc độ theo N | `figures_vi/fig01_v6_six_method_sum_rate.png` |
| Hình 4.2 — Xác suất đạt QoS | `figures_vi/fig07_v6_qos.png` |
| Hình 4.3 — Độ trễ theo N | `figures_vi/fig04_v6_six_method_cpu_latency.png` |
| Hình 4.4 — Đánh đổi tốc độ–độ trễ | `figures_vi/fig05_v6_quality_vs_latency.png` |
| (chưa dùng) Diễn biến trên tập xác thực | `figures_vi/fig06_v6_learning_curves.png`, dữ liệu `tables/TABLE_V6_VALIDATION_CURVES.csv` |

**Lưu ý về Bảng 4.5:** luận văn báo cáo giá trị **trung vị**, không phải trung
bình. Với AO–SCA tại `N = 16`, trung vị là `830,45` ms còn trung bình là
`755,34` ms (độ lệch chuẩn `243` ms). Tiêu đề bảng trong luận văn cần ghi rõ
điều này.

## Cảnh báo

Thư mục `results/physical_v6_full/` (không có hậu tố `_r2`) trên nhánh
`experiment/physical-v6-full` là một lần chạy **trước đó** và cho số liệu khác
hẳn — ví dụ DDPG tại `N = 32` chỉ đạt `4,53` thay vì `16,85`. Không trích dẫn
thư mục đó.
