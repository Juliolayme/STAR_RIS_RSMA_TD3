# Báo cáo kiểm tra và tích hợp hình minh họa

Bộ năm hình Draw.io đã được kiểm tra tự động trước khi build luận văn. Quy trình kiểm tra xác nhận tệp nguồn Draw.io là XML hợp lệ, bốn định dạng tài sản tồn tại, PDF có header hợp lệ, các nhãn kỹ thuật bắt buộc xuất hiện và mỗi chương tham chiếu đúng tệp PDF.

| Chương | Tệp hình | Draw.io (KB) | PDF (KB) | SVG (KB) | PNG (KB) | Nhãn kỹ thuật đã kiểm tra |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `chapter1_rsma_vs_conventional` | 25.4 | 251.9 | 1315.1 | 451.9 | 7 |
| 2 | `chapter2_ris_star_ris_overview` | 17.5 | 140.7 | 609.6 | 233.0 | 7 |
| 3 | `chapter3_system_model` | 14.6 | 164.0 | 582.2 | 267.2 | 7 |
| 4 | `chapter4_td3_training_pipeline` | 15.3 | 149.3 | 938.9 | 295.7 | 7 |
| 5 | `chapter5_quality_latency_tradeoff` | 7.2 | 120.2 | 442.7 | 186.4 | 6 |

## Kết quả review học thuật

- **Chương 1:** phân biệt OMA, SDMA, NOMA và RSMA bằng biểu tượng BS/user, resource blocks, spatial beams, power layers, common/private streams và SIC. Phần văn bản làm rõ ba user chỉ là minh họa và SDMA không phải mô hình SISO của thí nghiệm.
- **Chương 2:** thể hiện đúng RIS chỉ phản xạ, STAR-RIS đồng thời truyền qua/phản xạ, có hai miền phủ sóng, công thức hệ số và ba giao thức ES/MS/TS.
- **Chương 3:** thể hiện đúng BS SISO, bốn user, đường trực tiếp và đường ghép tầng, phân nhóm reflection/transmission, action vector và observation.
- **Chương 4:** thể hiện train/validation/test tách biệt, test bị khóa, validation-only checkpoint selection, deterministic test và không có đường quay từ test về training.
- **Chương 5:** biểu diễn trade-off định tính, không ngụy tạo tọa độ tuyệt đối; AO-SCA có chất lượng cao hơn, TD3 có QoS/latency cân bằng, AnalyticalRIS không đạt QoS.

## Điều kiện chấp nhận

- [x] Mỗi chương có ít nhất một hình minh họa.
- [x] Hình được chèn từ PDF vector, không dùng ảnh raster làm nguồn chính.
- [x] Hình rộng được đặt trên trang ngang để chữ và icon có thể đọc khi in A4.
- [x] Mỗi hình có caption, label, nguồn và đoạn văn liên kết với nội dung chương.
- [x] Không có claim MADDPG/CTDE hoặc mô hình MIMO trong phần mô tả hệ thống thực nghiệm.
