# REVIEW ĐỘC LẬP BA HÌNH CHƯƠNG 4 — BẢN SỬA

**Phạm vi review:** commit `242712d8e73e9f034565494b3cd2ce0e218b09dc` và các nguồn dữ liệu đã đồng bộ từ `results/six_method_v1/`.

## Kết luận

| Hình | Kết luận | Nhận xét chính |
|---|---|---|
| 4.1 — Hội tụ TD3/DDPG/PPO | **ĐẠT** | Đã thay đường TD3 cũ bằng dữ liệu xác thực của ba thuật toán, cùng protocol `six_method_v1`, trung bình 8 seed và có dải độ lệch chuẩn. Nội dung thể hiện được TD3 ổn định, DDPG mất ổn định khi N lớn và PPO chưa vào miền khả thi. |
| 4.2 — So sánh sáu phương pháp | **ĐẠT CÓ ĐIỀU KIỆN** | Kích thước chữ đã được xử lý bằng bố cục trang ngang. Dữ liệu đúng nguồn audit. Tuy nhiên bốn biểu đồ nhúng vẫn còn nhãn trục tiếng Anh, chưa thỏa yêu cầu “toàn bộ text trong ảnh dùng tiếng Việt, trừ tên”. |
| 4.3 — Chất lượng, QoS và độ trễ | **ĐẠT** | Đã loại bỏ panel speedup của bundle bốn phương pháp cũ. Cả ba panel được tính từ `TABLE_SIX_METHOD_CPU_LATENCY.csv` và `TABLE_SIX_METHOD_PERFORMANCE.csv` của cùng bundle sáu phương pháp. Các tỷ lệ AO-SCA/TD3 khớp 880, 1.438, 2.121, 2.860 và 3.038 lần. |

## Review Hình 4.1

Hình 4.1 đã giải quyết đúng lỗi khoa học quan trọng nhất của bản trước: không còn dùng đường huấn luyện TD3 từ bundle cũ. Ba panel hiện sử dụng dữ liệu xác thực của TD3, DDPG và PPO từ cùng benchmark sáu phương pháp.

Bố cục N=32 và N=128 phù hợp vì:

- N=32 cho thấy DDPG vẫn cạnh tranh và không bị TD3 thống trị trên mọi chỉ số;
- N=128 cho thấy khác biệt rõ về khả năng mở rộng và độ ổn định;
- panel toàn bộ UE đạt QoS giúp người đọc thấy chất lượng nghiệm chứ không chỉ tổng tốc độ.

Không nên đổi chú thích thành “đường huấn luyện” nếu dữ liệu bên trong là `VALIDATION_RAW.csv`. Caption nên dùng cụm “quá trình xác thực trong huấn luyện” hoặc “diễn biến chỉ số trên tập xác thực”.

## Review Hình 4.2

### Phần đã đạt

- Bố cục ngang 43,2 × 28,7 cm phù hợp để chèn trên trang landscape.
- Khi đặt khoảng 24–25 cm, chữ đạt khoảng 10 pt và có thể đọc khi in.
- Bốn panel đều đến từ bundle sáu phương pháp đã audit PASS.
- Không còn vấn đề thiếu DDPG/PPO hoặc trộn protocol.

### Điểm chưa đạt hoàn toàn

Các nhãn trục trong bốn biểu đồ nhúng vẫn bằng tiếng Anh. Đây là vi phạm trực tiếp quy tắc hình đã chốt với tác giả:

> Toàn bộ text trong ảnh dùng tiếng Việt, ngoại trừ tên thuật toán, tên công nghệ, từ viết tắt và ký hiệu.

Vì vậy, Hình 4.2 chỉ được xem là **đạt có điều kiện** cho đến khi bốn panel được vẽ lại từ:

`results/six_method_v1/tables/TABLE_SIX_METHOD_PERFORMANCE.csv`

Nhãn đề xuất:

- `Số phần tử STAR-RIS, N`;
- `Tổng tốc độ trung bình`;
- `Tỷ lệ người dùng đạt QoS`;
- `Xác suất toàn bộ người dùng đạt QoS`;
- `Mức vi phạm QoS trung bình`.

Không thay đổi dữ liệu, khoảng tin cậy, thứ tự phương pháp hoặc màu đã quy ước. Sau khi vẽ lại, vẫn xuất Hình 4.2 ở khổ ngang và chèn bằng `pdflscape`.

## Review Hình 4.3

Hình 4.3 đã đạt yêu cầu provenance. Cả ba panel hiện dùng cùng một bundle `six_method_v1`. Phép tính tỷ lệ độ trễ dùng `solve_ms_mean`, thống nhất với báo cáo reviewer và bảng latency.

Thông điệp khoa học hợp lệ:

- AO-SCA đạt tổng tốc độ cao nhất nhưng có độ trễ lớn;
- TD3 cung cấp trade-off tốt nhất trong nhóm DRL và giữ QoS gần 1;
- AnalyticalRIS nhanh nhất về thời gian tính nhưng không phải bộ tối ưu QoS khả thi;
- TD3 không được gọi là nhanh nhất tuyệt đối hoặc có tổng tốc độ cao nhất.

## Yêu cầu chèn LaTeX

Hình 4.2 bắt buộc đặt trên trang ngang:

```latex
\begin{landscape}
\begin{figure}[p]
  \centering
  \includegraphics[width=0.95\linewidth]{figures/pdf/chapter4_fig02_six_method_comparison.pdf}
  \caption{So sánh hiệu năng của sáu phương pháp trên cùng tập kiểm thử khóa.}
  \label{fig:six-method-comparison}
\end{figure}
\end{landscape}
```

Hình 4.1 và 4.3 có thể đặt dọc bằng `width=\textwidth`.

## Quyết định cuối

- Hình 4.1: được phép chèn vào bản luận văn.
- Hình 4.3: được phép chèn vào bản luận văn.
- Hình 4.2: chưa nên khóa bản nộp cuối cho đến khi chuyển nhãn trục sang tiếng Việt; về dữ liệu, kích thước và bố cục thì đã đạt.
