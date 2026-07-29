# BÁO CÁO VERIFY BỘ HÌNH LUẬN VĂN 4 CHƯƠNG

Đối chiếu với `thesis/DRAWIO_PROMPTS_THESIS_4_CHAPTERS.md`.

**Công cụ:** draw.io desktop CLI 31.0.2 · skill `drawio-skill` 2.1.0 · `validate.py` (lint cấu trúc) · vision self-check từng hình.

**Kết luận chung: 11/12 hình ĐẠT. Hình 4.2 đạt về kỹ thuật nhưng cần đọc kèm lưu ý ở §5.**

---

## 1. Chuẩn kích thước cho in A4

PDF do draw.io xuất ra ≈ **0,73 pt cho mỗi đơn vị canvas**. Muốn nhãn 18 pt in ra ≥ 10 pt khi chèn rộng 16 cm (khổ chữ A4) thì canvas phải **≤ 1100 đơn vị**. Toàn bộ 12 hình dùng canvas rộng 1100 và giữ nguyên cỡ chữ theo §2.2 của prompt.

Đo lại trên PDF đã xuất:

| Cỡ chữ trong draw.io | Vai trò | In ra khi chèn rộng 16 cm |
|---|---|---|
| 24 pt | Tiêu đề vùng | **13,7 pt** |
| 18 pt | Nhãn thiết bị | **10,3 pt** |
| 16 pt | Chú thích | **9,2 pt** |

- Số hình có nhãn 18 pt in ra < 9 pt: **0**
- Số hình cao quá 23,5 cm khi chèn rộng 16 cm: **0** (cao nhất 23,0 cm — vừa một trang A4)

---

## 2. Kiểm kê file

`thesis/figures/` — **48 file, đủ 12 hình × 4 định dạng**:

| Thư mục | Số file | Ghi chú |
|---|---|---|
| `drawio/chapter1..4/` | 12 | file nguồn, mở và chỉnh sửa được |
| `pdf/` | 12 | **vector, file chính chèn LaTeX** |
| `svg/` | 12 | nhúng XML (`-e`) |
| `png/` | 24 | 12 bản xem (2200 px) + 12 bản `.drawio.png` nhúng XML |

PNG chính đều **2200 px** chiều ngang → đạt yêu cầu "tối thiểu 2000 px".

Mọi PNG nhúng XML đã chạy `repair_png.py` (vá lỗi IEND bị cắt của draw.io CLI).

---

## 3. Bảng verify từng hình

`validate.py`: **0 error** trên cả 12 hình. Cột "TV" = toàn bộ câu chữ tiếng Việt (chỉ giữ tên thuật toán / công nghệ / viết tắt / ký hiệu).

| Hình | Tên file | PDF | Cấu trúc | Chữ ≠ mũi tên | TV | Kết luận |
|---|---|---|---|---|---|---|
| 1.1 | `chapter1_fig01_oma_sdma_noma_rsma` | 111 KB | 0 err, 0 crossing, 0 through | ✔ | ✔ | **Đạt** |
| 1.2 | `chapter1_fig02_ris_vs_star_ris` | 82 KB | 0 err, 0 crossing, 0 through | ✔ | ✔ | **Đạt** |
| 1.3 | `chapter1_fig03_drl_control_loop` | 103 KB | 0 err, 0 crossing, 0 through | ✔ | ✔ | **Đạt** |
| 2.1 | `chapter2_fig01_system_model_3d` | 77 KB | 0 err, 0 crossing, 0 through | ✔ | ✔ | **Đạt** |
| 2.2 | `chapter2_fig02_rsma_transmit_decode` | 97 KB | 0 err, 0 crossing, 0 through | ✔ | ✔ | **Đạt** |
| 2.3 | `chapter2_fig03_optimization_variables_constraints` | 100 KB | 0 err, 0 crossing, 0 through | ✔ | ✔ | **Đạt** |
| 3.1 | `chapter3_fig01_td3_architecture` | 76 KB | 0 err, 0 crossing, 0 through | ✔ | ✔ | **Đạt** |
| 3.2 | `chapter3_fig02_action_decoder` | 97 KB | 0 err, 0 crossing, 0 through | ✔ | ✔ | **Đạt** |
| 3.3 | `chapter3_fig03_train_validation_test_pipeline` | 102 KB | 0 err, 0 crossing, 2 through¹ | ✔ | ✔ | **Đạt** |
| 4.1 | `chapter4_fig01_training_overview` | 2,9 MB | 0 err, 0 crossing, 0 through | ✔ | ✔ | **Đạt** |
| 4.2 | `chapter4_fig02_six_method_comparison` | 1,4 MB | 0 err, 0 crossing, 0 through | ✔ | ✔ | **Đạt**, xem §5 |
| 4.3 | `chapter4_fig03_quality_qos_latency_tradeoff` | 1,0 MB | 0 err, 0 crossing, 0 through | ✔ | ✔ | **Đạt** |

¹ Hai cảnh báo "through vertex" của Hình 3.3 là mũi tên nằm **bên trong khung nền Tầng 2** — khung này là nền trực quan, không phải icon. Không phải lỗi.

PDF Chương 4 lớn (1–3 MB) vì **nhúng ảnh kết quả thật** ở 300 dpi.

---

## 4. Kiểm tra kỹ thuật theo §3 của prompt

| Mục | Kết quả |
|---|---|
| Không có MADDPG hoặc CTDE | ✔ Chỉ dùng TD3, DDPG, PPO, AO-SCA, AO-Grid, AnalyticalRIS |
| Mô hình thực nghiệm là SISO (trừ vùng SDMA giải thích khái niệm) | ✔ Hình 2.1 ghi rõ "một anten (SISO)"; vùng SDMA ở Hình 1.1 có ghi chú in nghiêng *"Chỉ minh họa khái niệm nhiều anten — mô hình luận văn là SISO"* |
| STAR-RIS có cả phản xạ và truyền qua | ✔ Hình 1.2, 2.1 |
| Miền phản xạ bên trái, miền truyền qua bên phải | ✔ Hình 1.2, 2.1 — có dải nền và nhãn miền |
| STAR-RIS nghiêng 3D | ✔ Đo được: 26,6° (RIS 1.2), **30,1°** (STAR-RIS 1.2), **29,7°** (2.1) — đều trong 25–35° |
| RSMA có luồng chung và các luồng riêng | ✔ Hình 1.1 (vùng RSMA), 2.1, 2.2 |
| TD3 có hai Critic | ✔ Hình 3.1: Critic 1 và Critic 2 độc lập + "Lấy giá trị nhỏ hơn"; Hình 1.3 ghi "Hai Critic, cập nhật trễ" |
| Tập kiểm thử không quay lại ảnh hưởng huấn luyện | ✔ Hình 3.3 có khối cảnh báo riêng, chữ đậm |
| AO-SCA không được gọi là nghiệm tối ưu toàn cục | ✔ Không xuất hiện ở bất kỳ hình nào |

### Ngôn ngữ và trình bày

| Mục | Kết quả |
|---|---|
| Toàn bộ câu chữ tiếng Việt | ✔ Chỉ giữ OMA, SDMA, NOMA, RSMA, RIS, STAR-RIS, TD3, DDPG, PPO, AO-SCA, AO-Grid, AnalyticalRIS, BS, UE, CSI, QoS, SIC, Actor, Critic, Replay Buffer, ScenarioBank, ES/MS/TS và ký hiệu toán |
| Chữ không chồng lên mũi tên | ✔ Đã sửa 9 vi phạm phát hiện qua vision self-check (xem §6) |
| Mũi tên không xuyên qua icon | ✔ `validate.py` 0 through-vertex trên 11/12 hình |
| Không có vùng trống quá lớn | ✔ Đã sửa ở Hình 1.2 và 2.3 |
| Không dùng quá nhiều hộp chữ nhật | ✔ Thiết bị và công nghệ đều là icon: cột anten, người dùng cầm điện thoại, panel lưới nghiêng 3D, chip AI, trụ dữ liệu, bộ lọc SIC, pin, núm pha, khóa, dấu kiểm, tam giác cảnh báo, bánh răng, lưới tìm kiếm |
| PDF crop gọn | ✔ `-b 8`, không lề trắng lớn |
| PNG ≥ 2000 px | ✔ 2200 px |
| File Draw.io mở và sửa lại được | ✔ 12/12 parse sạch; SVG/PDF/PNG đều nhúng XML |

---

## 5. Lưu ý bắt buộc khi dùng hình Chương 4

### 5.1. Nguồn dữ liệu

Ba hình Chương 4 **nhúng trực tiếp ảnh kết quả đã kiểm toán từ nhánh `main`**, không vẽ lại và không hiệu chỉnh dữ liệu:

| Hình | Nguồn |
|---|---|
| 4.1 | `results/final_thesis_paper_bundle/figures/fig01_training_sum_rate`, `fig02_training_qos_fraction`, `fig03_training_violation` |
| 4.2 | `results/six_method_v1/figures/fig01_six_method_sum_rate`, `fig02_…_qos_fraction`, `fig03_…_all_qos`, `fig04_…_violation` |
| 4.3 | `results/six_method_v1/figures/fig05_…_cpu_latency`, `fig06_…_quality_latency` + `final_thesis_paper_bundle/figures/fig11_td3_speedup` |

Bộ `six_method_v1` có `SIX_METHOD_AUDIT.json` verdict **PASS**, đủ 6 phương pháp, `N ∈ {16, 32, 64, 96, 128}`, 8 seed cho nhóm DRL, 1000 kịch bản kiểm thử khóa mỗi N.

**Sai lệch có chủ ý so với prompt:** prompt Hình 4.3 chỉ định `fig10_cpu_latency` và `fig12_quality_latency_tradeoff` (bộ 4 phương pháp cũ). Tôi dùng `fig05` / `fig06` của `six_method_v1` vì đã có **đủ 6 phương pháp** — đúng tinh thần câu *"Sau khi có DDPG/PPO, chỉ bổ sung vị trí theo dữ liệu thật"*.

### 5.2. Câu chữ kết luận — chỉ dùng claim được audit cho phép

Đối chiếu `results/six_method_v1/SIX_METHOD_REVIEW.md`:

**Đã dùng (allowed):**
- "AO-SCA đạt tổng tốc độ cao nhất trong thiết lập đã đánh giá"
- "TD3 là phương pháp DRL tốt nhất và mở rộng tốt nhất trong ba phương pháp học"
- "DDPG mất bền vững QoS khi N lớn; PPO chưa vào được vùng khả thi"
- "TD3 giữ khoảng 63–71% tổng tốc độ của AO-SCA với 0,25–0,34 ms mỗi quyết định"
- "AnalyticalRIS có độ trễ thấp nhất nhưng tỷ lệ QoS bằng 0, nên không phải bộ tối ưu khả thi"

**Đã tránh (forbidden):** "TD3 tốt nhất" / "TD3 đạt sum-rate cao nhất" · "TD3 vượt DDPG ở mọi chỉ số" (tại `N = 32` DDPG có QoS nhích hơn và độ trễ thấp hơn) · "AnalyticalRIS tốt nhất vì nhanh nhất" · "AO-SCA là tối ưu toàn cục" · "TD3 nhanh nhất".

### 5.3. Cỡ chữ bên trong biểu đồ nhúng — cần đặt hình lớn

Đây là hạn chế còn lại và **chưa khắc phục được trong khuôn khổ một hình A4**:

- Biểu đồ đặt **full width** trong hình (890 đơn vị ≈ 13 cm khi in) → chữ trục **đọc được**: Hình 4.1 (vi phạm QoS), Hình 4.3 (không gian chất lượng–độ trễ).
- Biểu đồ đặt **nửa width** (525 đơn vị ≈ 7,6 cm khi in) → chữ trục **nhỏ, khoảng 4–5 pt**: 2 ô trên của Hình 4.1, cả 4 ô của Hình 4.2, 2 ô trên của Hình 4.3.

Nguyên nhân là bản chất: prompt yêu cầu ghép 3–4 biểu đồ vào **một** hình, mà mỗi biểu đồ gốc rộng 20 cm ở 300 dpi.

**Khuyến nghị:** đặt Hình 4.2 **kín một trang** (`\begin{figure}[p]` + `width=\textwidth`), hoặc tách thành 4 hình con nếu cần đọc số trên trục. Độ phân giải không phải vấn đề — ảnh nguồn 300 dpi, ở 7,6 cm vẫn tương đương ~300 dpi khi in.

---

## 6. Lỗi đã phát hiện và sửa trong quá trình làm

Vision self-check bắt được các lỗi mà `validate.py` không thấy:

| # | Lỗi | Hình | Cách sửa |
|---|---|---|---|
| 1 | Nhãn "Trạm gốc" bị mũi tên xuyên qua | 1.1 (4 vùng), 2.1, 3.1 | Chuyển nhãn lên **trên** icon |
| 2 | 6 mũi tên RSMA cắt chéo nhau thành mớ rối | 1.1 | Bỏ chip trung gian, phát 6 tia trực tiếp theo thứ tự điểm ra đơn điệu → 0 crossing |
| 3 | Icon `aws3.shield` render thành khối đỏ vô hình dạng; `mockup.barChart` thành mấy vạch ngang; pin điện quá mảnh | 1.3, 2.3 | Thay bằng icon dựng từ hình cơ bản (tam giác cảnh báo, cột tăng dần, pin có vạch dung lượng) |
| 4 | Ghi chú bị đường trực tiếp nét đứt cắt qua | 2.1 | Dời ghi chú và hộp β ra khỏi hành lang tia |
| 5 | 4 đường nét đứt cùng xuất phát một điểm tạo nêm trông như mũi tên ngược | 2.1 | Tách điểm ra trên chu vi BS |
| 6 | Đường "truyền qua kênh vô tuyến" cắt qua tiêu đề "Phía thu" | 2.2 | Dịch tiêu đề sang phải |
| 7 | Luồng Tầng 2 đi xuyên qua hộp "Bộ giải mã hành động" | 3.3 | Sắp lại Tầng 2 thành 2 cột |
| 8 | Vùng trống lớn ở giữa hình | 1.2, 2.3 | Mở rộng vùng minh họa / nối mục tiêu xuống hệ thống bằng mũi tên |
| 9 | Chữ in ra chỉ 4,4–5,8 pt ở khổ A4 | cả 7 hình bản đầu | Dựng lại toàn bộ theo chuẩn canvas ≤ 1100 (§1) |

### Hai lỗi công cụ đáng ghi nhận

1. **`id="at"` bị draw.io xử lý đặc biệt.** Mọi mũi tên nối tới cell có id `at` **im lặng không render**, trong khi `validate.py` vẫn báo 0 error và cell vẫn tồn tại trong XML. Đã đổi thành `atn`. → Tránh id quá ngắn.

2. **Ảnh nhúng phải dùng `image=data:image/png,<base64>`, KHÔNG dùng `;base64`.** Dấu `;` là ký tự phân tách style của draw.io nên `data:image/png;base64,…` làm style bị cắt và ảnh không render (hình vẫn xuất ra bình thường, chỉ thiếu ảnh).

---

## 7. Cách chèn vào LaTeX

```latex
% Hình thường (Chương 1–3)
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/pdf/chapter2_fig01_system_model_3d.pdf}
  \caption{Mô hình hệ thống SISO STAR-RIS hỗ trợ RSMA.}
  \label{fig:system-model}
\end{figure}

% Hình Chương 4 — nên đặt kín trang để đọc được chữ trên trục
\begin{figure}[p]
  \centering
  \includegraphics[width=\textwidth]{figures/pdf/chapter4_fig02_six_method_comparison.pdf}
  \caption{So sánh sáu phương pháp trên cùng tập kiểm thử khóa.}
  \label{fig:six-method}
\end{figure}
```

Trong hình **không có caption** (đúng §2.6) — caption viết ở phía LaTeX.

---

## 8. Việc còn lại

- [ ] Đặt Hình 4.2 kín một trang, hoặc tách 4 hình con nếu cần đọc số trên trục (§5.3).
- [ ] Nếu chỉnh sửa hình: mở file trong `thesis/figures/drawio/`, sửa, rồi xuất lại đủ 4 định dạng (`-e` cho svg/pdf/png, chạy `repair_png.py` sau khi xuất PNG có `-e`).
- [ ] Giữ canvas ≤ 1100 đơn vị khi thêm nội dung, nếu không cỡ chữ in A4 sẽ hụt trở lại.
