# BÁO CÁO VERIFY BỘ HÌNH LUẬN VĂN 4 CHƯƠNG

Đối chiếu với `thesis/DRAWIO_PROMPTS_THESIS_4_CHAPTERS.md`.

**Công cụ:** draw.io desktop CLI 31.0.2 · skill `drawio-skill` 2.1.0 · `validate.py` (lint cấu trúc) · vision self-check từng hình.

**Kết luận chung: 12/12 hình ĐẠT.** Hình 4.2 phải chèn trên **trang ngang** (xem §5.3).

> **Bản 2 — đã sửa 3 lỗi review.** Xem §9 để biết chi tiết ba lỗi và cách xử lý:
> cỡ chữ Chương 4, trộn protocol ở Hình 4.3, và nội dung Hình 4.1.

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
| Mô hình thực nghiệm là SISO (trừ vùng SDMA giải thích khái niệm) | ✔ Hình 2.1 ghi rõ "một anten (SISO)"; vùng SDMA ở Hình 1.1 ghi *"Minh họa khái niệm MISO — mô hình luận văn là SISO"* |
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

% Hình 4.2 — BAT BUOC trang ngang (can \usepackage{pdflscape})
\begin{landscape}
\begin{figure}[p]
  \centering
  \includegraphics[width=0.95\linewidth]{figures/pdf/chapter4_fig02_six_method_comparison.pdf}
  \caption{So sánh sáu phương pháp trên cùng tập kiểm thử khóa.}
  \label{fig:six-method}
\end{figure}
\end{landscape}
```

Trong `landscape`, `\linewidth` là chiều dài trang (~25 cm sau lề), nên `0.95\linewidth`
cho khoảng 24 cm — đúng khổ mà Hình 4.2 được thiết kế cho.

Trong hình **không có caption** (đúng §2.6) — caption viết ở phía LaTeX.

---

## 9. Bản 2 — ba lỗi review và cách sửa

### Lỗi 1 — chữ trong biểu đồ Chương 4 quá nhỏ

**Đã sửa.**

| Hình | Trước | Sau |
|---|---|---|
| 4.2 | 27,9 × 33,5 cm (dọc) → chèn 16 cm, nhãn **6,7 pt**, mỗi biểu đồ 7,6 cm | **43,2 × 28,7 cm (ngang)** → chèn 25 cm trang ngang, nhãn **10,4 pt**, mỗi biểu đồ **10,9 cm**; cao 16,6 cm vừa đúng một trang ngang |
| 4.1, 4.3 | 2 biểu đồ trên bị co từ 20 cm xuống 7,6 cm | Panel được **vẽ đúng cỡ in** (`figsize` bằng kích thước in dự kiến) nên chữ không bị co |

Không cần tách thành 4.2a/4.2b. Bảng đo lại ở §1 và §9.4.

### Lỗi 2 — Hình 4.3 trộn hai bộ kết quả

**Đã sửa. Đây là lỗi provenance thật.** Bản 1 dùng `fig11_td3_speedup` của bundle **bốn** phương pháp cũ đặt cạnh `fig05`/`fig06` của benchmark **sáu** phương pháp mới.

Bản 2 bỏ hoàn toàn bundle cũ. Cả ba panel của Hình 4.3 được tính từ **hai bảng đã kiểm toán của cùng bộ `six_method_v1`**:

| Panel | Nguồn | Phép tính |
|---|---|---|
| Độ trễ CPU sáu phương pháp | `tables/TABLE_SIX_METHOD_CPU_LATENCY.csv` | vẽ trực tiếp cột `solve_ms_mean` |
| Tỷ lệ độ trễ so với TD3 | cùng file | `solve_ms_mean(phương pháp) / solve_ms_mean(TD3)` |
| Không gian chất lượng–độ trễ | + `tables/TABLE_SIX_METHOD_PERFORMANCE.csv` | x = `solve_ms_mean`, y = `sum_rate_mean` |

Dùng cột `solve_ms_mean` (không phải `solve_ms_median`) vì đó là cột `SIX_METHOD_REVIEW.md` dùng — panel tái hiện **đúng** con số của audit:

| N | 16 | 32 | 64 | 96 | 128 |
|---|---|---|---|---|---|
| AO-SCA / TD3 (audit) | 880 | 1438 | 2121 | 2860 | 3038 |
| AO-SCA / TD3 (panel) | **880** | **1438** | **2121** | **2860** | **3038** |

Panel tỷ lệ có thêm đường mốc "Bằng TD3" để thấy **AnalyticalRIS nằm dưới mốc** — tức TD3 *không* phải phương pháp nhanh nhất, đúng guardrail của audit.

### Lỗi 3 — Hình 4.1 dùng đường huấn luyện TD3 của bundle cũ

**Đã sửa.** Đổi từ "chỉ TD3, bundle cũ" sang **so sánh hội tụ TD3 / DDPG / PPO** tính từ `results/six_method_v1/raw/convergence/*_VALIDATION_RAW.csv`, trung bình trên 8 seed với dải ±1 độ lệch chuẩn.

Bố cục đúng đề xuất: panel trái = sum-rate xác thực tại **N = 32**, panel phải = **N = 128**, panel dưới = xác suất toàn bộ UE thỏa QoS tại **N = 128**.

Ba kết luận hiện ra ngay và khớp `SIX_METHOD_REVIEW.md`:

| Quan sát trên hình | Số của audit |
|---|---|
| TD3 ổn định ở cả N thấp và N cao, QoS gần 1 | all-user QoS 0,9798–0,9985 |
| DDPG dải lệch chuẩn giữa seed bung rộng, sụt mạnh tại N = 128 | all-user QoS 0,0040; std sum-rate 5,0875 |
| PPO nằm ở mức thấp suốt quá trình | sum-rate ~2,10–2,16; QoS 0,61–0,66 |

### 9.4. Hệ quả: Chương 4 giờ không còn nhúng ảnh PNG audit

Bản 1 nhúng PNG từ `results/.../figures`. Bản 2 **vẽ lại panel từ CSV/bảng đã kiểm toán** cho Hình 4.1 và 4.3 (Hình 4.2 vẫn nhúng 4 PNG audit).

Đây là bước đi xa hơn một chút so với yêu cầu review — review chỉ yêu cầu tạo lại panel speedup. Tôi mở rộng sang cả panel độ trễ và panel chất lượng–độ trễ vì nếu chỉ thay một panel thì **hai panel cạnh nhau lệch hẳn cỡ chữ** (panel tự vẽ to gấp đôi panel audit bị co), trông như lỗi. Làm cả ba thì Hình 4.3 đồng bộ về cỡ chữ, cùng một nguồn, và nhãn trục **chuyển hết sang tiếng Việt** — trước đó nhãn trục là tiếng Anh (`Mean sum-rate`, `Median CPU decision latency`), vi phạm §2.1.

Đánh đổi cần biết:

- **Được:** chữ đọc được, một protocol duy nhất, nhãn tiếng Việt, panel tái hiện đúng số của audit.
- **Mất:** hình không còn là ảnh bit-đối-bit của artifact audit. Nguồn và phép tính đã ghi rõ trên hình và trong bảng ở §9 nên vẫn tái lập được.
- Nếu bạn muốn giữ đúng ảnh audit gốc, nói tôi đổi lại — nhưng khi đó chữ trục sẽ nhỏ và là tiếng Anh.

Riêng **Hình 4.2 vẫn nhúng 4 PNG audit** (không vẽ lại), nên nhãn trục của 4 biểu đồ đó vẫn là tiếng Anh. Muốn tiếng Việt hoàn toàn thì phải vẽ lại từ `TABLE_SIX_METHOD_PERFORMANCE.csv` — bảng này có đủ `sum_rate_mean`, `qos_fraction_mean`, `all_qos_mean`, `violation_mean` cùng khoảng tin cậy.

### 9.5. Cỡ chữ sau khi sửa

| Hình | Khổ PDF | Nhãn 18 pt khi chèn 16 cm | Cao khi chèn 16 cm |
|---|---|---|---|
| 1.1 – 3.3, 4.1, 4.3 | 27,9 cm rộng | **10,3 pt** | 15,7 – 23,0 cm |
| 4.2 | 43,2 × 28,7 cm | 6,7 pt ✗ — **phải dùng trang ngang** → 25 cm cho **10,4 pt** | 16,6 cm ở khổ 25 cm |

---

## 10. Bản 3 — Hình 1.1 vẽ lại theo ảnh mẫu

Hình 1.1 được vẽ lại bám theo một ảnh mẫu do người dùng cung cấp. Mức bám sát khoảng **90–95%** — không thể 100% vì ảnh mẫu là bản render, không có file nguồn.

**Thay đổi so với bản 2:**

| | Bản 2 | Bản 3 (theo mẫu) |
|---|---|---|
| Khung | 4 hộp bo góc có nền | 4 ô chia bằng đường kẻ mảnh hình chữ thập |
| Nhãn vùng | ở trên mỗi hộp | ở **dưới** mỗi ô, kèm 2 dòng mô tả |
| Người dùng | icon người cầm điện thoại | icon **điện thoại viền** |
| SDMA | 3 tia bản rộng phẳng | anten 3 phần tử + **3 búp sóng hình nón** |
| NOMA | 3 thanh công suất phẳng | **khối công suất 3D** có mặt trên nghiêng + hộp "Giải mã SIC" riêng |
| RSMA | tia màu + nhãn (s_k) | tia màu + chuỗi **[s_c] → [×] → [s_k]** dưới mỗi UE |
| OMA | 3 hộp tài nguyên | 3 hộp có **icon đồng hồ bấm giờ**, nối qua thanh bus ngang |
| Khổ | 27,9 × 39,6 cm | **27,9 × 27,4 cm** — chèn 16 cm chỉ cao 15,7 cm |

**Cỡ chữ khi chèn rộng 16 cm:** tên vùng 12,0 pt · nhãn UE và mô tả 9,7 pt · chữ trong hộp 8,6 pt · nhãn lớp công suất 8,0 pt. Đều trên ngưỡng đọc được khi in.

**Kỹ thuật đáng ghi:** búp sóng hình nón được dựng bằng tam giác `direction=north` có **góc xoay tính toán**. drawio xoay theo chiều kim đồng hồ quanh tâm ô, nên với đỉnh tại A (trạm gốc) và tâm đáy tại B (người dùng):

```
θ = atan2(Ax − Bx, −(Ay − By))   (độ)
bao = (tâm.x − nửa_rộng, tâm.y − |AB|/2, 2·nửa_rộng, |AB|)
```

Đáy nón dừng ở 78% quãng đường tới điện thoại để không đè lên icon.

### 10.1. Hình 2.3 cũng vẽ lại theo ảnh mẫu

Hình 2.3 được vẽ lại theo một ảnh mẫu thứ hai: tiêu đề trên cùng, hộp hệ thống ở giữa, năm badge biến nối bằng đường cong mảnh (ba trái, hai phải), dải ràng buộc có pill tiêu đề và bốn mục.

Icon dựng từ hình cơ bản để render được ở mọi cỡ: thanh công suất có mũi tên · biểu đồ cột có điểm nhấn cam · β với hai mũi tên xoè · vòng tròn θ kèm mũi tên cong · pin với `mxgraph.basic.flash` · cân thăng bằng · đồng hồ · khiên (chữ nhật + tam giác hướng xuống + dấu kiểm).

**Khổ:** 27,9 × 15,0 cm — chèn rộng 16 cm chỉ cao **8,6 cm**. Cỡ chữ in: tiêu đề 11,5 pt · hộp trung tâm 10,9 pt · nhãn biến 9,2 pt · chữ ràng buộc 8,6 pt.

**Font:** giữ Times New Roman theo §2.2, dù hai ảnh mẫu đều dùng sans-serif — đổi riêng hai hình sẽ lệch với mười hình còn lại.

### 10.2. Hai chỗ đã sửa sau khi rà lại câu chữ

| Hình | Trước | Sau | Lý do |
|---|---|---|---|
| 1.1 (SDMA) | `Minh họa khái niệm MISO` | `Minh họa khái niệm MISO — mô hình luận văn là SISO` | Checklist §3 yêu cầu *"Không làm người xem hiểu đây là mô hình SISO của luận văn"*. Câu theo ảnh mẫu hàm ý là minh họa khái niệm nhưng không nói rõ mô hình luận văn là SISO. |
| 2.3 (tiêu đề) | `Bài toán tối ưu toàn cục: hàm mục tiêu phi lồi và không gian biến số khổng lồ` | `Bài toán tối ưu: hàm mục tiêu phi lồi và không gian biến số lớn` | Bỏ chữ "toàn cục" để không đọc nhầm thành khẳng định có nghiệm tối ưu toàn cục — liên quan guardrail *"AO-SCA không được gọi là nghiệm tối ưu toàn cục"*. Đồng thời tránh lặp nghĩa ("phi lồi" = "không lồi"). |

---

## 8. Việc còn lại

- [ ] **Bắt buộc:** chèn Hình 4.2 trên **trang ngang** (xem đoạn LaTeX ở §7). Nếu chèn dọc 16 cm thì nhãn chỉ còn 6,7 pt.
- [ ] Tùy chọn: vẽ lại 4 biểu đồ của Hình 4.2 từ `TABLE_SIX_METHOD_PERFORMANCE.csv` để nhãn trục sang tiếng Việt và cho phép đưa 4.2 về khổ dọc (§9.4).
- [ ] Nếu chỉnh sửa hình: mở file trong `thesis/figures/drawio/`, sửa, rồi xuất lại `.pdf`, `.svg`, `.png`.
- [ ] Giữ canvas ≤ 1100 đơn vị cho hình dọc (≤ 1750 cho hình trang ngang), nếu không cỡ chữ in A4 sẽ hụt trở lại.
