# PROMPT TẠO HÌNH DRAW.IO CHO LUẬN VĂN 4 CHƯƠNG

## 1. Mục tiêu

Tạo bộ hình minh họa cho luận văn **“TỐI ƯU PHÂN BỔ TÀI NGUYÊN SỬ DỤNG HỌC TĂNG CƯỜNG SÂU TRONG MẠNG STAR-RIS HỖ TRỢ RSMA”**. Mỗi hình phải giúp người đọc nhìn vào là hiểu được thiết bị, hướng truyền tín hiệu, cách hoạt động của công nghệ hoặc thuật toán và mối liên hệ với nội dung chương.

Bộ hình dự kiến gồm 12 hình, mỗi chương 3 hình. Không tạo hình chỉ để trang trí.

---

## 2. Quy tắc thiết kế bắt buộc

### 2.1. Ngôn ngữ

- Toàn bộ câu chữ trong hình dùng **tiếng Việt**.
- Chỉ giữ nguyên tên thuật toán, tên công nghệ, từ viết tắt và ký hiệu: OMA, SDMA, NOMA, RSMA, RIS, STAR-RIS, TD3, DDPG, PPO, AO-SCA, AO-Grid, AnalyticalRIS, BS, UE, CSI, QoS, SINR, SIC, Actor, Critic, Replay Buffer, ScenarioBank, cùng các ký hiệu toán học.
- Không dùng các câu tiếng Anh như “Training pipeline”, “Common stream”, “Private stream”, “Reflection side”, “Transmission side”. Thay bằng “Quy trình huấn luyện”, “Luồng chung”, “Luồng riêng”, “Miền phản xạ”, “Miền truyền qua”.

### 2.2. Phong cách

- Phong cách học thuật, sạch, hiện đại, nền trắng hoặc rất nhạt.
- Font Times New Roman, Cambria hoặc Liberation Serif.
- Tiêu đề vùng 22–26 pt; nhãn thiết bị 18–22 pt; chú thích 16–18 pt.
- Không dùng bóng đổ mạnh, hiệu ứng hoạt hình hoặc gradient đậm.
- Chữ phải đọc được khi hình được thu nhỏ để in trên A4.

### 2.3. Dùng icon thay cho hình khối

Không dùng hình chữ nhật đơn giản làm thiết bị chính. Ưu tiên:

- Trạm gốc: icon cột anten hoặc tháp phát sóng.
- Người dùng: icon người dùng cầm điện thoại, điện thoại hoặc laptop.
- RIS/STAR-RIS: panel lưới gồm nhiều phần tử.
- TD3/DRL: icon vi mạch AI hoặc mạng nơ-ron.
- Replay Buffer/ScenarioBank: icon cơ sở dữ liệu hình trụ.
- QoS: icon khiên hoặc dấu kiểm.
- Vi phạm QoS: icon cảnh báo.
- Độ trễ: icon đồng hồ bấm giờ.
- Checkpoint: icon tệp mô hình có khóa.
- SIC: icon bộ lọc, phễu giải mã hoặc bộ thu có bước loại bỏ tín hiệu.
- Công suất: icon pin hoặc thanh mức năng lượng.
- Pha: icon núm xoay hoặc vòng tròn pha.
- Chia năng lượng: icon bộ chia thành hai nhánh.

Khung mảnh có thể dùng để nhóm nội dung, nhưng thiết bị và công nghệ chính phải thể hiện bằng icon.

### 2.4. Quy tắc STAR-RIS 3D

Áp dụng cho mọi hình có STAR-RIS:

- STAR-RIS phải là panel lưới **nghiêng theo phối cảnh 3D khoảng 25–35 độ**, không vẽ phẳng chính diện.
- Panel có độ dày nhẹ và gồm nhiều ô phần tử.
- **Miền phản xạ nằm bên trái STAR-RIS.**
- **Miền truyền qua nằm bên phải STAR-RIS.**
- Tín hiệu tới đi từ BS đến STAR-RIS.
- Tia phản xạ quay về phía trái.
- Tia truyền xuyên qua sang phía phải.
- Ghi rõ “Tia tới”, “Tia phản xạ”, “Tia truyền qua”.
- Không đặt người dùng phản xạ bên phải hoặc người dùng truyền qua bên trái.
- Không làm người xem hiểu STAR-RIS là relay khuếch đại.

### 2.5. Quy ước mũi tên

- Nét xám đứt: đường trực tiếp.
- Xanh lá: đường phản xạ.
- Cam: đường truyền qua.
- Tím: luồng chung.
- Các màu riêng: luồng riêng.
- Tia bản rộng: búp sóng không gian.
- Đường hai chiều: vòng lặp huấn luyện.
- Đường một chiều: luồng dữ liệu hoặc suy luận.
- Mỗi hình có hơn ba loại đường phải có chú giải.

### 2.6. Đầu ra

Mỗi hình xuất đủ `.drawio`, `.svg`, `.pdf`, `.png`. PDF vector là file chính chèn vào LaTeX. Không đặt caption trong hình.

---

## 3. Verify sau khi vẽ

### Kỹ thuật

- [ ] Không có MADDPG hoặc CTDE.
- [ ] Mô hình thực nghiệm là SISO, trừ vùng SDMA chỉ dùng để giải thích khái niệm.
- [ ] STAR-RIS có cả phản xạ và truyền qua.
- [ ] Miền phản xạ bên trái, miền truyền qua bên phải.
- [ ] STAR-RIS nghiêng 3D.
- [ ] RSMA có luồng chung và các luồng riêng.
- [ ] TD3 có hai Critic.
- [ ] Tập kiểm thử không quay lại ảnh hưởng huấn luyện.
- [ ] AO-SCA không được gọi là nghiệm tối ưu toàn cục.

### Ngôn ngữ và trình bày

- [ ] Toàn bộ câu chữ bằng tiếng Việt, ngoại trừ tên và ký hiệu được phép giữ nguyên.
- [ ] Chữ không chồng lên mũi tên.
- [ ] Mũi tên không xuyên qua icon.
- [ ] Không có vùng trống quá lớn.
- [ ] Không dùng quá nhiều hộp chữ nhật.
- [ ] PDF crop gọn, không có lề trắng lớn.
- [ ] PNG tối thiểu 2000 px theo chiều ngang đối với hình rộng.
- [ ] File Draw.io mở lại và chỉnh sửa được.

---

# CHƯƠNG 1 — CƠ SỞ LÝ THUYẾT VÀ CÁC CÔNG NGHỆ

## Hình 1.1 — So sánh OMA, SDMA, NOMA và RSMA

**Tên file:** `chapter1_fig01_oma_sdma_noma_rsma`

**Prompt:** Vẽ infographic ngang gồm bốn vùng OMA, SDMA, NOMA và RSMA. Mỗi vùng có một icon trạm gốc phía trên, ba icon người dùng phía dưới và đường truyền thể hiện đúng cơ chế.

- **OMA:** Ba tài nguyên trực giao có icon đồng hồ/phổ tần, mỗi tài nguyên nối bằng nét đứt đến một người dùng. Nhãn “Khe thời gian riêng”, “Dải tần riêng”. Dòng kết luận: “Mỗi người dùng sử dụng tài nguyên trực giao riêng”.
- **SDMA:** Trạm gốc nhiều anten, ba tia bản rộng hướng đến ba người dùng. Nhãn “Búp sóng không gian”. Thêm ghi chú “Minh họa khái niệm hệ đa anten”. Không làm người xem hiểu đây là mô hình SISO của luận văn.
- **NOMA:** Chỉ vẽ một tín hiệu tổng hợp từ BS gồm ba lớp công suất cao, trung bình, thấp cùng đi đến ba người dùng. Người dùng mạnh có icon SIC và trình tự “Giải mã lớp công suất lớn → loại bỏ bằng SIC → giải mã tín hiệu của mình”. Không vẽ ba beam riêng.
- **RSMA:** Một luồng tím “Luồng chung $s_c$” đến tất cả người dùng và ba luồng riêng “$s_1$, $s_2$, $s_3$”. Mỗi user có trình tự “Giải mã $s_c$ → SIC → giải mã $s_k$”. Làm vùng RSMA nổi bật nhẹ.

**Chú giải:** nét đứt = tài nguyên trực giao; tia rộng = búp sóng; tím = luồng chung; màu riêng = luồng riêng; lớp dày/mỏng = mức công suất NOMA.

**Đoạn mô tả:** Hình minh họa cách OMA, SDMA, NOMA và RSMA tổ chức tài nguyên và xử lý nhiễu. RSMA chia thông điệp thành một luồng chung và các luồng riêng, nhờ đó linh hoạt giữa giải mã một phần nhiễu và xem phần còn lại như tạp âm.

**Verify riêng:** NOMA không giống SDMA; RSMA có common và private streams; mọi user RSMA đều giải mã luồng chung; toàn bộ chữ tiếng Việt.

---

## Hình 1.2 — So sánh RIS và STAR-RIS

**Tên file:** `chapter1_fig02_ris_vs_star_ris`

**Prompt:** Chia hình thành hai nửa.

- **RIS truyền thống:** BS bên trái, panel RIS nghiêng nhẹ ở giữa, tia tới và tia phản xạ quay về phía có người dùng. Phía sau RIS ghi “Vùng phủ sóng hạn chế”. Nhãn “Chỉ phản xạ”, “Phủ sóng nửa không gian”.
- **STAR-RIS:** BS bên trái, panel STAR-RIS nghiêng 30 độ theo phối cảnh 3D. Miền phản xạ bên trái, miền truyền qua bên phải. Một tia tới từ BS, một tia phản xạ quay trái và một tia truyền xuyên qua sang phải. Đặt người dùng ở cả hai miền. Ghi “ES: chia năng lượng”, “MS: chuyển chế độ”, “TS: chuyển theo thời gian”; làm nổi bật ES. Ghi công thức nhỏ $\beta_n^T+\beta_n^R=1$.

Bên dưới có bảng ngắn so sánh “Điều khiển tín hiệu”, “Phạm vi phục vụ”, “Khả năng phục vụ hai phía”.

**Đoạn mô tả:** RIS truyền thống chủ yếu điều khiển thành phần phản xạ, còn STAR-RIS đồng thời tạo tín hiệu phản xạ và truyền qua để phục vụ người dùng ở hai phía. Luận văn sử dụng STAR-RIS thụ động theo giao thức chia năng lượng.

**Verify riêng:** STAR-RIS nghiêng 3D; miền phản xạ bên trái; miền truyền qua bên phải; đủ tia tới, phản xạ, truyền qua.

---

## Hình 1.3 — Vòng lặp học tăng cường và vị trí của TD3

**Tên file:** `chapter1_fig03_drl_control_loop`

**Prompt:** Vẽ vòng lặp bằng icon gồm:

1. “Môi trường STAR-RIS–RSMA”: icon BS, STAR-RIS, người dùng.
2. “Thông tin trạng thái kênh”: icon dữ liệu/sóng.
3. “Bộ tác tử TD3”: icon chip AI.
4. “Hành động”: icon công suất, bộ chia và núm pha; ghi “Phân bổ công suất”, “Tỷ lệ luồng chung”, “Hệ số chia năng lượng”, “Pha truyền và phản xạ”.
5. “Phần thưởng”: icon biểu đồ và khiên; ghi “Tổng tốc độ” và “Phạt vi phạm QoS”.

Mũi tên: Môi trường → Trạng thái → TD3 → Hành động → Môi trường; Môi trường → Phần thưởng → TD3.

Bên dưới đặt ba icon nhỏ: DDPG “Một Critic”; PPO “On-policy”; TD3 “Hai Critic, cập nhật trễ”. Không ghi TD3 tốt nhất.

**Đoạn mô tả:** Tác tử nhận trạng thái kênh, tạo phương án phân bổ tài nguyên và nhận phần thưởng dựa trên tổng tốc độ cùng mức đáp ứng QoS. TD3 phù hợp với không gian hành động liên tục và có cơ chế giảm sai số ước lượng.

---

# CHƯƠNG 2 — MÔ HÌNH HỆ THỐNG VÀ BÀI TOÁN TỐI ƯU

## Hình 2.1 — Mô hình hệ thống SISO STAR-RIS hỗ trợ RSMA

**Tên file:** `chapter2_fig01_system_model_3d`

**Prompt:** Vẽ sơ đồ phối cảnh 3D gồm một BS SISO bên trái, STAR-RIS nghiêng 30 độ ở giữa và bốn người dùng. U1, U3 ở bên trái thuộc miền phản xạ; U2, U4 ở bên phải thuộc miền truyền qua.

- Panel STAR-RIS có 4×6 hoặc 5×7 ô, có độ dày nhẹ, nhãn “STAR-RIS thụ động” và $\beta_n^T+\beta_n^R=1$.
- BS → STAR-RIS: xanh dương, “Kênh BS–STAR-RIS”.
- STAR-RIS → U1, U3: xanh lá, “Đường phản xạ”.
- STAR-RIS → U2, U4: cam, “Đường truyền qua”.
- BS → mỗi user: nét xám đứt, “Đường trực tiếp”.
- Bên cạnh BS có “Luồng chung $s_c$” màu tím và “Luồng riêng $s_1,s_2,s_3,s_4$”.

**Đoạn mô tả:** Hệ thống gồm một BS SISO, một STAR-RIS thụ động và bốn người dùng ở hai miền. Mỗi user nhận tín hiệu qua đường trực tiếp và đường ghép tầng đi qua STAR-RIS; BS phát một luồng chung và bốn luồng riêng theo RSMA.

**Verify riêng:** BS SISO; đủ bốn user; U1/U3 bên trái, U2/U4 bên phải; STAR-RIS nghiêng 3D; có đường trực tiếp.

---

## Hình 2.2 — Quy trình phát và giải mã tín hiệu RSMA

**Tên file:** `chapter2_fig02_rsma_transmit_decode`

**Prompt:** Vẽ sơ đồ từ trái sang phải:

- Bốn thông điệp $W_1$–$W_4$ được tách thành phần chung và phần riêng.
- Các phần chung được ghép thành “Thông điệp chung”; phần riêng giữ riêng.
- Bộ mã hóa tạo “Luồng chung $s_c$” và “Các luồng riêng $s_k$”.
- BS phát “Tín hiệu tổng hợp”, kèm công thức nhỏ $x=\sqrt{p_c}s_c+\sum_k\sqrt{p_k}s_k$.
- Ở phía thu, một user mẫu thực hiện: “Nhận tín hiệu tổng hợp → Giải mã luồng chung → Loại bỏ bằng SIC → Giải mã luồng riêng → Tốc độ người dùng”.
- Luồng chung màu tím, luồng riêng các màu khác. Làm rõ tất cả user đều giải mã luồng chung.

**Đoạn mô tả:** Mỗi thông điệp được chia thành phần chung và riêng. Tất cả người dùng giải mã luồng chung trước, loại bỏ bằng SIC rồi giải mã luồng riêng tương ứng.

---

## Hình 2.3 — Các biến tối ưu và ràng buộc

**Tên file:** `chapter2_fig03_optimization_variables_constraints`

**Prompt:** Trung tâm là icon hệ thống STAR-RIS–RSMA. Xung quanh là năm nhóm biến bằng icon:

1. Công suất: pin/thanh năng lượng; “Công suất luồng chung”, “Công suất các luồng riêng”.
2. Phân bổ tốc độ chung: biểu đồ chia tỷ lệ; “Tỷ lệ tốc độ chung cho từng người dùng”.
3. Hệ số chia năng lượng: bộ chia hai nhánh; “Hệ số $\beta^T$”.
4. Pha truyền: núm pha; “Pha truyền $\theta^T$”.
5. Pha phản xạ: núm pha; “Pha phản xạ $\theta^R$”.

Phía dưới dùng icon khóa/khiên cho ràng buộc:

- Tổng công suất không vượt giới hạn.
- Tổng tỷ lệ luồng chung bằng 1.
- $0\leq\beta^T\leq1$.
- Pha thuộc $[-\pi,\pi)$.
- Mỗi người dùng đạt tốc độ tối thiểu.

Phía trên đặt icon biểu đồ tăng: “Tối đa hóa tổng tốc độ” và “Bảo đảm QoS”.

**Đoạn mô tả:** Bộ tối ưu lựa chọn đồng thời công suất, tỷ lệ tốc độ chung, hệ số chia năng lượng và pha STAR-RIS dưới các ràng buộc vật lý, công suất và QoS.

---

# CHƯƠNG 3 — XÂY DỰNG THUẬT TOÁN TD3

## Hình 3.1 — Kiến trúc TD3

**Tên file:** `chapter3_fig01_td3_architecture`

**Prompt:** Vẽ bằng icon mạng nơ-ron và chip AI:

- Trạng thái $s_t$; Actor; hành động $a_t$; môi trường STAR-RIS–RSMA; phần thưởng.
- Critic 1 và Critic 2 độc lập.
- Actor đích, Critic đích 1, Critic đích 2.
- Replay Buffer hình trụ.

Luồng:

- Trạng thái → Actor → Hành động → Môi trường.
- Môi trường → phần thưởng và trạng thái mới.
- Transition → Replay Buffer.
- Replay Buffer → hai Critic.
- Hai Critic → “Lấy giá trị nhỏ hơn”.
- Critic → cập nhật Actor.
- Mạng chính → “Cập nhật mềm” → mạng đích.

Đặt ba biểu tượng nhỏ:

1. “Hai Critic – giảm đánh giá quá cao”.
2. “Làm trơn hành động đích”.
3. “Actor cập nhật chậm hơn Critic”.

**Đoạn mô tả:** TD3 dùng một Actor và hai Critic. Việc lấy giá trị nhỏ hơn của hai Critic, làm trơn action đích và cập nhật Actor trễ giúp quá trình học ổn định hơn.

---

## Hình 3.2 — Bộ giải mã hành động vật lý

**Tên file:** `chapter3_fig02_action_decoder`

**Prompt:** Đầu vào là icon Actor TD3 và “Vector hành động chuẩn hóa trong $[-1,1]$”. Chia thành năm nhánh:

1. Công suất → “Chiếu lên simplex công suất” → $p_c,p_1,\ldots,p_K$.
2. Tỷ lệ tốc độ chung → “Chuẩn hóa tổng bằng 1” → $\eta_1,\ldots,\eta_K$.
3. Hệ số chia năng lượng → “Ánh xạ về $[0,1]$” → $\beta^T$.
4. Pha truyền → “Ánh xạ về $[-\pi,\pi)$” → $\theta^T$.
5. Pha phản xạ → “Ánh xạ về $[-\pi,\pi)$” → $\theta^R$.

Năm nhánh hội tụ vào icon khiên “Hành động khả thi về mặt vật lý”, sau đó đi vào môi trường STAR-RIS–RSMA.

**Đoạn mô tả:** Bộ giải mã chuyển action chuẩn hóa của Actor thành các biến điều khiển hợp lệ, bảo đảm ràng buộc simplex, miền beta và miền pha trước khi tính tín hiệu và tốc độ.

---

## Hình 3.3 — Quy trình huấn luyện, xác thực và kiểm thử

**Tên file:** `chapter3_fig03_train_validation_test_pipeline`

**Prompt:** Vẽ ba tầng ngang.

### Tầng 1 — Dữ liệu kịch bản

Ba icon cơ sở dữ liệu:

- “Tập huấn luyện”.
- “Tập xác thực”.
- “Tập kiểm thử”.

Mỗi bank ghi seed, checksum, số lượng kịch bản và “Không giao nhau”. Đặt icon khóa trên tập kiểm thử.

### Tầng 2 — Huấn luyện

Icon môi trường, Actor, hai Critic, Replay Buffer, bộ giải mã action và bộ điều khiển QoS. Luồng:

- Tập huấn luyện → môi trường.
- Môi trường → trạng thái → Actor.
- Actor → bộ giải mã → môi trường.
- Môi trường → phần thưởng/vi phạm → Replay Buffer và bộ điều khiển QoS.
- Replay Buffer → hai Critic → cập nhật Actor.
- Tập xác thực chỉ đi đến “Đánh giá định kỳ” và “Chọn checkpoint tốt nhất”.

### Tầng 3 — Kiểm thử và báo cáo

Checkpoint tốt nhất + tập kiểm thử → “Suy luận xác định, không có nhiễu thăm dò” → CSV thô → thống kê → biểu đồ → đo độ trễ → báo cáo.

Đặt biểu tượng cấm từ tập kiểm thử quay về huấn luyện và ghi “Tập kiểm thử không được dùng để chọn checkpoint”.

**Đoạn mô tả:** Train, validation và test được tách độc lập. Checkpoint chỉ được chọn trên validation; test chỉ sử dụng sau khi mô hình đã khóa để tránh rò rỉ thông tin.

---

# CHƯƠNG 4 — MÔ PHỎNG VÀ ĐÁNH GIÁ

## Hình 4.1 — Quá trình huấn luyện TD3

**Tên file:** `chapter4_fig01_training_overview`

**Prompt:** Tạo bố cục tổng hợp từ ba biểu đồ thật trong repository:

- `fig01_training_sum_rate.pdf`.
- `fig02_training_qos_fraction.pdf`.
- `fig03_training_violation.pdf`.

Không vẽ lại hoặc sửa dữ liệu. Bố cục: sum-rate bên trái trên, QoS bên phải trên, violation phía dưới. Thêm dải chú thích bằng icon:

- biểu đồ tăng: “Cải thiện tổng tốc độ”;
- khiên: “Tỷ lệ QoS tiến gần 1”;
- cảnh báo giảm: “Vi phạm QoS giảm dần”.

**Đoạn mô tả:** Ba chỉ số sum-rate, QoS và violation phải được phân tích đồng thời để đánh giá quá trình học và khả năng tiến gần miền khả thi.

---

## Hình 4.2 — So sánh sáu phương pháp

**Tên file:** `chapter4_fig02_six_method_comparison`

**Chỉ tạo sau khi có kết quả khóa và kiểm toán của:** TD3, DDPG, PPO, AO-SCA, AO-Grid, AnalyticalRIS.

**Prompt:** Tạo bố cục 2×2 gồm:

1. Tổng tốc độ theo $N$.
2. Tỷ lệ người dùng thỏa QoS.
3. Xác suất toàn bộ người dùng thỏa QoS.
4. Mức vi phạm QoS.

Dùng cùng $N\in\{16,32,64,96,128\}$ và cùng test bank. Dùng icon chip AI cho TD3/DDPG/PPO; bánh răng cho AO-SCA; lưới tìm kiếm cho AO-Grid; icon căn chỉnh pha cho AnalyticalRIS. Thêm chú giải “Nhóm DRL” và “Nhóm tối ưu truyền thống”. Không ghi “TD3 tốt nhất” nếu dữ liệu không chứng minh. Không tạo số liệu giả. Nếu DDPG/PPO chưa xong, không chèn hình này vào luận văn cuối.

**Đoạn mô tả:** Hình so sánh sáu phương pháp trên cùng kịch bản kiểm thử, đồng thời xét sum-rate, QoS và violation thay vì chỉ một chỉ số.

---

## Hình 4.3 — Đánh đổi chất lượng, QoS và độ trễ

**Tên file:** `chapter4_fig03_quality_qos_latency_tradeoff`

**Prompt:** Tạo hình tổng hợp từ dữ liệu thật:

- `fig10_cpu_latency.pdf`.
- `fig11_td3_speedup.pdf`.
- `fig12_quality_latency_tradeoff.pdf`.

Bố cục: độ trễ CPU bên trái, speedup ở giữa, không gian chất lượng–độ trễ bên phải. Icon: TD3 = chip AI + khiên QoS; DDPG/PPO = chip AI; AO-SCA = bánh răng + đồng hồ; AO-Grid = lưới; AnalyticalRIS = căn chỉnh pha + đồng hồ nhanh.

Không ghi “TD3 nhanh nhất”, “TD3 có chất lượng tốt nhất” hoặc “AO-SCA là tối ưu toàn cục”. Dùng các thông điệp:

- “AO-SCA đạt tổng tốc độ cao nhưng độ trễ lớn”.
- “AnalyticalRIS có độ trễ thấp nhưng cần đánh giá QoS”.
- “TD3 hướng đến cân bằng giữa chất lượng, QoS và độ trễ”.

Sau khi có DDPG/PPO, chỉ bổ sung vị trí theo dữ liệu thật.

**Đoạn mô tả:** Một phương pháp không thể chỉ được đánh giá bằng tổng tốc độ. Cần xem đồng thời chất lượng nghiệm, khả năng đáp ứng QoS và thời gian ra quyết định.

---

## 4. Prompt tổng quát giao cho skill Draw.io

> Hãy tạo hình minh họa học thuật bằng Draw.io cho luận văn kỹ thuật. Sử dụng icon thiết bị và icon công nghệ thay cho hình chữ nhật đơn giản. Toàn bộ câu chữ trong hình dùng tiếng Việt, ngoại trừ tên thuật toán, tên công nghệ, từ viết tắt và ký hiệu toán học. Trạm gốc dùng icon anten; người dùng dùng icon người hoặc điện thoại; STAR-RIS dùng panel lưới nghiêng theo phối cảnh 3D. Với mọi hình có STAR-RIS, miền phản xạ phải ở bên trái và miền truyền qua ở bên phải; tín hiệu tới đi từ BS đến STAR-RIS, sau đó tách thành tia phản xạ quay về trái và tia truyền xuyên qua phải. Hình phải có bố cục rõ, mũi tên đúng ý nghĩa, chữ không chồng lên đường truyền, phù hợp in A4 và xuất đủ Draw.io, SVG, PDF, PNG. Sau khi vẽ, kiểm tra kỹ thuật, ngôn ngữ, bố cục và khả năng chèn LaTeX trước khi kết luận hoàn thành.

---

## 5. Quy ước tên file

```text
chapter1_fig01_oma_sdma_noma_rsma.*
chapter1_fig02_ris_vs_star_ris.*
chapter1_fig03_drl_control_loop.*
chapter2_fig01_system_model_3d.*
chapter2_fig02_rsma_transmit_decode.*
chapter2_fig03_optimization_variables_constraints.*
chapter3_fig01_td3_architecture.*
chapter3_fig02_action_decoder.*
chapter3_fig03_train_validation_test_pipeline.*
chapter4_fig01_training_overview.*
chapter4_fig02_six_method_comparison.*
chapter4_fig03_quality_qos_latency_tradeoff.*
```

## 6. Cấu trúc thư mục

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

## 7. Báo cáo verify

Sau khi hoàn thành, tạo `thesis/FIGURE_REVIEW_4_CHAPTERS.md`, ghi tên hình, mục đích, đường dẫn, kích thước PDF, trạng thái mở file, khả năng đọc khi in A4, kiểm tra tiếng Việt, STAR-RIS 3D, hướng mũi tên, thuật ngữ và kết luận đạt/chưa đạt.

Không đánh dấu hoàn thành nếu STAR-RIS không nghiêng 3D, đặt sai miền phản xạ/truyền qua, còn câu tiếng Anh không cần thiết, NOMA giống SDMA, RSMA thiếu luồng chung, TD3 thiếu Critic thứ hai, test quay lại huấn luyện hoặc biểu đồ Chương 4 không lấy từ dữ liệu thật.
