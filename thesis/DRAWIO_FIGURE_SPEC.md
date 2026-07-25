# Đặc tả hình minh họa Draw.io cho luận văn

## Quy ước chung

- Khổ ngang 16:9 hoặc tỷ lệ gần 1,6:1; nền trắng.
- Font Times New Roman hoặc Liberation Serif; cỡ chữ tối thiểu 18 pt trong file gốc.
- Mũi tên một chiều, đầu mũi tên rõ; tránh đường cắt nhau.
- Mỗi nhóm dùng một màu pastel khác nhau nhưng vẫn đọc được khi in trắng đen.
- Không đặt caption trong hình; caption do LaTeX tạo.
- Xuất hai định dạng cùng basename: `.svg` và `.pdf`.
- Tệp PDF phải crop sát nội dung, không có trang trắng thừa.

---

## Hình Chương 1 – `chapter1_research_landscape`

**Mục đích:** mô tả bản đồ nghiên cứu và khoảng trống mà luận văn giải quyết.

**Bố cục:** ba tầng từ trái sang phải.

1. Tầng nền tảng bên trái gồm ba hộp lớn:
   - RSMA: common/private streams, SIC, interference management.
   - RIS/STAR-RIS: reflection, transmission, ES/MS/TS, coupled phase.
   - Optimization/DRL: AO, SCA, DDPG, PPO, TD3.
2. Tầng vấn đề ở giữa gồm bốn hộp:
   - High-dimensional continuous action.
   - Non-convex rate/QoS constraints.
   - Online solve latency.
   - Fair and reproducible evaluation.
3. Tầng đóng góp bên phải là một hộp trung tâm lớn:
   - SISO STAR-RIS–RSMA shared physics.
   - TD3 as learned optimizer.
   - Locked ScenarioBank.
   - Quality–QoS–latency analysis.
4. Mũi tên từ từng nền tảng sang vấn đề liên quan, sau đó hội tụ vào hộp đóng góp.
5. Dải cuối hình ghi: “No MADDPG/CTDE claims; AO-SCA is local, not global optimum”.

**Liên kết nội dung:** đặt sau mục “Khoảng trống và hướng tiếp cận của luận văn”.

---

## Hình Chương 2 – `chapter2_conceptual_foundation`

**Mục đích:** liên kết luồng tính toán giữa TD3, STAR-RIS và RSMA.

**Bố cục:** sơ đồ vòng kín từ trái sang phải rồi hồi tiếp.

1. Khối “Channel State”:
   - Re/Im of direct, BS–RIS, RIS–user channels.
   - User side indicator.
2. Khối “TD3 Actor”:
   - 2 hidden layers, LayerNorm + ReLU.
   - Tanh output.
3. Khối “Physical Action Decoder” chia thành 4 nhánh:
   - Power simplex.
   - Common-rate simplex.
   - Beta box with betaT + betaR = 1.
   - Transmit/reflection phases.
4. Khối “STAR-RIS Effective Channel”:
   - sqrt(beta) exp(j theta).
   - Direct + cascaded path.
5. Khối “RSMA Rate Calculator”:
   - Decode common at all users.
   - Rc = min Rck.
   - SIC and private rates.
6. Khối “Metrics/Reward”:
   - Sum-rate.
   - QoS fraction/all-QoS.
   - Violation and dual penalty.
7. Mũi tên hồi tiếp reward về TD3 critic/actor.
8. Gắn chú thích nhỏ: “Action is always physically feasible; QoS is handled by reward/selection”.

**Liên kết nội dung:** đặt cuối Chương 2 trước kết luận chương.

---

## Hình Chương 3 – `chapter3_system_model`

**Mục đích:** minh họa đúng mô hình vật lý source code.

**Bố cục:** BS ở trái, STAR-RIS giữa, users ở hai phía phải/trên/dưới.

1. BS một anten, nhãn “SISO BS”.
2. STAR-RIS gồm lưới N phần tử, có hai vùng:
   - Transmission side.
   - Reflection side.
3. Bốn user SISO:
   - U1, U3 phía reflection.
   - U2, U4 phía transmission.
4. Vẽ đường trực tiếp nét đứt từ BS đến từng user.
5. Vẽ đường BS→STAR-RIS và STAR-RIS→user nét liền.
6. Trên đường phát ghi:
   - Common stream sc.
   - Private streams s1…sK.
7. Bên cạnh STAR-RIS đặt công thức:
   - phiT = sqrt(betaT) exp(j thetaT).
   - phiR = sqrt(1-betaT) exp(j thetaR).
8. Dưới hình đặt hộp action vector:
   - powers | common fractions | betaT | thetaT | thetaR.
9. Không vẽ nhiều anten hoặc active beamforming.

**Liên kết nội dung:** đặt sau phần giả định/độ phức tạp của Chương 3.

---

## Hình Chương 4 – `chapter4_td3_pipeline`

**Mục đích:** minh họa pipeline huấn luyện và chống rò rỉ test.

**Bố cục:** ba lane ngang.

### Lane 1 – Data
- Train ScenarioBank (10,000).
- Validation ScenarioBank (1,000).
- Test ScenarioBank (1,000).
- Mỗi bank có seed, checksum, metadata.
- Vẽ dấu “disjoint” giữa ba bank.

### Lane 2 – Training
- Environment/shared physics.
- Replay buffer.
- Actor + actor target.
- Twin critics + critic targets.
- Target smoothing and delayed update.
- QoS dual controller.
- Validation every 5,000 steps.

### Lane 3 – Selection and reporting
- Feasibility-first checkpoint key.
- `best.pt`.
- Deterministic test, noise = 0.
- Raw per-scenario CSV.
- Paired statistics + Holm.
- CPU one-thread latency.
- Final tables/figures/audit.

**Mũi tên quan trọng:**
- Train bank chỉ vào training.
- Validation chỉ vào checkpoint selection.
- Test chỉ nhận `best.pt`; tuyệt đối không có mũi tên từ test quay về training/selection.
- Vẽ biểu tượng khóa trên test bank.

**Liên kết nội dung:** đặt cuối Chương 4.

---

## Hình Chương 5 – `chapter5_quality_latency_tradeoff`

**Mục đích:** biểu diễn định tính đường biên trade-off.

**Bố cục:** đồ thị 2D.

- Trục X: Decision latency (thấp → cao).
- Trục Y: Sum-rate quality (thấp → cao).
- Dùng hình dạng/viền để mã hóa QoS:
  - Viền xanh đậm: QoS reliable.
  - Viền đỏ nét đứt: QoS fails.

**Các điểm:**
1. AnalyticalRIS: rất trái, thấp; viền đỏ; ghi “~2.07–2.30x faster than TD3; QoS fails”.
2. AO-Grid: giữa-trái, thấp-trung bình; ghi “TD3 +169.1–211.7% sum-rate”.
3. TD3: trái-trung tâm, trung bình-cao; viền xanh; ghi “QoS fraction 0.9892–0.9994”.
4. AO-SCA: phải, cao nhất; ghi “+4.4476–7.2246 bit/s/Hz vs TD3”.
5. Mũi tên từ AO-SCA về TD3 ghi “TD3 884–3690x faster”.

**Lưu ý:** đây là hình định tính. Không dùng tọa độ tuyệt đối hoặc scale tuyến tính giả khi chưa nhập raw data.

**Liên kết nội dung:** đặt sau phần threats to validity, trước hàm ý thực tiễn.

---

## Logo UTH – `uth-logo`

- Sử dụng logo chính thức được trường cho phép.
- Xuất PDF nền trong suốt hoặc nền trắng.
- Không kéo méo tỷ lệ.
- Tệp: `thesis/figures/uth-logo.pdf`.
