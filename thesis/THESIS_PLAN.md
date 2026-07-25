# Kế hoạch viết và đóng băng luận văn STAR-RIS–RSMA TD3

## 1. Mục tiêu bản thảo

Bản LaTeX này được xây dựng theo cấu trúc 5 chương, bám sát source code hiện tại và quy định trình bày của UTH. Mục tiêu dung lượng là khoảng 80–95 trang nội dung chính khi dùng cỡ chữ 13, giãn dòng 1,5; phụ lục có thể làm tổng số trang cao hơn nhưng vẫn dưới giới hạn 120 trang không kể phụ lục theo hướng dẫn.

Tên đề tài hiện dùng:

> TỐI ƯU PHÂN BỔ TÀI NGUYÊN SỬ DỤNG HỌC TĂNG CƯỜNG SÂU TRONG MẠNG STAR-RIS HỖ TRỢ RSMA

Phương pháp chính: TD3 đơn tác tử. Không sử dụng lại claim MADDPG, CTDE hoặc dữ liệu cũ.

## 2. Cấu trúc nội dung

- Mở đầu: bối cảnh, khoảng trống, mục tiêu, câu hỏi, phạm vi, đóng góp.
- Chương 1: tổng quan RSMA, RIS, STAR-RIS, tối ưu, DRL và khoảng trống.
- Chương 2: cơ sở lý thuyết, công thức RSMA, energy splitting, MDP, DDPG/TD3, QoS, thống kê.
- Chương 3: mô hình SISO, kênh, state/action, reward, bài toán tối ưu và giả định.
- Chương 4: kiến trúc TD3, dual controller, checkpoint selection, baseline, ablation, ScenarioBank và CI.
- Chương 5: kết quả đã kiểm toán, quality–latency trade-off, QoS, thống kê, hạn chế.
- Kết luận và hướng phát triển.
- Phụ lục: ký hiệu/cấu hình, tái lập, checklist và danh sách hình Draw.io.

## 3. Hợp đồng nhất quán source–thesis

1. Công thức kênh phải khớp `src/star_ris_rsma/physics.py`.
2. Action phải khớp `physical_v3`: simplex power, simplex common fractions, beta trong [0,1], pha trong [-pi, pi).
3. State phải khớp `blockwise_v2` trong `env.py`.
4. Reward huấn luyện phải mô tả dual multiplier trong `experiment_v2.py`.
5. Checkpoint phải mô tả feasibility-first validation selection.
6. Kết quả chỉ dùng bundle đã audit; không tạo số liệu trung gian.
7. AO-SCA là local proximal first-order solver, không phải global optimum.
8. AnalyticalRIS là phase alignment + equal allocation, không phải full analytical optimizer.
9. TD3 được định vị là QoS-reliable, low-latency compromise.

## 4. Giai đoạn hoàn thiện trước bản nộp

### Giai đoạn A – bản thảo có thể đọc
- Hoàn tất LaTeX, mục lục, công thức, bảng, trích dẫn.
- CI build PDF và commit PDF vào nhánh.
- Placeholder cho 5 hình Draw.io.

### Giai đoạn B – bổ sung tài sản trực quan
- Vẽ 5 hình theo `DRAWIO_FIGURE_SPEC.md`.
- Xuất SVG và PDF vào `thesis/figures/`.
- Thêm logo UTH chính thức tại `thesis/figures/uth-logo.pdf`.

### Giai đoạn C – chốt số liệu
- Tự động trích bảng per-N từ raw CSV.
- Bổ sung DDPG/PPO chỉ khi đủ seed, ScenarioBank và manifest.
- Thay bảng range bằng bảng mean/CI đầy đủ.
- Rà soát statistical claims.

### Giai đoạn D – rà soát hội đồng
- Kiểm tra số trang, tên đề tài, loại văn bản, tháng/năm.
- Soát thuật ngữ Việt–Anh và ký hiệu.
- Soát IEEE references theo thứ tự xuất hiện.
- Render toàn PDF và kiểm tra tràn lề, bảng, caption, bookmark.

## 5. Điều kiện chấp nhận bản cuối

- CI xanh và PDF mở được.
- Không có citation undefined, reference undefined hoặc missing figure ở bản nộp.
- Trang chính từ Mở đầu đến Tài liệu tham khảo đánh số liên tục ở giữa lề trên.
- Khổ A4, lề trái 3 cm, phải 2 cm, trên/dưới 2,5 cm.
- Nội dung 13 pt, giãn dòng 1,5; chương 14 pt, in hoa, đậm, căn giữa.
- Mọi bảng/hình đánh số theo chương và có nguồn khi cần.
- Không có claim vượt quá bằng chứng của repository.
