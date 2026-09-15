# Luận văn LaTeX - STAR-RIS / RSMA / DRL

Đây là source LaTeX bản review của luận văn.

## Biên dịch

Khuyến nghị XeLaTeX + Biber:

```bash
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

Source ưu tiên Times New Roman nếu hệ thống có font này; nếu không có thì tự động fallback sang Tinos để kiểm tra biên dịch.

## Hình minh họa

- Chương 1-3: ưu tiên hình TikZ do tác giả vẽ lại dựa trên tài liệu tham khảo, tránh sao chép trực tiếp hình có bản quyền.
- Chương 4: ưu tiên hình sinh từ dữ liệu/code của chính đề tài. Hai placeholder `learning_curves.png` và `qos.png` có thể thay bằng các hình tiếng Việt trong thư mục `results/.../figures_vi/` của repo.
- Nếu muốn dùng trực tiếp hình từ paper: chỉ nên dùng khi paper/figure có giấy phép cho phép tái sử dụng; nếu không, hãy vẽ lại và ghi `Nguồn: Tác giả xây dựng dựa trên [x]`.

## Lưu ý khoa học

- Không dùng các tên version nội bộ V3/V6 trong nội dung luận văn.
- Không khẳng định TD3 luôn tốt hơn DDPG/PPO.
- So sánh chính: nhóm DRL (off-policy/on-policy) với nhóm tối ưu lặp AO, nhấn mạnh đánh đổi chất lượng - QoS - độ trễ.
- Các hằng số 0.70, 0.99, 10 và 0.25π phải được mô tả là lựa chọn thiết kế/tiên nghiệm, không phải định luật vật lý.
