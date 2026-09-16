# Hình vẽ TikZ cho Chương 1–3

Mỗi tệp `.tex` là một tài liệu độc lập, biên dịch bằng **XeLaTeX** (cần tiếng
Việt có dấu):

```bash
xelatex hinh11.tex
```

Bản `.pdf` đi kèm đã biên dịch sẵn nên có thể `\includegraphics` ngay.

## Danh mục

| Tệp | Hình | Nội dung |
|---|---|---|
| `hinh11.tex` | 1.1 | So sánh OMA, SDMA, NOMA, RSMA với `K = 4`, kèm dải quan hệ bao hàm theo `ρ_c` |
| `hinh12.tex` | 1.2 | So sánh vùng phục vụ của RIS phản xạ và STAR–RIS, kèm chi tiết một phần tử ở chế độ ES |
| `hinh_thongnhat.tex` | 2.1, 3.1, 3.5 | Mô hình hệ thống; khung DRL; kiến trúc Actor và Critic |
| `hinh_kientruc.tex` | 2.3 | Kiến trúc thu–phát RSMA một lớp cho `K` UE trong hệ SISO |
| `hinh_ch23.tex` | 2.4, 2.5, 3.2, 3.3, 3.4 | Trần tốc độ luồng riêng; căn pha; véc-tơ trạng thái; bộ giải mã hành động; độ sắc softmax |

## Font

Các tệp đang đặt `\setmainfont{Liberation Serif}` để biên dịch được trên máy
không có font Times. Khi đưa vào luận văn, đổi thành:

```latex
\setmainfont{Times New Roman}
```

Nếu biên dịch bằng pdfLaTeX thay vì XeLaTeX, bỏ gói `fontspec` và dùng:

```latex
\usepackage[utf8]{inputenc}
\usepackage[T5]{fontenc}
\usepackage{times}
```

## Bảng màu dùng chung

Định nghĩa ở đầu `hinh_thongnhat.tex`, nên dùng thống nhất cho mọi hình:

```latex
\definecolor{cUEa}{RGB}{217,119,6}     % UE1
\definecolor{cUEb}{RGB}{13,148,136}    % UE2
\definecolor{cUEc}{RGB}{37,99,235}     % UE3
\definecolor{cUEd}{RGB}{124,58,237}    % UE4
\definecolor{cChung}{RGB}{190,24,93}   % luồng chung
\definecolor{cKenh}{RGB}{22,101,52}    % kênh / STAR-RIS
```

## Cách nhúng

Mỗi `.tex` chứa nhiều khối `tikzpicture`. Để dùng một hình cụ thể, sao chép
khối `\begin{tikzpicture} ... \end{tikzpicture}` tương ứng vào chương, hoặc
`\includegraphics` từ bản `.pdf` đã biên dịch. Hình 2.3 khá rộng, nếu tràn lề
thì bọc thêm:

```latex
\resizebox{\textwidth}{!}{\begin{tikzpicture} ... \end{tikzpicture}}
```

## Kiểm tra trước khi nộp

Các hình này được dựng theo mã nguồn ở nhánh `experiment/physical-v6-full`:
hằng số `0,70`–`0,99`, độ sắc softmax `10`, hiệu chỉnh pha `0,25π`, chuẩn hóa
trạng thái `0,35`–`1,0`–`0,75` và cắt biên `±8`. Nếu cấu hình thay đổi, sửa lại
các con số trong hình cho khớp.
