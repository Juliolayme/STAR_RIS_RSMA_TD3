from __future__ import annotations

"""Apply reviewer-requested thesis revisions deterministically.

The script rewrites the stable academic narrative, removes volatile repository
identifiers from the thesis body, adds symbol/purpose/role explanations after
all displayed equations, expands related work and AO/SCA, and updates the
cover to use the official UTH logo downloaded by the workflow.
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
THESIS = ROOT / "thesis"


def write_text(name: str, content: str) -> None:
    path = THESIS / name
    path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"updated {path.relative_to(ROOT)}")


INTRODUCTION = r"""
\chapter*{MỞ ĐẦU}
\addcontentsline{toc}{chapter}{MỞ ĐẦU}

\section*{1. Tính cấp thiết của đề tài}
\addcontentsline{toc}{section}{1. Tính cấp thiết của đề tài}
Các hệ thống thông tin vô tuyến thế hệ mới phải đồng thời nâng cao tốc độ dữ liệu, phục vụ nhiều người dùng, duy trì chất lượng dịch vụ và rút ngắn thời gian ra quyết định. Trong điều kiện băng thông và công suất hữu hạn, hiệu năng không chỉ phụ thuộc vào mức năng lượng phát mà còn phụ thuộc vào cách quản lý nhiễu và cách điều khiển môi trường truyền sóng.

Đa truy cập phân chia tốc độ (Rate-Splitting Multiple Access, RSMA) là một kỹ thuật quản lý nhiễu linh hoạt. Thông điệp của mỗi người dùng được chia thành phần chung và phần riêng. Mọi người dùng giải mã luồng chung trước, loại bỏ luồng này, rồi giải mã luồng riêng của mình. Nhờ đó, RSMA có thể điều chỉnh liên tục giữa hai cách xử lý nhiễu: xem nhiễu như tạp âm hoặc giải mã một phần nhiễu \cite{Mao2018RSMABridging,Mao2022RSMAFundamentals}.

Bề mặt thông minh tái cấu hình đồng thời truyền và phản xạ (Simultaneously Transmitting and Reflecting Reconfigurable Intelligent Surface, STAR--RIS) cho phép tạo cả thành phần truyền qua và phản xạ từ cùng một bề mặt. So với bề mặt thông minh tái cấu hình (Reconfigurable Intelligent Surface, RIS) phản xạ truyền thống, STAR--RIS có thể phục vụ người dùng ở cả hai phía của bề mặt \cite{Xu2021STARRIS,Mu2021STARAided}. Khi kết hợp STAR--RIS với RSMA, hệ thống có thêm nhiều biến điều khiển gồm công suất, tỷ lệ tốc độ chung, hệ số chia năng lượng và pha của từng phần tử.

Bài toán tối ưu đồng thời các biến trên có tính phi lồi và số chiều tăng theo số phần tử STAR--RIS. Các bộ giải lặp có thể đạt nghiệm tốt nhưng phải giải lại từ đầu hoặc tinh chỉnh lại cho từng hiện thực kênh. Điều này tạo ra khoảng cách giữa chất lượng nghiệm và thời gian ra quyết định khi hệ thống phải phản ứng nhanh với nhiều trạng thái kênh.

Việc lựa chọn học tăng cường sâu (Deep Reinforcement Learning, DRL) xuất phát từ đặc điểm dữ liệu và không gian tìm kiếm. Học máy hoặc học sâu có giám sát cần một tập lớn cặp ``trạng thái kênh--hành động tối ưu'' làm nhãn. Để tạo các nhãn này, phải chạy một bộ giải lặp tốn thời gian trên rất nhiều mẫu kênh; hơn nữa, nhãn thu được còn phụ thuộc vào chất lượng và khởi tạo của bộ giải. Học tăng cường dạng bảng cũng không phù hợp vì trạng thái và hành động đều liên tục, trong khi số chiều hành động tăng tuyến tính theo số phần tử STAR--RIS nên không thể liệt kê đầy đủ mọi trạng thái--hành động. DRL dùng mạng nơ-ron để xấp xỉ chính sách, học trực tiếp từ tương tác với môi trường và tổng quát hóa giữa các mẫu kênh chưa gặp. Luận văn lựa chọn thuật toán gradient chính sách xác định sâu trì hoãn kép (Twin Delayed Deep Deterministic Policy Gradient, TD3) vì thuật toán này xử lý hành động liên tục, tái sử dụng dữ liệu bằng Replay Buffer và giảm sai số đánh giá quá cao bằng hai Critic, làm trơn hành động đích và cập nhật Actor trễ \cite{Fujimoto2018TD3}.

Chất lượng dịch vụ (Quality of Service, QoS) được xem là ràng buộc trung tâm thay vì chỉ là chỉ số phụ. Vì vậy, luận văn không chỉ đánh giá tổng tốc độ mà còn xem xét tỷ lệ người dùng đạt QoS, xác suất toàn bộ người dùng cùng đạt QoS, mức vi phạm, tính ổn định theo seed và độ trễ ra quyết định.

\section*{2. Vấn đề nghiên cứu}
\addcontentsline{toc}{section}{2. Vấn đề nghiên cứu}
Hệ thống được nghiên cứu là mạng truyền xuống một đầu vào, một đầu ra (Single-Input Single-Output, SISO) gồm một trạm gốc, một STAR--RIS thụ động và bốn người dùng. STAR--RIS vận hành theo giao thức phân chia năng lượng (Energy Splitting, ES). Trạm gốc phát một luồng chung và bốn luồng riêng theo RSMA. Bài toán cần lựa chọn đồng thời:
\begin{itemize}
  \item công suất của luồng chung và các luồng riêng;
  \item tỷ lệ phân bổ tốc độ chung cho từng người dùng;
  \item hệ số chia năng lượng truyền qua và phản xạ;
  \item pha truyền và pha phản xạ của từng phần tử STAR--RIS.
\end{itemize}

Các biến trên tác động lẫn nhau thông qua kênh hiệu dụng và biểu thức tỷ số tín hiệu trên nhiễu và can nhiễu (Signal-to-Interference-plus-Noise Ratio, SINR). Tốc độ luồng chung còn bị giới hạn bởi người dùng giải mã yếu nhất. Vì vậy, miền nghiệm có nhiều cực trị cục bộ và khó giải nhanh khi số phần tử tăng.

Luận văn đặt TD3 trong hai nhóm so sánh. Nhóm DRL gồm gradient chính sách xác định sâu (Deep Deterministic Policy Gradient, DDPG) \cite{Lillicrap2015DDPG} và tối ưu chính sách lân cận (Proximal Policy Optimization, PPO) \cite{Meng2024PPOSTARRSMA}. Nhóm truyền thống gồm tối ưu luân phiên (Alternating Optimization, AO), xấp xỉ lồi liên tiếp (Successive Convex Approximation, SCA), tìm kiếm lưới và căn chỉnh pha phân tích. Phương pháp kết hợp AO--SCA được xem là bộ giải cục bộ, không phải nghiệm tối ưu toàn cục.

\section*{3. Mục tiêu nghiên cứu}
\addcontentsline{toc}{section}{3. Mục tiêu nghiên cứu}
\subsection*{3.1. Mục tiêu tổng quát}
Xây dựng và đánh giá một bộ tối ưu TD3 cho bài toán phân bổ tài nguyên trong mạng STAR--RIS hỗ trợ RSMA, hướng đến sự cân bằng giữa tổng tốc độ, khả năng đáp ứng QoS và độ trễ ra quyết định.

\subsection*{3.2. Mục tiêu cụ thể}
\begin{enumerate}[label=\arabic*)]
  \item Xây dựng mô hình tín hiệu, kênh hiệu dụng, tốc độ RSMA và mô hình ES của STAR--RIS nhất quán với cấu hình thực nghiệm.
  \item Phát biểu bài toán tối ưu đồng thời các biến công suất, tốc độ chung, chia năng lượng và pha dưới các ràng buộc vật lý và QoS.
  \item Xây dựng trạng thái, hành động, hàm thưởng, bộ giải mã hành động vật lý và quy trình huấn luyện TD3.
  \item So sánh TD3 với DDPG, PPO, AO--SCA, AO--Grid và AnalyticalRIS trên cùng tập kịch bản kiểm thử khóa.
  \item Đánh giá kết quả bằng nhiều seed, thống kê ghép cặp, hiệu chỉnh Holm và đo độ trễ trên cùng bộ xử lý trung tâm (Central Processing Unit, CPU).
\end{enumerate}

\section*{4. Đối tượng và phạm vi nghiên cứu}
\addcontentsline{toc}{section}{4. Đối tượng và phạm vi nghiên cứu}
Đối tượng nghiên cứu gồm mô hình STAR--RIS hỗ trợ RSMA, bài toán phân bổ tài nguyên liên tục, thuật toán TD3 và các phương pháp so sánh. Cấu hình mô phỏng dùng một trạm gốc SISO, bốn người dùng SISO và một STAR--RIS có số phần tử $N\in\{16,32,64,96,128\}$.

Thông tin trạng thái kênh (Channel State Information, CSI) được cung cấp hoàn hảo cho bộ ra quyết định. Kênh được sinh tổng hợp theo phân bố Gaussian phức độc lập. Người dùng được gán cố định vào miền truyền qua hoặc miền phản xạ. Mô hình chưa xét chuyển động người dùng, tương quan thời gian, suy hao theo vị trí, kênh Rician, đa ô, lỗi ước lượng CSI, lượng tử hóa pha, sai số khử nhiễu, tiêu thụ công suất phần cứng hoặc thử nghiệm thiết bị thật.

Các thuật toán DRL sử dụng cùng ngân sách 100.000 tương tác, cùng tập kịch bản huấn luyện--xác thực--kiểm thử và cùng quy tắc lựa chọn checkpoint; tuy nhiên, chưa thực hiện tìm kiếm siêu tham số với ngân sách bằng nhau cho mọi thuật toán. Do đó, kết luận chỉ áp dụng cho các cấu hình và giao thức thực nghiệm được trình bày trong luận văn.

\section*{5. Phương pháp nghiên cứu}
\addcontentsline{toc}{section}{5. Phương pháp nghiên cứu}
\begin{itemize}
  \item \textbf{Tổng hợp tài liệu:} nghiên cứu cơ sở RSMA, RIS, STAR--RIS, DDPG, PPO, TD3 và các kỹ thuật tối ưu phi lồi.
  \item \textbf{Mô hình hóa:} xây dựng phương trình kênh, tín hiệu, SINR, tốc độ, QoS, hàm mục tiêu và ràng buộc.
  \item \textbf{Thiết kế thuật toán:} xây dựng Actor, hai Critic, Replay Buffer, bộ điều khiển đối ngẫu QoS và phép chiếu hành động vào miền vật lý.
  \item \textbf{Mô phỏng:} triển khai bằng Python/PyTorch, huấn luyện tám seed trên từng kích thước STAR--RIS và đánh giá trên tập kiểm thử khóa.
  \item \textbf{Phân tích thống kê:} sử dụng trung bình, độ lệch chuẩn, khoảng tin cậy (Confidence Interval, CI), kiểm định ghép cặp, kích thước ảnh hưởng và hiệu chỉnh Holm.
  \item \textbf{Đánh giá hệ thống:} đo độ trễ của sáu phương pháp trên cùng một CPU một luồng.
\end{itemize}

\section*{6. Đóng góp của luận văn}
\addcontentsline{toc}{section}{6. Đóng góp của luận văn}
\begin{enumerate}[label=\arabic*)]
  \item Xây dựng mô hình SISO STAR--RIS--RSMA thống nhất, trong đó mọi phương pháp dùng chung mô hình vật lý và bộ tính tốc độ.
  \item Xây dựng bộ giải mã hành động bảo đảm công suất và tỷ lệ tốc độ nằm trên simplex, hệ số chia năng lượng thuộc $[0,1]$ và pha nằm trong miền hợp lệ.
  \item Xây dựng TD3 có hai Critic, làm trơn hành động đích, cập nhật Actor trễ, chuẩn hóa trạng thái theo khối và điều khiển đối ngẫu cho QoS.
  \item Xây dựng quy trình tách biệt huấn luyện--xác thực--kiểm thử, lựa chọn checkpoint theo ưu tiên tính khả thi và lưu các thông tin cần thiết để kiểm tra lại kết quả.
  \item Thực hiện so sánh sáu phương pháp trên năm kích thước STAR--RIS, tám seed cho mỗi thuật toán DRL và 1.000 kịch bản kiểm thử khóa.
  \item Đưa ra kết luận cân bằng: AO--SCA đạt tổng tốc độ cao nhất, AnalyticalRIS có độ trễ thấp nhất, còn TD3 là phương pháp DRL ổn định nhất và tạo điểm cân bằng tốt giữa chất lượng, QoS và độ trễ trong giao thức đã đánh giá.
\end{enumerate}

\section*{7. Kết cấu luận văn}
\addcontentsline{toc}{section}{7. Kết cấu luận văn}
Ngoài phần Mở đầu, Kết luận, Tài liệu tham khảo và Phụ lục, luận văn gồm bốn chương. Chương 1 trình bày cơ sở lý thuyết, các nghiên cứu liên quan, khoảng trống nghiên cứu và nguyên lý của RSMA, STAR--RIS, DRL cùng AO--SCA. Chương 2 xây dựng mô hình hệ thống và bài toán tối ưu. Chương 3 trình bày quá trình xây dựng TD3, các phương pháp so sánh và giao thức thực nghiệm. Chương 4 báo cáo kết quả mô phỏng, phân tích thống kê, độ trễ, khả năng mở rộng và các phát hiện chính.
"""


CHAPTER1 = r"""
\chapter{CƠ SỞ LÝ THUYẾT, NGHIÊN CỨU LIÊN QUAN VÀ KHOẢNG TRỐNG}
\label{chap:theory}

\section{Phân bổ tài nguyên và nhiễu trong mạng vô tuyến}
Trong một mạng truyền xuống nhiều người dùng, trạm gốc phải quyết định cách chia công suất, thời gian, tần số hoặc các bậc tự do khác cho từng người dùng. Mục tiêu tăng tổng tốc độ thường ưu tiên người dùng có kênh tốt, trong khi yêu cầu QoS buộc hệ thống duy trì tốc độ tối thiểu cho cả người dùng có kênh yếu. Hai mục tiêu này có thể xung đột.

Nhiễu liên người dùng xuất hiện khi nhiều tín hiệu được truyền trên cùng tài nguyên. Một hệ thống có thể tránh nhiễu bằng cách tách người dùng, triệt nhiễu bằng xử lý không gian, giải mã một phần nhiễu hoặc chấp nhận nhiễu như tạp âm. Mỗi lựa chọn tạo ra một mức cân bằng khác nhau giữa hiệu quả phổ, độ phức tạp của bộ thu và yêu cầu về CSI.

