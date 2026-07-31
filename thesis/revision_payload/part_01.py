Trong luận văn, tài nguyên được hiểu theo nghĩa rộng, gồm công suất của các luồng RSMA, tỷ lệ tốc độ chung và cấu hình STAR--RIS. Bài toán không chỉ là chia công suất mà là phối hợp nhiều nhóm biến để cải thiện kênh hiệu dụng và đồng thời bảo đảm QoS.

\section{Các kỹ thuật đa truy cập}
\subsection{OMA, SDMA và NOMA}
Đa truy cập trực giao (Orthogonal Multiple Access, OMA) tách người dùng theo thời gian, tần số hoặc mã. Cách này dễ triển khai và hạn chế nhiễu, nhưng mức tái sử dụng tài nguyên thấp vì mỗi phần tài nguyên thường chỉ phục vụ một nhóm người dùng.

Đa truy cập phân chia theo không gian (Space-Division Multiple Access, SDMA) sử dụng nhiều anten để tạo các búp sóng hướng đến những người dùng khác nhau. SDMA phù hợp với hệ nhiều đầu vào, một đầu ra (Multiple-Input Single-Output, MISO), nhưng hiệu năng phụ thuộc vào khả năng phân tách không gian và chất lượng CSI.

Đa truy cập phi trực giao (Non-Orthogonal Multiple Access, NOMA) miền công suất chồng nhiều tín hiệu trên cùng tài nguyên. Người dùng có kênh mạnh thực hiện khử nhiễu liên tiếp (Successive Interference Cancellation, SIC) theo một thứ tự giải mã. NOMA có thể cải thiện hiệu quả phổ trong một số cấu hình, nhưng phụ thuộc vào chênh lệch kênh, phân bổ công suất và độ chính xác của SIC \cite{Clerckx2021NOMACritical}.

\ThesisFigure{figures/pdf/chapter1_fig01_oma_sdma_noma_rsma.pdf}
{So sánh nguyên lý hoạt động của OMA, SDMA, NOMA và RSMA}
{fig:ch1-access-comparison}[0.96\linewidth]
\figuresource{Tác giả xây dựng dựa trên \cite{Mao2018RSMABridging,Mao2022RSMAFundamentals,Clerckx2021NOMACritical} (2026)}

Hình~\ref{fig:ch1-access-comparison} minh họa sự khác nhau giữa bốn kỹ thuật. OMA cấp tài nguyên trực giao riêng; SDMA tách người dùng bằng búp sóng; NOMA truyền một tín hiệu chồng công suất và giải mã theo thứ tự; RSMA truyền một luồng chung cùng nhiều luồng riêng. Ba người dùng trong hình chỉ dùng để minh họa, trong khi mô phỏng của luận văn sử dụng bốn người dùng.

\subsection{Ý tưởng của RSMA}
Trong RSMA một lớp, thông điệp của người dùng được chia thành phần chung và phần riêng. Các phần chung được ghép thành một thông điệp chung, mã hóa thành luồng $s_c$. Phần riêng của người dùng $k$ được mã hóa thành luồng $s_k$. Tín hiệu phát SISO có dạng
\begin{equation}
 x=\sqrt{p_c}s_c+\sum_{k=1}^{K}\sqrt{p_k}s_k.
 \label{eq:ch1-rsma-signal}
\end{equation}
Trong đó, $x$ là tín hiệu tổng được BS phát; $s_c$ và $s_k$ lần lượt là ký hiệu của luồng chung và luồng riêng người dùng $k$; $p_c$ và $p_k$ là công suất tương ứng; $K$ là số người dùng. Công thức mô tả nguyên lý chồng tín hiệu của RSMA và xác định trực tiếp nhóm biến công suất cần tối ưu trong bài toán.

Mỗi người dùng thực hiện ba bước: giải mã luồng chung, loại bỏ luồng chung bằng SIC, rồi giải mã luồng riêng của mình. Tốc độ luồng chung phải đủ thấp để mọi người dùng đều giải mã được; vì vậy người dùng yếu nhất tạo thành điểm nghẽn. Sau khi xác định tốc độ chung, hệ thống phân bổ một phần tốc độ này cho từng người dùng để hỗ trợ QoS.

RSMA linh hoạt hơn cách phân loại cứng giữa SDMA và NOMA. Khi không dùng luồng chung, hệ thống gần với chiến lược chỉ truyền các luồng riêng và xem nhiễu là tạp âm. Khi tăng phần chung, nhiều người dùng cùng giải mã một phần nhiễu. Khả năng điều chỉnh liên tục này là lý do RSMA phù hợp với bài toán tối ưu tài nguyên \cite{Mao2018RSMABridging,Mao2022RSMAFundamentals}.

RSMA không tự động tạo ra nghiệm tốt. Công suất luồng chung quá lớn có thể làm giảm công suất luồng riêng; công suất luồng chung quá nhỏ có thể khiến người dùng yếu không đạt QoS. Do đó, các biến công suất và tỷ lệ tốc độ chung phải được tối ưu theo trạng thái kênh.

\section{RIS và STAR--RIS}
\subsection{RIS phản xạ truyền thống}
RIS là một bề mặt gồm nhiều phần tử có thể điều chỉnh đáp ứng pha hoặc biên độ. Sóng từ trạm gốc tới RIS được phản xạ theo cấu hình đã chọn để tăng cường hoặc giảm bớt tín hiệu tại các vị trí mong muốn. RIS không tạo ra dữ liệu mới và trong mô hình thụ động không khuếch đại như relay; nó điều khiển đường truyền gián tiếp \cite{Wu2019IRSBeamforming,DiRenzo2020SmartRadio}.

Lợi ích của RIS phụ thuộc vào mô hình suy hao, diện tích bề mặt, CSI và phần cứng. Vì vậy, không nên diễn giải số phần tử lớn hơn như một bảo đảm về mức tăng cố định trong mọi hệ thống thực tế \cite{Bjornson2020RISMyths,Wu2021IRSTutorial}.

\subsection{STAR--RIS và ba giao thức hoạt động}
STAR--RIS mở rộng RIS bằng cách tạo đồng thời thành phần phản xạ và truyền qua. Ba giao thức thường được sử dụng gồm:
\begin{itemize}
  \item \textbf{ES:} mỗi phần tử chia năng lượng tới thành phần truyền qua và phản xạ trong cùng thời điểm;
  \item \textbf{chuyển đổi chế độ (Mode Switching, MS):} mỗi phần tử tại một thời điểm chỉ hoạt động ở chế độ truyền hoặc phản xạ;
  \item \textbf{chuyển đổi theo thời gian (Time Switching, TS):} toàn bộ bề mặt luân phiên hai chế độ theo thời gian.
\end{itemize}

Luận văn sử dụng ES. Với phần tử thứ $n$, tỷ lệ công suất truyền qua và phản xạ thỏa
\begin{equation}
 \beta_n^T+\beta_n^R=1.
 \label{eq:ch1-es}
\end{equation}
Trong đó, $\beta_n^T$ và $\beta_n^R$ là tỷ lệ công suất truyền qua và phản xạ của phần tử $n$. Công thức biểu diễn định luật bảo toàn năng lượng trong mô hình STAR--RIS thụ động; nó tạo ràng buộc vật lý bắt buộc cho bộ giải mã hành động và làm giảm hai biến biên độ xuống còn một bậc tự do trên mỗi phần tử.

Biên độ tác động lên tín hiệu tương ứng là $\sqrt{\beta_n^T}$ và $\sqrt{\beta_n^R}$, vì $\beta$ biểu diễn tỷ lệ công suất chứ không phải biên độ.

\ThesisFigure{figures/pdf/chapter1_fig02_ris_vs_star_ris.pdf}
{So sánh phạm vi điều khiển tín hiệu của RIS và STAR--RIS}
{fig:ch1-ris-star}[0.96\linewidth]
\figuresource{Tác giả xây dựng dựa trên \cite{Xu2021STARRIS,Mu2021STARAided,Khalid2022STARDesign} (2026)}

Trong Hình~\ref{fig:ch1-ris-star}, RIS truyền thống chỉ tạo tia phản xạ và chủ yếu phục vụ một nửa không gian. STAR--RIS được vẽ nghiêng theo phối cảnh, với miền phản xạ ở bên trái và miền truyền qua ở bên phải. Tia tới được chia thành hai nhánh, giúp hệ thống phục vụ người dùng ở hai phía trong cùng tài nguyên thời gian--tần số.

Mô hình của luận văn cho phép pha truyền và pha phản xạ độc lập. Một số phần cứng STAR--RIS thụ động có thể yêu cầu quan hệ ghép giữa hai pha \cite{Liu2021STARCoupledPhase,Wang2022STAROptimization}. Vì vậy, đây là giả định lý tưởng hóa và được xem là giới hạn của nghiên cứu.

\section{Lý do lựa chọn học tăng cường sâu}
\subsection{Hạn chế của học có giám sát và học tăng cường dạng bảng}
Học máy hoặc học sâu có giám sát cần nhãn mục tiêu cho từng trạng thái kênh. Trong bài toán này, một nhãn là toàn bộ vector công suất, tỷ lệ tốc độ chung, hệ số chia năng lượng và hai vector pha. Tạo nhãn chất lượng cao đòi hỏi chạy một bộ giải tối ưu lặp cho rất nhiều trạng thái kênh. Chi phí tạo nhãn có thể lớn hơn chi phí huấn luyện, và mô hình học có nguy cơ kế thừa sai lệch của chính bộ giải sinh nhãn.

Học tăng cường dạng bảng không cần nhãn tối ưu, nhưng phải lưu giá trị cho các cặp trạng thái--hành động rời rạc. Trạng thái CSI trong luận văn là liên tục; hành động cũng liên tục và có số chiều $2K+1+3N$. Chỉ riêng việc lượng tử hóa mỗi chiều thành một số mức nhỏ cũng làm số tổ hợp tăng theo hàm mũ, nên không thể bao phủ đầy đủ bằng bảng.

\subsection{Vai trò của DRL và TD3}
DRL thay bảng bằng mạng nơ-ron, nhờ đó có thể xấp xỉ chính sách trên không gian liên tục và tổng quát giữa các trạng thái kênh. Tác tử không cần biết hành động tối ưu trước; nó nhận phản hồi từ tổng tốc độ và mức vi phạm QoS. Sau huấn luyện, Actor tạo quyết định bằng một lần truyền thẳng, phù hợp với mục tiêu giảm độ trễ suy luận.

Trong các thuật toán hành động liên tục, DDPG có hiệu suất mẫu tốt nhờ học off-policy nhưng nhạy với sai số Critic. PPO ổn định nhờ giới hạn thay đổi chính sách nhưng dùng dữ liệu on-policy. TD3 kế thừa khả năng xử lý hành động liên tục của DDPG và bổ sung hai Critic, làm trơn hành động đích cùng cập nhật Actor trễ để giảm sai số xấp xỉ \cite{Lillicrap2015DDPG,Fujimoto2018TD3}. Đây là cơ sở phương pháp luận để chọn TD3 làm thuật toán chính, còn DDPG và PPO được giữ làm đối chứng học sâu.

\ThesisFigure{figures/pdf/chapter1_fig03_drl_control_loop.pdf}
{Vòng lặp DRL và vai trò của TD3 trong bài toán phân bổ tài nguyên}
{fig:ch1-drl-loop}[0.94\linewidth]
\figuresource{Tác giả xây dựng dựa trên \cite{Lillicrap2015DDPG,Fujimoto2018TD3,Meng2024PPOSTARRSMA} (2026)}

Hình~\ref{fig:ch1-drl-loop} cho thấy môi trường cung cấp trạng thái kênh cho TD3. Actor tạo hành động phân bổ tài nguyên; môi trường tính tổng tốc độ và mức vi phạm QoS để tạo phần thưởng. Vòng lặp này được lặp lại trên nhiều kịch bản trong giai đoạn huấn luyện. Do kênh kế tiếp được lấy mẫu độc lập và không phụ thuộc hành động hiện tại, TD3 trong luận văn được diễn giải như một bộ tối ưu học được theo ngữ cảnh hơn là một bộ điều khiển động lực dài hạn.

\section{Các nghiên cứu liên quan}
\subsection{RSMA và RIS}
Các nghiên cứu nền tảng về RSMA chỉ ra rằng việc chia thông điệp thành phần chung và riêng tạo một khuôn khổ linh hoạt để quản lý nhiễu, bao quát nhiều trường hợp của SDMA và NOMA \cite{Mao2018RSMABridging,Mao2022RSMAFundamentals}. Bài toán tối đa tổng tốc độ của RSMA thường dẫn đến tối ưu không lồi do các biến công suất, tiền mã hóa và thứ tự giải mã tác động lẫn nhau \cite{Joudeh2016RSMASumRate}. Khi kết hợp RIS với RSMA, bề mặt tái cấu hình bổ sung các biến pha thụ động và tạo sự phụ thuộc chặt giữa kênh hiệu dụng với phân bổ tài nguyên \cite{Li2022RSMARISInterplay,Aboumahmoud2024RISRSMASurvey}.

\subsection{STAR--RIS và tối ưu tài nguyên}
Các công trình khởi đầu về STAR--RIS xây dựng mô hình truyền--phản xạ đồng thời và các giao thức ES, MS, TS \cite{Xu2021STARRIS,Mu2021STARAided}. Những nghiên cứu tiếp theo xem xét ràng buộc ghép pha và phát triển khuôn khổ tối ưu cho phần cứng thụ động thực tế hơn \cite{Liu2021STARCoupledPhase,Wang2022STAROptimization}. Trong hệ STAR--RIS hỗ trợ RSMA, các hướng đã được nghiên cứu gồm tối ưu nhiều anten, massive MIMO, pha rời rạc, truyền thông bảo mật và STAR--RIS chủ động \cite{Ge2024RSMAMassiveMIMO,Liu2024DiscretePhaseRSMARate,Hashempour2024SecureSTARRS,Maghrebi2024CooperativeActiveSTAR}.

\subsection{Học tăng cường cho STAR--RIS và RSMA}
Học tăng cường đã được sử dụng để xử lý các biến liên tục hoặc hỗn hợp trong STAR--RIS, chẳng hạn bộ tạo búp sóng với ràng buộc ghép pha \cite{Zhong2022HybridRLSTAR}. Đối với mạng STAR--RIS hỗ trợ RSMA, PPO đã được áp dụng để tối đa tổng tốc độ trong một cấu hình hệ thống cụ thể \cite{Meng2024PPOSTARRSMA}. Các kết quả này cho thấy DRL có tiềm năng giảm chi phí tối ưu trực tuyến, nhưng hiệu năng phụ thuộc mạnh vào cách biểu diễn hành động, hàm thưởng, ngân sách tương tác và giao thức đánh giá.

\section{Khoảng trống nghiên cứu và định vị của luận văn}
