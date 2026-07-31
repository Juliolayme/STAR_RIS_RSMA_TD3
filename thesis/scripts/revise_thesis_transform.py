from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
THESIS=ROOT/"thesis"
MARKER="% FORMULA_EXPLANATIONS_REVIEW_V1"

CH2_EXPLANATIONS=['Trong đó, $\\mathcal{CN}(0,\\nu^2)$ ký hiệu phân bố Gaussian phức tròn có trung bình bằng không '
 'và phương sai $\\nu^2$; $g_n$ là hệ số BS--STAR--RIS tại phần tử $n$; $h_{r,k,n}$ là hệ số từ '
 'phần tử $n$ đến người dùng $k$. Công thức xác định quy luật sinh các trạng thái kênh và tỷ lệ '
 'tương đối giữa đường trực tiếp với hai đoạn của đường ghép tầng; các hệ số này tạo đầu vào cho '
 'mọi phương pháp đánh giá.',
 'Trong đó, $n_k$ là mẫu nhiễu tại người dùng $k$ và $\\sigma^2$ là công suất nhiễu, bằng '
 '$0{,}001$ trong mô phỏng. Công thức đưa ảnh hưởng của nhiễu nhiệt vào mô hình thu và tạo thành '
 'phần nền trong mẫu số của các biểu thức SINR.',
 'Trong đó, $\\beta_n^T$ và $\\beta_n^R$ là tỷ lệ công suất truyền qua và phản xạ. Công thức bảo '
 'đảm tổng năng lượng đầu ra của phần tử thụ động bằng năng lượng tới theo mô hình ES; trong bài '
 'toán tối ưu, chỉ cần điều khiển $\\beta_n^T$ vì $\\beta_n^R$ được suy ra trực tiếp.',
 'Trong đó, $\\phi_n^T$ và $\\phi_n^R$ là hệ số phức truyền và phản xạ; '
 '$\\theta_n^T,\\theta_n^R\\in[-\\pi,\\pi)$ là các pha điều khiển; $j=\\sqrt{-1}$. Công thức '
 'chuyển tỷ lệ công suất thành biên độ bằng căn bậc hai và ghép biên độ với pha; đây là cầu nối '
 'trực tiếp từ biến điều khiển STAR--RIS đến kênh hiệu dụng.',
 'Trong đó, $\\operatorname{diag}(\\cdot)$ tạo ma trận đường chéo; $\\bm\\Phi_T$ và $\\bm\\Phi_R$ '
 'lần lượt mô tả đáp ứng truyền qua và phản xạ của $N$ phần tử. Công thức giúp biểu diễn gọn tác '
 'động đồng thời của toàn bộ STAR--RIS khi nhân với vector kênh.',
 'Trong đó, $u_k$ chọn đúng ma trận truyền hoặc phản xạ cho người dùng $k$. Công thức hợp nhất hai '
 'miền người dùng trong một biểu thức duy nhất và tránh phải viết hai mô hình kênh riêng biệt.',
 'Trong đó, $(\\cdot)^H$ là chuyển vị liên hợp; $\\bm h_{r,k}^{H}\\bm\\Phi_{u_k}\\bm g$ là tổng '
 'đóng góp của đường BS--STAR--RIS--người dùng. Công thức xác định kênh thực sự mà tín hiệu RSMA '
 'nhìn thấy và là nơi các biến beta, pha cùng CSI tương tác phi tuyến.',
 'Trong đó, $G_k$ là độ lợi công suất vô hướng của người dùng $k$ và $|\\cdot|$ là mô-đun số phức. '
 'Công thức chuyển kênh hiệu dụng phức thành đại lượng dùng trong SINR, qua đó liên kết cấu hình '
 'STAR--RIS với tốc độ truyền.',
 'Trong đó, $s_c$ là ký hiệu luồng chung; $s_k$ là ký hiệu luồng riêng của người dùng $k$; $p_c$ '
 'và $p_k$ là công suất tương ứng; các ký hiệu dữ liệu độc lập và có công suất trung bình bằng '
 'một. Công thức mô tả cách RSMA chồng luồng chung với các luồng riêng, đồng thời xác định các '
 'biến công suất mà thuật toán phải phân bổ.',
 'Trong đó, $P_{\\max}$ là ngân sách công suất tối đa của BS. Công thức giới hạn tổng năng lượng '
 'của tất cả luồng và tạo miền khả thi cơ bản cho khối phân bổ công suất.',
 'Trong đó, $y_k$ là tín hiệu thu, $h_k^{\\mathrm{eff}}x$ là thành phần tín hiệu sau kênh hiệu '
 'dụng và $n_k$ là nhiễu. Công thức là mô hình đầu vào của bộ giải mã và là điểm xuất phát để suy '
 'ra SINR của luồng chung lẫn luồng riêng.',
 'Trong đó, $\\gamma_{c,k}$ là SINR luồng chung tại người dùng $k$; tử số là công suất luồng chung '
 'sau kênh; mẫu số gồm tổng công suất các luồng riêng sau kênh và nhiễu. Công thức đo khả năng mỗi '
 'người dùng giải mã luồng chung và làm cơ sở cho ràng buộc ``mọi người dùng đều phải giải mã '
 "được''.",
 'Trong đó, $R_{c,k}$ là tốc độ khả dụng của luồng chung tại người dùng $k$, đơn vị bit/s/Hz. Công '
 'thức Shannon chuyển SINR thành tốc độ và cung cấp giới hạn giải mã riêng của từng người dùng.',
 'Trong đó, $R_c$ là tốc độ chung có thể được tất cả người dùng giải mã. Phép lấy nhỏ nhất tạo nút '
 'thắt RSMA; công thức này khiến bài toán không trơn tại điểm đổi người dùng yếu nhất và ảnh hưởng '
 'trực tiếp đến phân bổ tốc độ chung.',
 'Trong đó, $\\eta_k$ là phần tỷ lệ dành cho người dùng $k$. Công thức đặt vector $\\bm\\eta$ trên '
 'simplex đơn vị, bảo đảm toàn bộ tốc độ chung được phân bổ mà không tạo thêm tốc độ ngoài $R_c$.',
 'Trong đó, $C_k$ là tốc độ chung được ghi nhận cho người dùng $k$. Công thức biến tỷ lệ $\\eta_k$ '
 'thành đóng góp tốc độ thực và cho phép dùng luồng chung để hỗ trợ người dùng có tốc độ riêng '
 'thấp.',
 'Trong đó, $\\gamma_{p,k}$ là SINR của luồng riêng người dùng $k$; tử số là công suất mong muốn; '
 'mẫu số gồm nhiễu từ các luồng riêng khác và AWGN. Công thức lượng hóa phần nhiễu còn lại sau SIC '
 'và quyết định tốc độ riêng.',
 'Trong đó, $R_{p,k}$ là tốc độ riêng và $R_k$ là tổng tốc độ phục vụ người dùng $k$. Công thức '
 'cộng phần tốc độ chung được phân bổ với tốc độ riêng; $R_k$ là đại lượng trực tiếp dùng trong '
 'mục tiêu tổng tốc độ và các ràng buộc QoS.',
 'Trong đó, $R_{\\mathrm{sum}}$ là tổng tốc độ phổ của toàn bộ người dùng. Công thức là hàm mục '
 'tiêu chất lượng chính mà các phương pháp tối ưu cố gắng cực đại hóa.',
 'Trong đó, $\\mathbb{I}(\\cdot)$ là hàm chỉ báo; $q_{\\mathrm{frac}}\\in[0,1]$ là tỷ lệ người '
 'dùng đạt ngưỡng. Công thức đo mức bao phủ QoS trung bình trong từng kịch bản và cho biết phương '
 'pháp phục vụ được bao nhiêu người dùng.',
 'Trong đó, $q_{\\mathrm{all}}$ bằng một chỉ khi người dùng yếu nhất vẫn đạt ngưỡng. Công thức tạo '
 'tiêu chí khả thi nghiêm ngặt ở cấp kịch bản và nhạy với bất kỳ người dùng nào bị vi phạm.',
 'Trong đó, $V$ là tổng phần thiếu hụt tốc độ so với ngưỡng; toán tử $\\max(\\cdot,0)$ loại bỏ '
 'người dùng đã đạt QoS. Công thức phân biệt vi phạm nhẹ với vi phạm nghiêm trọng và được dùng '
 'trong hàm thưởng cùng hàm merit.',
 'Trong đó, $\\mathcal A$ là toàn bộ hành động vật lý; $\\bm p$ là vector công suất; $\\bm\\eta$ '
 'là vector phân bổ tốc độ chung; $\\bm\\beta^T$ là vector chia năng lượng; $\\bm\\theta^T$ và '
 '$\\bm\\theta^R$ là hai vector pha. Công thức xác định chính xác không gian quyết định chung mà '
 'TD3 và các baseline phải tìm kiếm.',
 'Trong đó, ký hiệu $(\\cdot)^T$ là chuyển vị; số phần tử của $\\bm p$ là $K+1$, của $\\bm\\eta$ '
 'là $K$, và của mỗi vector STAR--RIS là $N$. Công thức làm rõ cấu trúc và số chiều của từng khối, '
 'từ đó giải thích vì sao không gian hành động tăng theo $N$.',
 'Trong đó, $\\mathcal H$ chứa toàn bộ CSI và chỉ báo phía người dùng; $\\mathbf 1$ là vector toàn '
 'một; các bất đẳng thức vector được hiểu theo từng phần tử. Dòng đầu cực đại hóa tổng tốc độ; hai '
 'dòng tiếp theo bảo đảm simplex công suất và tốc độ chung; các dòng beta và pha bảo đảm tính khả '
 'thi của STAR--RIS; dòng cuối áp đặt QoS cho từng người dùng. Hệ công thức này là phát biểu trung '
 'tâm của luận văn, thống nhất mục tiêu, biến quyết định và mọi ràng buộc mà các thuật toán phải '
 'xử lý.',
 'Trong đó, $f$ là giá trị merit; $\\lambda_1,\\lambda_2\\ge0$ là trọng số phạt tuyến tính và bậc '
 'hai; $V$ là tổng vi phạm tuyến tính. Công thức cân bằng mục tiêu tổng tốc độ với mức độ vi phạm '
 'QoS, tạo tín hiệu tối ưu liên tục ngay cả khi nghiệm hiện tại chưa khả thi. Các bảng kết quả vẫn '
 'báo cáo riêng đại lượng vật lý, không dùng merit thay thế tổng tốc độ hoặc QoS.']

CH3_EXPLANATIONS=['Trong đó, $\\mathcal H$ là trạng thái kênh; $\\mu_{\\phi}$ là Actor có tham số $\\phi$; vế phải '
 'là toàn bộ biến phân bổ tài nguyên. Công thức mô tả mục tiêu học một ánh xạ trực tiếp từ CSI '
 'sang hành động vật lý, qua đó chuyển chi phí tối ưu lặp từ giai đoạn suy luận sang giai đoạn '
 'huấn luyện.',
 'Trong đó, $\\bm h_d$ gom các kênh trực tiếp; $\\bm g$ là kênh BS--STAR--RIS; $\\bm H_r$ gom các '
 'kênh STAR--RIS--người dùng; $\\operatorname{vec}(\\cdot)$ trải ma trận thành vector; '
 '$\\widetilde u_k=2u_k-1\\in\\{-1,1\\}$. Công thức tạo đầu vào thực cho mạng nơ-ron từ các hệ số '
 'kênh phức và giữ thông tin người dùng thuộc miền truyền hay phản xạ.',
 'Trong đó, hệ số 2 biểu diễn phần thực và phần ảo; các thành phần $K$, $N$, $KN$ lần lượt đến từ '
 '$\\bm h_d$, $\\bm g$, $\\bm H_r$; số hạng cuối $K$ là chỉ báo phía. Công thức định lượng mức '
 'tăng kích thước đầu vào theo $N$ và dùng để thiết kế lớp đầu của Actor, Critic.',
 'Trong đó, năm khối lần lượt điều khiển công suất, tỷ lệ tốc độ chung, chia năng lượng, pha '
 'truyền và pha phản xạ; $d_a$ là số chiều hành động. Công thức xác định giao diện chuẩn hóa giữa '
 'Actor và bộ giải mã vật lý, giúp đầu ra mạng luôn nằm trong miền hữu hạn.',
 'Trong đó, $K+1$ là số biến công suất, $K$ là số tỷ lệ tốc độ chung và $3N$ gồm một vector beta '
 'cùng hai vector pha. Công thức cho thấy trực tiếp nguyên nhân không gian hành động tăng nhanh '
 'khi số phần tử STAR--RIS lớn.',
 'Trong đó, $\\widetilde{\\bm p}$ và $\\widetilde{\\bm\\eta}$ là vector trung gian không âm; '
 '$\\Pi_{\\Delta_c}$ là phép chiếu lên simplex không âm có tổng bằng $c$; $\\operatorname{wrap}$ '
 'đưa pha về $[-\\pi,\\pi)$. Công thức bảo đảm mọi hành động sinh ra đều thỏa ràng buộc hình học '
 'về công suất, tốc độ chung, beta và pha trước khi tính reward; nhờ đó Actor tập trung học chất '
 'lượng thay vì học lại các ràng buộc cơ bản.',
 'Trong đó, $r_t$ là phần thưởng; $R_{\\mathrm{sum},t}$ là tổng tốc độ; $V_t$ là vi phạm tuyến '
 'tính; $V_{2,t}$ là vi phạm bậc hai; $\\lambda_t$ và $\\lambda_2$ là trọng số phạt. Công thức tạo '
 'tín hiệu học cân bằng giữa tăng chất lượng phổ và giảm vi phạm QoS.',
 'Trong đó, $R_{k,t}$ là tốc độ người dùng $k$ tại bước $t$. Công thức phạt mạnh hơn các vi phạm '
 'lớn so với vi phạm nhỏ, giúp chính sách tránh hy sinh nghiêm trọng một người dùng để tăng tổng '
 'tốc độ.',
 'Trong đó, $\\overline V_t$ là EMA của vi phạm; $10^{-3}$ là mức vi phạm mục tiêu; hệ số 20 là '
 'tốc độ cập nhật; $\\operatorname{clip}(\\cdot,4,64)$ chặn trọng số trong khoảng ổn định. Công '
 'thức tự tăng mức phạt khi vi phạm kéo dài và giảm áp lực phạt khi chính sách đã gần khả thi; đây '
 'là heuristic điều khiển QoS, không phải chứng minh hội tụ primal--dual.',
 'Trong đó, $d_s$ và $d_a$ là kích thước đầu vào, đầu ra; hai số 256 là số nút của hai lớp ẩn. '
 'Công thức mô tả cấu trúc Actor dùng để xấp xỉ ánh xạ từ trạng thái kênh sang hành động liên tục.',
 'Trong đó, $d_s+d_a$ là kích thước của trạng thái ghép hành động và đầu ra 1 là giá trị $Q$. Công '
 'thức mô tả bộ xấp xỉ giá trị dùng để đánh giá chất lượng dài hạn của hành động; hai Critic độc '
 'lập giúp TD3 giảm thiên lệch đánh giá quá cao.',
 'Trong đó, $d_a$ là số chiều hành động và 64 là số chiều tham chiếu. Công thức giữ nguyên mức '
 'nhiễu ở không gian nhỏ nhưng giảm độ lệch chuẩn hiệu dụng khi hành động quá lớn, tránh để năng '
 'lượng khám phá tăng không kiểm soát theo $N$.',
 "Trong đó, $s'_j$ là trạng thái kế tiếp của mẫu $j$; $\\mu_{\\phi'}$ là Actor đích; $\\epsilon_j$ "
 "là nhiễu làm trơn; $a'_j$ là hành động đích đã chặn. Công thức làm trơn vùng lân cận của hành "
 'động đích để Critic không khai thác các đỉnh giá trị hẹp do sai số xấp xỉ.',
 'Trong đó, $r_j$ là reward; $\\gamma$ là hệ số chiết khấu; $d_j$ là chỉ báo kết thúc; '
 "$Q_{\\psi_i'}$ là Critic đích thứ $i$; $y_j$ là mục tiêu huấn luyện. Công thức dùng giá trị nhỏ "
 'hơn của hai Critic để giảm đánh giá quá cao và cung cấp nhãn bootstrap cho cập nhật Critic.',
 'Trong đó, $L_{\\mu}$ là loss Actor; $B$ là kích thước mini-batch; $Q_{\\psi_1}$ là Critic thứ '
 'nhất; $\\mu_{\\phi}(s_j)$ là hành động Actor sinh ra. Công thức dùng dấu âm để biến bài toán '
 'cực đại hóa giá trị $Q$ thành bài toán cực tiểu hóa loss, nhờ đó Actor học chọn hành động được '
 'Critic đánh giá cao.',
 "Trong đó, $\\vartheta$ đại diện tham số mạng trực tuyến; $\\vartheta'$ là tham số mạng đích; "
 '$\\tau=0{,}005$ là hệ số cập nhật mềm. Công thức làm mạng đích thay đổi chậm hơn mạng trực '
 'tuyến, giúp mục tiêu bootstrap ổn định.',
 'Trong đó, $f$ là hàm merit ở \\eqref{eq:merit-ch2}; $\\widehat{\\nabla f}$ là gradient ước lượng '
 'bằng sai phân trung tâm; $\\rho$ là hệ số proximal; $\\bm z^{(t)}$ là nghiệm hiện tại. Công thức '
 'tạo hướng tăng cục bộ nhưng hạn chế bước đi quá xa, giúp mỗi bài toán con dễ xử lý hơn bài toán '
 'phi lồi ban đầu.',
 'Trong đó, $\\Pi_{\\mathcal Z}$ là phép chiếu lên miền khả thi của khối; $\\alpha_t\\in(0,1]$ là '
 'kích thước bước do backtracking chọn. Công thức đưa nghiệm surrogate về ràng buộc vật lý và điều '
 'chỉnh mức cập nhật; nếu merit thật giảm, $\\alpha_t$ được thu nhỏ rồi thử lại.']

RATIONALE='\n\\subsection{Lý do dùng DRL thay cho học có giám sát và RL dạng bảng}\nHọc có giám sát cần một tập lớn cặp trạng thái kênh--hành động tối ưu làm nhãn. Việc tạo nhãn đòi hỏi chạy bộ giải lặp trên từng mẫu kênh, nên chi phí tạo dữ liệu cao và chất lượng nhãn phụ thuộc chính bộ giải. Học tăng cường dạng bảng không cần nhãn nhưng không thể liệt kê không gian trạng thái và hành động liên tục; số chiều hành động còn tăng theo số phần tử STAR--RIS. DRL thay bảng bằng mạng nơ-ron để xấp xỉ chính sách, học trực tiếp từ phản hồi tổng tốc độ và QoS, đồng thời tạo quyết định nhanh bằng một lần truyền thẳng sau huấn luyện. TD3 được chọn vì phù hợp với hành động liên tục và giảm sai số đánh giá quá cao bằng hai Critic, làm trơn hành động đích và cập nhật Actor trễ.\n'
AO_BLOCK='\\subsection{AO--SCA}\nAO--SCA là một bộ giải lặp theo từng kịch bản kênh. Toàn bộ biến được chia thành hai khối: khối RSMA gồm công suất và tỷ lệ tốc độ chung; khối STAR--RIS gồm hệ số chia năng lượng, pha truyền và pha phản xạ. Nghiệm ban đầu được lấy từ cấu hình căn chỉnh pha đơn giản để tránh bắt đầu tại một điểm hoàn toàn ngẫu nhiên.\n\nMỗi vòng lặp ngoài gồm hai bước. Ở bước thứ nhất, cấu hình STAR--RIS được giữ cố định, do đó độ lợi kênh hiệu dụng tạm thời không đổi; bộ giải cập nhật công suất luồng chung, công suất các luồng riêng và tỷ lệ tốc độ chung. Ở bước thứ hai, khối RSMA vừa cập nhật được giữ cố định; bộ giải điều chỉnh beta và hai vector pha để cải thiện kênh hiệu dụng. Sau cả hai bước, hàm merit chính xác được tính lại.\n\nDo từng bài toán con vẫn phi lồi, phương pháp dùng SCA theo nguyên lý ở Chương~\\ref{chap:theory}. Gradient của hàm merit theo từng tọa độ được ước lượng bằng sai phân trung tâm. Tại nghiệm hiện tại, một mô hình tuyến tính--proximal được xây dựng: thành phần tuyến tính chỉ hướng tăng, còn thành phần proximal hạn chế bước đi quá xa khỏi vùng mà xấp xỉ cục bộ còn đáng tin cậy. Nghiệm ứng viên sau đó được chiếu về simplex công suất, simplex tỷ lệ tốc độ, miền beta và miền pha.\n\nMột thủ tục backtracking kiểm tra ứng viên bằng chính hàm merit ban đầu chứ không chỉ bằng surrogate. Nếu merit giảm, kích thước bước được thu nhỏ và ứng viên được đánh giá lại; chỉ bước không làm giảm merit mới được chấp nhận. Vòng lặp dừng khi mức cải thiện tương đối nhỏ hơn ngưỡng hoặc đạt số vòng tối đa. Vì vậy, AO--SCA là quá trình có hướng gồm khởi tạo, cập nhật luân phiên, xấp xỉ cục bộ, chiếu ràng buộc và kiểm tra đơn điệu.\n\nPhương pháp này có ưu điểm tối ưu trực tiếp từng hiện thực kênh và đạt chất lượng cao trong kết quả. Tuy nhiên, mỗi gradient sai phân hữu hạn cần nhiều lần đánh giá hàm, số tọa độ tăng theo số phần tử STAR--RIS, và toàn bộ quy trình phải lặp lại khi CSI thay đổi. AO--SCA chỉ là bộ giải cục bộ phụ thuộc khởi tạo; không được xem là nghiệm tối ưu toàn cục hoặc cận trên lý thuyết.\n\n\\subsection{AO--Grid}\nAO--Grid giữ nguyên nguyên lý tối ưu luân phiên nhưng thay bước gradient bằng một codebook hữu hạn. Với từng khối hoặc từng tọa độ, thuật toán lần lượt thử các ứng viên công suất, tỷ lệ tốc độ, beta hoặc pha; mỗi ứng viên được đưa về miền khả thi và được đánh giá bằng cùng hàm merit. Ứng viên tốt nhất không làm giảm merit được giữ lại trước khi chuyển sang tọa độ kế tiếp.\n\nMột vòng quét hoàn tất khi tất cả tọa độ đã được xem xét. Thuật toán tiếp tục quét cho đến khi không còn cải thiện đáng kể hoặc đạt giới hạn vòng lặp. AO--Grid xác định và dễ kiểm tra vì không phụ thuộc gradient, nhưng chất lượng bị giới hạn bởi độ mịn codebook. Lưới thưa cho độ trễ thấp hơn nhưng dễ bỏ qua nghiệm tốt; lưới dày làm số lần đánh giá tăng rất nhanh. Trong luận văn, AO--Grid là baseline bảo thủ để minh họa đánh đổi giữa độ mịn tìm kiếm, QoS, tổng tốc độ và thời gian xử lý.\n\n'
REPRO_BLOCK='\\section{Khả năng tái lập}\nMỗi thí nghiệm lưu tên phương pháp, kích thước STAR--RIS, seed, số tương tác, siêu tham số, định danh tập kịch bản và dữ liệu chỉ số chưa làm tròn. Các tập huấn luyện, xác thực và kiểm thử được tạo độc lập; checkpoint chỉ được chọn trên tập xác thực; tập kiểm thử được dùng sau khi mô hình đã khóa và không có nhiễu khám phá.\n\nQuy trình kiểm tra xác nhận đủ số seed và kịch bản, không có giá trị không hữu hạn, các phương pháp dùng chung tập kiểm thử và bảng tổng hợp khớp với dữ liệu chi tiết. Cách mô tả này tập trung vào nguyên tắc khoa học ổn định, không phụ thuộc tên thư mục, nhánh phát triển hoặc phiên bản mã nguồn cụ thể.\n\n'
SUMMARY_BLOCK='\\section{Tổng hợp các phát hiện chính}\n\\subsection{Tính khả thi và QoS của TD3}\nTD3 tạo hành động luôn thỏa các ràng buộc hình học nhờ bộ giải mã vật lý. Trên tập kiểm thử khóa, tỷ lệ người dùng đạt QoS từ 0,9893 đến 0,9994 và xác suất toàn bộ người dùng đạt QoS từ 0,9798 đến 0,9985. Kết quả cho thấy cơ chế phần thưởng, điều khiển đối ngẫu và lựa chọn checkpoint duy trì QoS ổn định trong phạm vi mô hình đã xét.\n\n\\subsection{So sánh giữa các phương pháp DRL}\nTD3 có tổng tốc độ cao hơn DDPG và PPO tại mọi $N$. DDPG cạnh tranh ở $N=32$ và có độ trễ thấp hơn TD3, nhưng mất ổn định khi $N$ tăng. PPO ổn định ở mức hiệu năng thấp nhưng chưa đạt QoS. Vì vậy, TD3 là cấu hình DRL có khả năng mở rộng tốt nhất trong ba phương pháp được đánh giá, không phải bằng chứng rằng TD3 luôn vượt mọi cấu hình DDPG hoặc PPO.\n\n\\subsection{So sánh với các phương pháp truyền thống}\nAO--SCA đạt tổng tốc độ cao nhất và QoS rất cao nhưng có độ trễ lớn hơn TD3 từ hàng trăm đến hàng nghìn lần tùy $N$. AO--Grid đạt QoS hoàn hảo nhưng tổng tốc độ thấp và cũng chậm hơn TD3 đáng kể. AnalyticalRIS nhanh nhất nhưng không đạt QoS. Kết quả thể hiện rõ đánh đổi giữa chất lượng nghiệm và thời gian ra quyết định.\n\n\\subsection{Ảnh hưởng của số phần tử STAR--RIS}\nTổng tốc độ TD3 tăng 1,6456 bit/s/Hz giữa $N=16$ và $N=128$, trong khi QoS được giữ gần một và độ trễ chỉ tăng khoảng 0,096 ms. AO--SCA tăng chất lượng nhiều hơn nhưng độ trễ tăng mạnh; DDPG giảm ổn định; PPO gần như không cải thiện. Xu hướng này chỉ áp dụng cho bộ sinh kênh và cấu hình đã sử dụng.\n\n'

DISPLAY_RE=re.compile(r"\\begin\{subequations\}.*?\\end\{subequations\}|\\begin\{align\}.*?\\end\{align\}|\\begin\{equation\}.*?\\end\{equation\}",re.S)

def insert_explanations(path, explanations):
    text=path.read_text(encoding="utf-8")
    if MARKER in text: return
    matches=list(DISPLAY_RE.finditer(text))
    if len(matches)!=len(explanations):
        raise RuntimeError(f"{path.name}: expected {len(explanations)} formulas, found {len(matches)}")
    out=[]; pos=0
    for m,e in zip(matches,explanations):
        out.extend([text[pos:m.end()],"\n",e,"\n"]); pos=m.end()
    out.append(text[pos:])
    path.write_text(MARKER+"\n"+"".join(out),encoding="utf-8")

def revise_chapter2():
    path=THESIS/"chapter2.tex"; insert_explanations(path,CH2_EXPLANATIONS)
    text=path.read_text(encoding="utf-8")
    for old,new in {"source code và mô phỏng":"mô hình và mô phỏng","Trong implementation":"Trong cách tham số hóa được sử dụng","theo công thức source":"theo công thức số chiều đã trình bày","policy":"chính sách","action":"hành động","user":"người dùng"}.items(): text=text.replace(old,new)
    path.write_text(text,encoding="utf-8")

def revise_chapter3():
    path=THESIS/"chapter3.tex"; insert_explanations(path,CH3_EXPLANATIONS[:15])
    text=path.read_text(encoding="utf-8")
    anchor="Chi phí huấn luyện được thực hiện trước; khi kiểm thử, Actor chỉ cần một lần truyền thẳng để tạo action.\n"
    if "Lý do dùng DRL thay cho học có giám sát" not in text: text=text.replace(anchor,anchor+RATIONALE)
    for old,new in {r"\texttt{blockwise\_v2}":"cơ chế chuẩn hóa theo khối",r"\texttt{physical\_v3}":"bộ giải mã vật lý","Source dùng":"Phương pháp sử dụng","source code":"mô hình triển khai","implementation":"cấu hình triển khai","action decoder":"bộ giải mã hành động","action":"hành động","policy":"chính sách","metric":"chỉ số"}.items(): text=text.replace(old,new)
    text,count=re.subn(r"\\subsection\{AO--SCA\}.*?(?=\\subsection\{AnalyticalRIS\})",lambda _: AO_BLOCK,text,flags=re.S)
    if count!=1: raise RuntimeError(f"AO block count={count}")
    text,count=re.subn(r"\\section\{Khả năng tái lập\}.*?(?=\\section\{Kết luận chương\})",lambda _: REPRO_BLOCK,text,flags=re.S)
    if count!=1: raise RuntimeError(f"repro block count={count}")
    path.write_text(text,encoding="utf-8")

def revise_chapter4():
    path=THESIS/"chapter4.tex"; text=path.read_text(encoding="utf-8")
    reps={r"Toàn bộ kết quả được lấy từ bundle \texttt{results/six\_method\_v1} đã có audit verdict \texttt{PASS}.":"Toàn bộ kết quả được lấy từ bộ dữ liệu thực nghiệm đã được kiểm tra về tính đầy đủ, tính nhất quán và việc dùng chung tập kịch bản.",r"Scientific commit & \texttt{99318fefa53bef91fa5f105ec71ddae73fc96c39}\\\n":"",r"\tablesource{Tác giả tổng hợp từ manifest và cấu hình benchmark \texttt{six\_method\_v1} (2026)}":r"\tablesource{Tác giả tổng hợp từ cấu hình thực nghiệm của luận văn (2026)}","ba implementation đã công bố":"ba cấu hình triển khai đã đánh giá",r"\figuresource{Tác giả tổng hợp từ các tệp \texttt{VALIDATION\_RAW.csv} của bundle \texttt{six\_method\_v1} (2026)}":r"\figuresource{Tác giả tổng hợp từ dữ liệu xác thực của ba thuật toán DRL (2026)}","bảng hiệu năng đã kiểm toán":"bảng hiệu năng đã được kiểm tra","mô hình đã công bố":"mô hình đã sử dụng","implementation DRL":"cấu hình DRL","audit provenance":"quy trình kiểm tra tính nhất quán"}
    for old,new in reps.items(): text=text.replace(old,new)
    text,count=re.subn(r"\\section\{Trả lời các câu hỏi nghiên cứu\}.*?(?=\\section\{Hàm ý thực tiễn\})",lambda _: SUMMARY_BLOCK,text,flags=re.S)
    if count!=1: raise RuntimeError(f"RQ block count={count}")
    text=re.sub(r"(?im)^.*scientific commit.*\n?","",text)
    path.write_text(text,encoding="utf-8")

def revise_frontmatter_style():
    path=THESIS/"frontmatter.tex"; text=path.read_text(encoding="utf-8")
    old="\\IfFileExists{figures/uth-logo.pdf}{%\n  \\includegraphics[width=4.1cm]{figures/uth-logo.pdf}\\par\n}{%\n  \\fbox{\\parbox[c][2.4cm][c]{4.3cm}{\\centering\\bfseries LOGO UTH}}\\par\n}"
    new="\\IfFileExists{figures/logo_uth.png}{%\n  \\includegraphics[width=8.0cm]{figures/logo_uth.png}\\par\n}{%\n  \\fbox{\\parbox[c][2.4cm][c]{8.0cm}{\\centering\\bfseries LOGO UTH}}\\par\n}"
    if text.count(old)!=2: raise RuntimeError(f"logo blocks={text.count(old)}")
    path.write_text(text.replace(old,new),encoding="utf-8")
    path=THESIS/"uththesis.sty"; text=path.read_text(encoding="utf-8")
    text=text.replace(r"\newenvironment{researchquestion}{\begin{quote}\itshape}{\end{quote}}"+"\n","")
    path.write_text(text,encoding="utf-8")

def validate():
    for name in ["introduction.tex","chapter1.tex","chapter2.tex","chapter3.tex","chapter4.tex"]:
        text=(THESIS/name).read_text(encoding="utf-8").lower()
        for term in ["câu hỏi nghiên cứu","rq1","rq2","rq3","rq4","scientific commit","git commit","physical_v3","blockwise_v2"]:
            if term in text: raise RuntimeError(f"forbidden {term} in {name}")

def main():
    revise_chapter2(); revise_chapter3(); revise_chapter4(); revise_frontmatter_style(); validate(); print("review revisions applied")

if __name__=="__main__": main()
