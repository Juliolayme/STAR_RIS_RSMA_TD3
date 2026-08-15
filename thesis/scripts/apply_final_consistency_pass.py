from __future__ import annotations

"""Apply the final thesis consistency fixes identified by code-theory audit.

This pass is intentionally idempotent. It fixes only stable scientific/documentation
inconsistencies; it does not alter raw experimental results or retrain any method.
"""

from pathlib import Path


THESIS = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one old block, found {count}")
    return text.replace(old, new, 1)


def replace_all(text: str, old: str, new: str) -> str:
    return text.replace(old, new)


def update(path: Path, transform) -> None:
    text = path.read_text(encoding="utf-8")
    updated = transform(text)
    if updated != text:
        path.write_text(updated, encoding="utf-8")
        print(f"updated {path.relative_to(THESIS.parent)}")
    else:
        print(f"unchanged {path.relative_to(THESIS.parent)}")


def chapter2(text: str) -> str:
    old = (
        "Trong một số kịch bản, miền QoS có thể khó đạt đối với một phương pháp nhất định. "
        "Vì vậy, quá trình huấn luyện và các baseline dùng hàm merit mềm\n"
    )
    new = (
        "Trong một số kịch bản, miền QoS có thể khó đạt đối với một phương pháp nhất định. "
        "Mọi phương pháp đều đánh giá QoS bằng cùng các đại lượng vi phạm $V$ và $V_2$, nhưng "
        "cơ chế sử dụng trọng số phạt khác nhau giữa hai nhóm. Trong huấn luyện DRL, trọng số "
        "phạt tuyến tính được điều chỉnh thích nghi theo quá trình học; các solver truyền thống "
        "dùng một hàm merit với trọng số cố định. Dạng tổng quát của hàm merit là\n"
    )
    text = replace_once(text, old, new, label="chapter2 merit intro")

    old = (
        "Trong đó, $f$ là giá trị merit; $\\lambda_1,\\lambda_2\\ge0$ là trọng số phạt tuyến tính "
        "và bậc hai; $V$ là tổng vi phạm tuyến tính. Công thức cân bằng mục tiêu tổng tốc độ với "
        "mức độ vi phạm QoS, tạo tín hiệu tối ưu liên tục ngay cả khi nghiệm hiện tại chưa khả thi. "
        "Các bảng kết quả vẫn báo cáo riêng đại lượng vật lý, không dùng merit thay thế tổng tốc độ hoặc QoS."
    )
    new = (
        "Trong đó, $f$ là giá trị merit; $\\lambda_1,\\lambda_2\\ge0$ là trọng số phạt tuyến tính "
        "và bậc hai; $V$ là tổng vi phạm tuyến tính. Đối với các solver truyền thống trong thí nghiệm, "
        "$\\lambda_1=\\lambda_2=8$. Đối với TD3, DDPG và PPO, $\\lambda_1$ được thay bởi trọng số "
        "thích nghi $\\lambda_t$, còn $\\lambda_2=8$. Công thức cân bằng mục tiêu tổng tốc độ với "
        "mức độ vi phạm QoS và tạo tín hiệu liên tục ngay cả khi nghiệm hiện tại chưa khả thi. Các "
        "bảng kết quả vẫn báo cáo riêng đại lượng vật lý, không dùng reward hoặc merit thay thế tổng tốc độ hay QoS."
    )
    return replace_once(text, old, new, label="chapter2 merit explanation")


def chapter3(text: str) -> str:
    old = (
        "Về hình thức, framework sử dụng quá trình quyết định Markov (Markov Decision Process, MDP). "
        "Tuy nhiên, kênh kế tiếp được lấy mẫu độc lập và không phụ thuộc hành động hiện tại. Do đó, "
        "bài toán gần với tối ưu theo ngữ cảnh hơn điều khiển động lực dài hạn. Hệ số chiết khấu vẫn "
        "được giữ theo thuật toán chuẩn, nhưng kết quả không được diễn giải như chiến lược điều khiển kênh theo thời gian."
    )
    new = old + (
        "\n\nTrong triển khai huấn luyện, một episode gồm 32 bước và cờ kết thúc được bật ở bước thứ 32. "
        "Đây là cấu trúc kỹ thuật để ngắt thành phần bootstrap trong mục tiêu Critic; nó không biểu diễn "
        "coherence time, chuyển động người dùng hay một quỹ đạo kênh vật lý. Mỗi bước trong episode vẫn "
        "lấy một mẫu kênh độc lập, vì vậy độ dài episode không làm thay đổi cách diễn giải bài toán như "
        "một bộ tối ưu học được theo ngữ cảnh."
    )
    text = replace_once(text, old, new, label="chapter3 episode explanation")

    old = "Với $K=4$, $d_s=8N+12$, tương ứng 140 chiều tại $N=16$ và 1.036 chiều tại $N=128$."
    new = (
        "Với $K=4$, công thức trên rút gọn thành $d_s=10N+12$, tương ứng 172 chiều tại $N=16$ "
        "và 1.292 chiều tại $N=128$. Khối chỉ báo phía người dùng vẫn được giữ trong trạng thái để "
        "biểu diễn rõ miền truyền qua/phản xạ; trong các thí nghiệm hiện tại, phép gán phía được cố định "
        "nên khối $\\widetilde{\\bm u}$ là một đặc trưng không đổi giữa các kịch bản."
    )
    text = replace_once(text, old, new, label="chapter3 state dimension")

    old = (
        "Thành phần tổng tốc độ khuyến khích chất lượng nghiệm; $V_t$ và $V_{2,t}$ phạt vi phạm QoS. "
        "Các chỉ số vật lý vẫn được lưu riêng, vì hai chính sách có reward gần nhau có thể có cấu trúc "
        "sum-rate và QoS khác nhau."
    )
    new = old + (
        "\n\nMôi trường đồng thời tính một reward tham chiếu với trọng số phạt tuyến tính cố định để phục vụ "
        "đánh giá thống nhất. Tuy nhiên, reward thực sự được lưu vào Replay Buffer và dùng để cập nhật "
        "TD3/DDPG là reward thích nghi trong \eqref{eq:reward-ch3}, với $\\lambda_t$ do bộ điều khiển QoS "
        "cập nhật. PPO cũng sử dụng cùng reward thích nghi trong rollout. Vì vậy, cần phân biệt reward "
        "tham chiếu của môi trường với reward huấn luyện của ba thuật toán DRL."
    )
    text = replace_once(text, old, new, label="chapter3 reward distinction")

    text = replace_all(text, "DDPG dùng cùng trạng thái, bộ giải mã hành động, reward, ScenarioBank và ngân sách tương tác với TD3.",
                       "DDPG dùng cùng trạng thái, bộ giải mã hành động, reward QoS thích nghi, ScenarioBank và ngân sách tương tác với TD3.")
    text = replace_all(text, "Implementation dùng sai số bình phương trung bình", "Cấu hình DDPG dùng sai số bình phương trung bình")
    text = replace_all(text, "PPO dùng Actor Gaussian và một mạng giá trị. Action được lấy mẫu", "PPO dùng Actor Gaussian và một mạng giá trị. Hành động được lấy mẫu")

    old = (
        "AnalyticalRIS căn chỉnh một vector pha theo kênh ghép tầng tổng hợp, đặt beta bằng 0,5 và chia đều "
        "công suất cùng tốc độ chung. Đây là tham chiếu nhanh, không phải nghiệm phân tích tối ưu của bài toán đầy đủ."
    )
    new = old + (
        "\n\nBa phương pháp truyền thống dùng cùng mô hình vật lý và cùng miền hành động với nhóm DRL, bao gồm "
        "quy ước sử dụng toàn bộ ngân sách công suất $\\mathbf{1}^T\\bm p=P_{\\max}$. Khác với reward "
        "huấn luyện DRL có $\\lambda_t$ thích nghi, AO--SCA và AO--Grid tối ưu merit của môi trường với "
        "trọng số cố định $\\lambda_1=\\lambda_2=8$; AnalyticalRIS không tối ưu merit mà chỉ đánh giá "
        "một cấu hình heuristic cố định theo từng kênh. Vì vậy, tính công bằng được hiểu là cùng mô hình "
        "vật lý, cùng miền khả thi và cùng tập kiểm thử, không phải mọi phương pháp sử dụng cùng một cơ chế cập nhật reward."
    )
    text = replace_once(text, old, new, label="chapter3 conventional reward clarification")

    old = (
        "Mỗi giá trị $N$ có 10.000 kịch bản huấn luyện, 1.000 kịch bản xác thực và 1.000 kịch bản kiểm thử. "
        "Ba tập được tạo độc lập, kiểm tra không trùng bằng fingerprint và lưu checksum. Tất cả phương pháp "
        "được đánh giá trên cùng tập kiểm thử."
    )
    new = (
        "Mỗi giá trị $N$ có 10.000 kịch bản huấn luyện, 1.000 kịch bản xác thực và 1.000 kịch bản kiểm thử. "
        "Ba tập được sinh bằng các seed khác nhau, sau đó kiểm tra không có mẫu trùng khớp chính xác bằng "
        "fingerprint và lưu checksum để truy vết. Kiểm tra này chứng minh không có bản sao kịch bản giữa các "
        "tập, nhưng không được diễn giải như một chứng minh độc lập thống kê tuyệt đối. Tất cả phương pháp "
        "được đánh giá trên cùng tập kiểm thử."
    )
    text = replace_once(text, old, new, label="chapter3 scenario bank wording")

    text = replace_all(text, "\\State Tính reward theo \\eqref{eq:reward-ch3} và lưu transition vào $\\mathcal D$",
                       "\\State Tính reward huấn luyện thích nghi theo \\eqref{eq:reward-ch3} và lưu transition vào $\\mathcal D$")
    text = replace_all(text, "Chương 4 sử dụng bundle sáu phương pháp đã kiểm toán để đánh giá kết quả.",
                       "Chương 4 sử dụng bộ kết quả sáu phương pháp đã được kiểm tra để đánh giá kết quả.")
    return text


def chapter4(text: str) -> str:
    text = replace_all(text, "action decoder", "bộ giải mã hành động")
    text = replace_all(text, "policy khả thi", "chính sách khả thi")
    text = replace_all(text, "action có số chiều", "hành động có số chiều")
    text = replace_all(text, "input/output", "đầu vào/đầu ra")
    text = replace_all(text, "0,9893", "0,9892")

    old = (
        "Từ $N=64$ trở lên, DDPG suy giảm rõ: xác suất toàn bộ người dùng đạt QoS giảm từ 0,6234 xuống "
        "0,1376 và 0,0040. Tại $N=128$, mức vi phạm trung bình đạt 0,5512. Trong khi đó TD3 giữ xác suất "
        "toàn bộ người dùng đạt QoS 0,9984. Kết quả cho thấy TD3 có khả năng mở rộng và ổn định tốt hơn "
        "trong ba cấu hình DRL được đánh giá."
    )
    new = (
        "Từ $N=64$ trở lên, cấu hình DDPG được đánh giá suy giảm rõ: xác suất toàn bộ người dùng đạt QoS "
        "giảm từ 0,6234 xuống 0,1376 và 0,0040. Tại $N=128$, mức vi phạm trung bình đạt 0,5512. Trong khi "
        "đó TD3 giữ xác suất toàn bộ người dùng đạt QoS 0,9984. Trong phạm vi $N\\le128$, kết quả quan sát "
        "cho thấy cấu hình TD3 ổn định hơn khi kích thước bài toán tăng. Đây là mối liên hệ thực nghiệm; do "
        "TD3 và DDPG còn khác về Critic, loss, Layer Normalization, gradient clipping, learning rate và lịch "
        "cập nhật, không thể quy sự suy giảm của DDPG chỉ cho số chiều hành động."
    )
    text = replace_once(text, old, new, label="chapter4 DDPG causal wording")

    old = (
        "Độ trễ trung bình của TD3 tăng từ 0,246 ms tại $N=16$ lên 0,342 ms tại $N=128$. DDPG nhanh hơn "
        "TD3 một lượng nhỏ, từ 0,220 đến 0,307 ms, do chỉ có một Actor tương tự về kích thước và không có "
        "khác biệt lớn trong suy luận. PPO chậm hơn, từ 0,382 đến 0,521 ms, nhưng vẫn ở dưới một mili giây."
    )
    new = (
        "Độ trễ trung bình của TD3 tăng từ 0,246 ms tại $N=16$ lên 0,342 ms tại $N=128$. DDPG nhanh hơn "
        "TD3 một lượng nhỏ, từ 0,220 đến 0,307 ms; PPO chậm hơn, từ 0,382 đến 0,521 ms, nhưng vẫn ở dưới "
        "một mili giây. Phép đo kiểm thử của TD3 và DDPG chủ yếu chạy Actor rồi tính các chỉ số vật lý; hai "
        "Critic của TD3 không nằm trên đường suy luận. Vì vậy, chênh lệch nhỏ giữa TD3 và DDPG nên được xem "
        "là hệ quả của kiến trúc Actor và overhead triển khai, không phải do TD3 phải chạy hai Critic khi kiểm thử."
    )
    text = replace_once(text, old, new, label="chapter4 inference latency")

    text = replace_all(text,
        "Các tỷ lệ chỉ áp dụng cho CPU runner và implementation đã công bố. Chúng không phải cam kết latency trên mọi phần cứng hoặc khi tích hợp đầy đủ quy trình ước lượng CSI và báo hiệu.",
        "Các tỷ lệ chỉ áp dụng cho nền tảng CPU một luồng và cấu hình đo đã sử dụng. Chúng không phải cam kết độ trễ đầu-cuối trên mọi phần cứng và chưa bao gồm ước lượng CSI, truyền báo hiệu hoặc thời gian điều khiển phần cứng STAR--RIS."
    )

    old = (
        "Bảng~\\ref{tab:td3-paired-tests-holm} thực hiện 25 so sánh liên quan trực tiếp đến TD3: năm phương pháp "
        "đối chiếu tại năm giá trị $N$. Cả kiểm định $t$ ghép cặp và Wilcoxon ghép cặp sau hiệu chỉnh Holm đều "
        "bác bỏ giả thuyết không ở mức $\\alpha=0{,}05$ cho 25/25 trường hợp."
    )
    new = (
        "Bảng~\\ref{tab:td3-paired-tests-holm} xác định trước một họ 25 giả thuyết liên quan trực tiếp đến TD3, "
        "gồm năm phương pháp đối chiếu tại năm giá trị $N$. Đối với kiểm định $t$ ghép cặp, Holm được áp dụng "
        "một lần trên toàn bộ 25 $p$-value của họ này; Wilcoxon cũng được hiệu chỉnh độc lập trên đúng 25 "
        "$p$-value tương ứng. Sau hiệu chỉnh, cả hai kiểm định đều bác bỏ giả thuyết không ở mức "
        "$\\alpha=0{,}05$ cho 25/25 trường hợp."
    )
    text = replace_once(text, old, new, label="chapter4 Holm family")

    old = (
        "Khi $N$ tăng từ 16 lên 128, chiều trạng thái tăng từ 140 lên 1.036 và chiều action tăng từ 57 lên 393. "
        "TD3 vẫn duy trì QoS gần một và tổng tốc độ tăng. DDPG lại suy giảm khi action lớn, trong khi PPO giữ hiệu năng thấp."
    )
    new = (
        "Khi $N$ tăng từ 16 lên 128, chiều trạng thái tăng từ 172 lên 1.292 và chiều hành động tăng từ 57 lên 393. "
        "TD3 vẫn duy trì QoS gần một và tổng tốc độ tăng trong phạm vi khảo sát. Cấu hình DDPG được đánh giá suy "
        "giảm khi kích thước bài toán tăng, trong khi PPO giữ hiệu năng thấp; các xu hướng này là quan sát thực nghiệm "
        "trên $N\\le128$, không phải bằng chứng nhân quả riêng cho số chiều hành động."
    )
    text = replace_once(text, old, new, label="chapter4 scalability dimensions")

    return text


def appendix_a(text: str) -> str:
    replacements = {
        "16 & 140 & 57 & 48\\\\": "16 & 172 & 57 & 48\\\\",
        "32 & 268 & 105 & 96\\\\": "32 & 332 & 105 & 96\\\\",
        "64 & 524 & 201 & 192\\\\": "64 & 652 & 201 & 192\\\\",
        "96 & 780 & 297 & 288\\\\": "96 & 972 & 297 & 288\\\\",
        "128 & 1036 & 393 & 384\\\\": "128 & 1292 & 393 & 384\\\\",
    }
    for old, new in replacements.items():
        text = replace_all(text, old, new)

    anchor = "Số bước hành động ngẫu nhiên ban đầu & 2.000\\\\\n"
    addition = (
        "Số bước hành động ngẫu nhiên ban đầu & 2.000\\\\\n"
        "Độ dài episode kỹ thuật & 32 bước; dùng để tạo cờ kết thúc và ngắt bootstrap, không mô tả coherence time của kênh\\\\\n"
    )
    if "Độ dài episode kỹ thuật" not in text:
        text = replace_once(text, anchor, addition, label="appendix episode row")
    return text


def conclusion(text: str) -> str:
    text = replace_all(text, "0,9893", "0,9892")
    text = replace_all(text, "implementation", "cấu hình triển khai")
    return text


def validate() -> None:
    checks = {
        "chapter3.tex": ["d_s=10N+12", "172 chiều", "1.292 chiều", "episode gồm 32 bước", "reward tham chiếu"],
        "chapter4.tex": ["chiều trạng thái tăng từ 172 lên 1.292", "một họ 25 giả thuyết", "hai Critic của TD3 không nằm trên đường suy luận"],
        "appendixA.tex": ["16 & 172 & 57 & 48", "128 & 1292 & 393 & 384", "Độ dài episode kỹ thuật"],
    }
    for name, required in checks.items():
        text = (THESIS / name).read_text(encoding="utf-8")
        for needle in required:
            if needle not in text:
                raise RuntimeError(f"{name}: missing expected final text: {needle}")

    forbidden = {
        "chapter3.tex": ["d_s=8N+12", "140 chiều tại $N=16$", "1.036 chiều tại $N=128$"],
        "chapter4.tex": ["chiều trạng thái tăng từ 140 lên 1.036", "DDPG lại suy giảm khi action lớn", "0,9893"],
        "conclusion.tex": ["0,9893"],
        "appendixA.tex": ["16 & 140 & 57", "128 & 1036 & 393"],
    }
    for name, needles in forbidden.items():
        text = (THESIS / name).read_text(encoding="utf-8")
        for needle in needles:
            if needle in text:
                raise RuntimeError(f"{name}: stale text remains: {needle}")


def main() -> None:
    update(THESIS / "chapter2.tex", chapter2)
    update(THESIS / "chapter3.tex", chapter3)
    update(THESIS / "chapter4.tex", chapter4)
    update(THESIS / "appendixA.tex", appendix_a)
    update(THESIS / "conclusion.tex", conclusion)
    validate()
    print("final thesis consistency pass: PASS")


if __name__ == "__main__":
    main()
