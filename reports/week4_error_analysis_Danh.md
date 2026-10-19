# Mục 5.2 — Phân tích Lỗi Dự đoán & Điểm mù Mô hình (Error Analysis)

> **Tài liệu nộp cho Person 4:** Bản thảo Mục 5.2 do **Danh (Person 3 - Error Analysis)** biên soạn.
> **Đầu vào:** Bảng dự đoán hợp nhất của Person 1 `reports/test_predictions.parquet` (2.233 mẫu Test, 3 mô hình + Dummy) và dữ liệu thô `data/raw/bank.csv`.
> **Mã nguồn:** `src/evaluation/error_analysis.py` — chạy lại toàn bộ bằng `python -m src.evaluation.error_analysis`.
> **Artifact bàn giao:** thư mục `reports/error_analysis/` (18 bảng CSV), trong đó quan trọng nhất:
> - `misclassified_tree.csv`, `misclassified_linear.csv`: toàn bộ mẫu dự đoán sai, đã ánh xạ về giá trị thô.
> - `largest_residuals_tree.csv`: 20 mẫu có sai số thô $|y - \hat{p}|$ lớn nhất.
> - `group_profile_tree.csv`, `flag_error_rates_tree.csv`, `segment_*_tree.csv`: thống kê đặc điểm FP/FN.
> - `head_to_head.csv`, `tree_right_linear_wrong.csv`, `linear_right_tree_wrong.csv`: so sánh đối đầu.
> - `representative_errors.csv`: 10 mẫu sai tiêu biểu kèm chẩn đoán.

Mô hình được phân tích chính là **Random Forest (Tuned)** — ứng viên Champion theo Mục 4.2.3 — đối chiếu với Logistic Regression (Tuned) và SVC RBF (Tuned). Lớp dương là `yes` (khách hàng đăng ký tiền gửi), ngưỡng phân lớp $\tau = 0.5$.

---

## 5.2.1. Phương pháp

1. **Trích xuất lỗi.** Với mỗi mô hình, mỗi mẫu Test được gán nhãn TP/TN/FP/FN. Hàm `extract_misclassified()` lọc toàn bộ các dòng $y_{\text{test}} \neq y_{\text{pred}}$, sắp theo mức độ "tự tin khi sai" $|\hat{p} - 0.5|$. Vì đây là bài toán phân loại, phần "sai số thô lớn nhất" được tính trên thang xác suất: $r_i = |y_i - \hat{p}_i|$ (`largest_residuals()`); hàm `largest_absolute_errors()` viết dạng tổng quát nên dùng được trực tiếp cho bài toán hồi quy.
2. **Ánh xạ ngược về dữ liệu thô.** `X_test.parquet` đã bị `reset_index`, nên chỉ số gốc bị mất. Module tái lập chính xác phép chia của Tuần 2 (`split_data`, `test_size=0.2`, `stratify=True`, `random_state=42`) trên `bank.csv`, rồi **kiểm chứng** nhãn khớp từng dòng với `y_test.parquet` và với cột `y_true` của Person 1 (nếu lệch, chương trình dừng). Khi biến đổi lại tập thô bằng `preprocessing_pipeline.pkl`, kết quả trùng khớp `X_test.parquet` (`np.allclose = True`). Nhờ vậy mỗi lỗi được gắn với `raw_index` — số thứ tự dòng trong `bank.csv` — cùng giá trị thật trước khi làm sạch/clip/scale.
3. **Gắn cờ các giả thuyết lỗi** cho từng mẫu:
   - *Nhóm hiếm:* thuộc một hạng mục có tỉ lệ < 5% trong tập Train (ví dụ `job=student`, `month=mar`, `poutcome=other`).
   - *Sát biên quyết định:* $|\hat{p} - 0.5| < 0.1$.
   - *Nhiễu do điền khuyết:* giá trị `unknown` ở `job/education/contact/poutcome` bị `MissingHandler` gộp thành một mức `missing`.
   - *Nhiễu do cắt outlier:* giá trị nằm ngoài ngưỡng 1.5×IQR học trên Train và bị `OutlierClipper` cắt (ví dụ `balance > 4052` bị gán bằng 4052).
   - *Chồng lấn lớp (kNN):* tỉ lệ nhãn `yes` trong 15 láng giềng gần nhất **thuộc tập Train**, đo trong đúng không gian 32 đặc trưng mà mô hình nhìn thấy. Nếu đa số láng giềng mang nhãn ngược với nhãn thật, mẫu nằm trong vùng hai lớp chồng lên nhau.
4. **Kiểm tra hậu kiểm bằng `duration`.** `duration` bị loại khỏi huấn luyện vì chỉ biết được sau cuộc gọi (leakage). Ở đây nó **chỉ** được dùng để giải thích lỗi, không đưa vào mô hình.
5. **So sánh đối đầu** giữa các họ thuật toán bằng bảng 2×2 đúng/sai và **kiểm định McNemar chính xác** (nhị thức hai phía trên các cặp bất đồng).

---

## 5.2.2. Tổng quan lỗi trên tập Test

| Mô hình | TP | TN | FP | FN | Tổng lỗi | Tỉ lệ lỗi | FN / tổng lỗi | Lỗi sát biên | Lỗi tự tin cao ($\lvert\hat{p}-0.5\rvert \ge 0.3$) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 652 | 944 | 231 | 406 | 637 | 28.5% | 63.7% | 43.8% | 11.0% |
| **Random Forest** | 665 | 968 | 207 | 393 | **600** | **26.9%** | 65.5% | 30.0% | 16.8% |
| SVC RBF | 619 | 1004 | 171 | 439 | 610 | 27.3% | 72.0% | 18.7% | 4.8% |

Nhận xét:
- **Lỗi chủ yếu là FN (bỏ sót khách hàng tiềm năng):** chiếm 64–72% tổng lỗi ở cả ba mô hình, dù tập Test gần cân bằng (1.058 yes / 1.175 no). Các mô hình đều thiên về dự đoán `no` khi thiếu tín hiệu rõ ràng; SVC nghiêng mạnh nhất (Precision cao nhất nhưng FN nhiều nhất), khớp với nhận xét ở Mục 4.2.2.
- **Phần lớn lỗi là lỗi chung:** 431 mẫu bị **cả ba** mô hình đoán sai, tức **71.8% số lỗi của Random Forest** cũng là lỗi của hai mô hình còn lại. Ba họ thuật toán có giả thiết cấu trúc rất khác nhau mà vẫn sai cùng chỗ, nên phần lớn lỗi bắt nguồn từ **dữ liệu/đặc trưng** chứ không phải từ việc chọn thuật toán.

---

## 5.2.3. Phân tích Dương tính giả (FP) và Âm tính giả (FN)

Để công bằng, FN được so với TP (cùng nhãn thật `yes`), FP được so với TN (cùng nhãn thật `no`). Số liệu của Random Forest (`group_profile_tree.csv`):

| Đặc điểm (giá trị thô) | TP (n=665) | **FN (n=393)** | TN (n=968) | **FP (n=207)** |
|:---|---:|---:|---:|---:|
| Trung vị `balance` (€) | 1.093 | **378** | 368,5 | **949** |
| Tỉ lệ đang có khoản vay (housing/loan) | 25.3% | **68.7%** | 70.0% | **31.9%** |
| Tỉ lệ chưa từng được liên hệ (`pdays=-1`) | 51.9% | **86.5%** | 88.4% | **68.1%** |
| Tỉ lệ `poutcome=success` | 28.7% | **0.0%** | 0.0% | **8.2%** |
| Tỉ lệ `contact=unknown` | 2.3% | **23.4%** | 36.2% | **6.3%** |
| Tỉ lệ `month=may` | 10.1% | **27.5%** | 34.7% | **15.5%** |
| Xác suất dự đoán trung bình $\hat{p}$ | 0.765 | 0.316 | 0.283 | 0.667 |
| Tỉ lệ `yes` trung bình trong 15 láng giềng Train | 0.709 | **0.321** | 0.285 | **0.632** |
| **Trung vị `duration` (s) — hậu kiểm** | 314 | **732** | 172 | **185** |

### a) FN có hồ sơ gần như trùng với TN; FP có hồ sơ gần như trùng với TP
Trên mọi đặc trưng trước cuộc gọi, **nhóm FN gần như không phân biệt được với nhóm TN**: số dư thấp (378 € so với 368 €), ~70% đang vay, ~87% chưa từng được liên hệ, không ai có `poutcome=success`. Ngược lại, **nhóm FP giống nhóm TP**: số dư cao, ít vay, một phần có lịch sử thành công. Nói cách khác, mô hình đã học đúng quy luật trung bình của dữ liệu; các mẫu sai là những ngoại lệ nằm "nhầm phía" của quy luật đó.

### b) Bằng chứng định lượng về chồng lấn lớp (kNN)
- **88.5% số FN** có đa số láng giềng Train mang nhãn `no`; **74.4% số FP** có đa số láng giềng mang nhãn `yes`.
- Gộp lại, **83.7% toàn bộ lỗi** của Random Forest nằm trong vùng chồng lấn. Tỉ lệ lỗi trong vùng này là **74.9%**, so với **6.3%** ngoài vùng (gấp ~12 lần).

Lưu ý khi diễn giải: kNN cũng là một bộ phân loại cục bộ nên chỉ số này không độc lập hoàn toàn với Random Forest. Tuy vậy, nó cho thấy trong **chính dữ liệu huấn luyện**, những điểm "giống hệt" mẫu bị sai lại mang nhãn ngược lại, nên với bộ đặc trưng hiện có, không mô hình nào có thể phân tách các mẫu này một cách ổn định.

### c) Nguyên nhân gốc: thông tin quyết định chỉ xuất hiện *trong* cuộc gọi
Kiểm tra hậu kiểm bằng `duration` (không dùng để huấn luyện) cho thấy:
- **FN có cuộc gọi dài bất thường:** trung vị 732 s, cao hơn hẳn cả TP (314 s); phân vị 25% của FN (535 s) đã cao hơn trung vị TP. Đây là những khách hàng có hồ sơ "lạnh" (chưa từng liên hệ, đang vay, số dư thấp, gọi vào tháng 5) nhưng **được thuyết phục trong một cuộc trò chuyện dài**.
- **FP có cuộc gọi ngắn như TN:** trung vị 185 s so với 172 s. Đây là những khách có hồ sơ "đẹp" nhưng **từ chối nhanh**.
- Riêng `duration` đã đạt ROC-AUC = **0.803** trên tập Test, cao hơn mọi mô hình (Random Forest: 0.775).

**Kết luận:** phần lớn lỗi là **lỗi không thể khử (irreducible error)** của bài toán dự đoán *trước cuộc gọi*. Đây là cái giá có chủ đích của quyết định loại `duration` để tránh leakage, không phải khiếm khuyết của mô hình. Thêm siêu tham số hay đổi thuật toán khó cải thiện đáng kể phần lỗi này; muốn cải thiện cần **đặc trưng mới có sẵn trước cuộc gọi** (lịch sử giao dịch, thu nhập, kênh tương tác...).

---

## 5.2.4. Kiểm chứng các giả thuyết về đặc điểm chung của mẫu sai

Tỉ lệ lỗi của Random Forest khi có / không có từng đặc điểm (`flag_error_rates_tree.csv`), với *Lift* = tỉ lệ lỗi khi có cờ / tỉ lệ lỗi khi không có cờ:

| Đặc điểm | Số mẫu | Lỗi khi có | Lỗi khi không | Lift | % trên tổng lỗi | Kết luận |
|:---|---:|---:|---:|---:|---:|:---|
| Vùng chồng lấn lớp (kNN) | 670 | 74.9% | 6.3% | **11.95** | 83.7% | **Nguyên nhân chính** |
| Sát biên quyết định ($\lvert\hat{p}-0.5\rvert<0.1$) | 460 | 39.1% | 23.7% | **1.65** | 30.0% | Xác nhận |
| Chưa từng được liên hệ (`pdays=-1`) | 1.682 | 28.6% | 21.6% | **1.32** | 80.2% | Xác nhận |
| Thuộc nhóm hiếm (< 5% Train) | 764 | 25.3% | 27.7% | 0.91 | 32.2% | **Bác bỏ** |
| Điền khuyết `job/education/contact` | 544 | 23.4% | 28.0% | 0.83 | 21.2% | Bác bỏ (về tổng thể) — xem dưới |
| Bị cắt outlier (bất kỳ cột nào) | 607 | 24.1% | 27.9% | 0.86 | 24.3% | Bác bỏ |
| `balance` bị cắt về 4.052 | 246 | 26.8% | 26.9% | 1.00 | 11.0% | Không ảnh hưởng |
| `poutcome=success` | 208 | 8.2% | 28.8% | 0.28 | 2.8% | Dễ đoán, nhưng mọi lỗi đều là FP |

**1. Nhóm dữ liệu hiếm — giả thuyết bị bác bỏ.** Mẫu thuộc nhóm hiếm *không* bị sai nhiều hơn. Các tháng hiếm (`mar`, `sep`, `oct`, `dec`, mỗi tháng < 4% Train) lại có tỉ lệ lỗi thấp nhất (8–21%) vì tỉ lệ `yes` ở đó rất cao (86–92%), mô hình chỉ cần đoán `yes`. Các nghề hiếm (`self-employed` 31.9%, `unemployed` 28.2%) chỉ cao hơn trung bình vài điểm phần trăm. Một lý do là `job` và `education` **không lọt vào 32 đặc trưng** do bộ chọn Mutual Information giữ lại ở Tuần 2, nên mô hình gần như "không nhìn thấy" nghề nghiệp. Nhóm duy nhất đáng chú ý là `poutcome=other` (4.8% Train, lỗi 33.0%, cao nhất trong các hạng mục `poutcome`).

**2. Sát biên quyết định — xác nhận một phần.** 30% lỗi của Random Forest có $|\hat{p} - 0.5| < 0.1$; Logistic Regression còn cao hơn (43.8%). Phần lỗi này **có thể điều chỉnh bằng ngưỡng**: nếu ngân hàng ưu tiên Recall, hạ $\tau$ xuống 0.4–0.45 sẽ chuyển một phần FN sát biên thành TP (đổi lại thêm FP). Đây là gợi ý cho Person 4 khi chốt mô hình.

**3. Nhiễu do điền khuyết — không làm tăng lỗi tổng thể nhưng tạo một điểm mù lệch hướng.** Về tổng thể, mẫu `contact=unknown` có tỉ lệ lỗi *thấp hơn* (22.3%) vì 77% trong số đó thực sự là `no`. Nhưng lỗi trong nhóm này gần như toàn là FN (92 FN so với 13 FP). Trong 107 khách `contact=unknown` thực sự đăng ký, **Random Forest chỉ nhận ra 15 người (Recall 14.0%) và Logistic Regression chỉ 2 người (Recall 1.9%)**, so với Recall chung 62.9%. Mức `missing` sau khi điền khuyết đã trở thành một "nhãn phạt" gần như tuyệt đối: Logistic Regression gán hệ số **−0.402** cho `contact_missing`, cùng với hệ số **+0.299** cho `contact_known`. Mô hình học rằng "không rõ kênh liên lạc ⇒ không đăng ký", và đây là một **thiên lệch hệ thống**, không phải nhiễu ngẫu nhiên.

**4. Nhiễu do cắt outlier — không ảnh hưởng đáng kể.** Mẫu có `balance > 4.052` bị cắt về 4.052 có tỉ lệ lỗi 26.8%, bằng trung bình (26.9%). Việc mất thông tin ở đuôi phải không gây hại vì số dư rất cao không làm tăng thêm xác suất đăng ký so với số dư "khá cao". Mẫu bị cắt `campaign` (>6 lần gọi) còn có lỗi thấp hơn (22.1%) vì gần như chắc chắn là `no`.

**5. Thiếu lịch sử liên hệ — điểm mù lớn nhất.** 80.2% lỗi rơi vào nhóm chưa từng được liên hệ. Trong nhóm này Random Forest chỉ đạt **Recall 50.4%** và ROC-AUC 0.731, so với **Recall 85.8%** và ROC-AUC 0.819 ở nhóm đã từng liên hệ. Không có lịch sử, mô hình chỉ còn các đặc trưng nhân khẩu học/tài chính vốn có sức phân biệt yếu.

**6. `poutcome=success` — đường tắt (shortcut) dùng chung của mọi mô hình.** 208 mẫu có `poutcome=success`, trong đó 91.8% là `yes`. Cả ba mô hình đều dự đoán `yes` cho toàn bộ nhóm, nên **cả 17 khách từ chối trong nhóm đều bị cả ba mô hình đoán sai (FP)**. Đây là đặc trưng mạnh nhất (hệ số Logistic **+1.670**), đủ sức lấn át mọi tín hiệu khác.

---

## 5.2.5. So sánh đối đầu giữa các họ thuật toán

| Cặp so sánh (A vs B) | Cả hai đúng | Chỉ A đúng | Chỉ B đúng | Cả hai sai | McNemar $p$ |
|:---|---:|---:|---:|---:|---:|
| **Random Forest vs Logistic Regression** | 1.459 | **174** | 137 | 463 | **0.041** |
| SVC RBF vs Logistic Regression | 1.484 | 139 | 112 | 498 | 0.101 |
| Random Forest vs SVC RBF | 1.520 | 113 | 103 | 497 | 0.540 |

- **Random Forest tốt hơn Logistic Regression có ý nghĩa thống kê** ($p = 0.041 < 0.05$): trên 311 mẫu hai mô hình bất đồng, Cây đúng 174 lần, Tuyến tính đúng 137 lần.
- **Random Forest và SVC RBF không khác biệt có ý nghĩa** ($p = 0.54$). Hai mô hình phi tuyến sai ở những chỗ rất giống nhau; lựa chọn giữa chúng nên dựa vào bài toán nghiệp vụ (Recall hay Precision) hơn là độ chính xác.
- Các mẫu bất đồng đều có độ tự tin thấp (trung bình $|\hat{p}-0.5| \approx 0.08$–$0.12$, so với 0.21–0.25 ở mẫu cả hai đều đúng). Khác biệt giữa hai họ chỉ nằm ở **dải sát biên**, phù hợp với mức chênh Macro-F1 nhỏ (0.726 so với 0.710) ở Mục 4.2.

### a) Trường hợp Cây đúng — Tuyến tính sai: giới hạn của giả thiết cộng tính
- **Tương tác giữa các đặc trưng.** Logistic Regression cộng các hiệu ứng một cách độc lập, nên một hệ số âm lớn (`contact_missing` −0.40, `month_may` −0.42, `any_loan` −0.40) **luôn** kéo xác suất xuống, bất kể ngữ cảnh. Random Forest có thể "bỏ qua" hình phạt đó khi các nhánh khác cho tín hiệu dương. Trong 30 mẫu "Cây đúng – Tuyến tính sai" mà Tuyến tính tự tin nhất, 77% là khách `yes` bị Tuyến tính bỏ sót. Năm mẫu đứng đầu (raw_index 4482, 294, 4584, 924, 5257) **đều có `contact=unknown`**: Tuyến tính cho $\hat{p} = 0.16$–$0.25$, Cây cho $0.54$–$0.76$.
- **Quan hệ không đơn điệu của lịch sử liên hệ.** Ở nhóm khách có `poutcome ∈ {failure, other}`, tỉ lệ lỗi của Tuyến tính là 35.4%, còn của Cây là 29.8% (Cây đúng riêng 53 mẫu, Tuyến tính 34 mẫu). Ở nhóm đã từng liên hệ nói chung: 25.2% so với 21.6%. Ảnh hưởng của `pdays`/`previous` lên khả năng đăng ký không tuyến tính (gần đây vs. lâu rồi, ít vs. nhiều lần), và các phép biến đổi log/ratio ở Tuần 2 vẫn chưa đủ để mô hình tuyến tính nắm bắt.

### b) Trường hợp Tuyến tính đúng — Cây sai: giới hạn do phương sai cao
- Learning curve Tuần 3 cho thấy Random Forest có khoảng cách train–validation lớn (0.886 so với 0.722, gap 0.164). Trên tập Test, điều này thể hiện qua 137 mẫu Cây sai nhưng Tuyến tính đúng: một nửa trong 30 mẫu Cây sai tự tin nhất là khách **đã từng được liên hệ** (`pdays > 0`). Đây là vùng thưa dữ liệu (chỉ 25% mẫu), nơi các cây sâu tới 20 tầng dễ "học thuộc" vài điểm lân cận.
- Ví dụ raw_index **6365** (`poutcome=other`, `previous=3`, `balance=0`, nhãn thật `no`): 13/15 láng giềng Train là `yes`, Cây bám theo cụm cục bộ này và cho $\hat{p}=0.82$ (FP). Tuyến tính "làm trơn" toàn cục nên cho $\hat{p}=0.41$ và đoán đúng.
- Kiểu lỗi này **giảm được** bằng cách tăng `min_samples_leaf` hoặc giảm `max_depth`, đổi lấy một ít bias.

---

## 5.2.6. Bảng phân tích 10 mẫu dự đoán sai tiêu biểu

Các mẫu được chọn tự động bởi `select_representative_errors()` (mỗi "ô" lấy mẫu điển hình nhất của một kiểu lỗi), sau đó được phân tích thủ công. Cột "kNN yes" là tỉ lệ `yes` trong 15 láng giềng Train. $\hat{p}$ theo thứ tự LR / RF / SVC. Toàn bộ thuộc tính thô xem tại `reports/error_analysis/representative_errors.csv`.

| # | raw_index | Kiểu lỗi | Thuộc tính thô nổi bật | Nhãn thật | $\hat{p}$ LR / RF / SVC | kNN yes | Phân tích nguyên nhân kỹ thuật |
|:-:|:-:|:---|:---|:-:|:-:|:-:|:---|
| 1 | 60 | **FN tự tin cao** (RF) | 51 tuổi, blue-collar, `balance=7180`, housing=yes, `contact=unknown`, tháng 5, chưa từng liên hệ; *duration=927 s* | yes | 0.25 / **0.03** / 0.27 | 0.00 | Mọi tín hiệu trước cuộc gọi đều là tín hiệu `no` điển hình (tháng 5, đang vay, kênh liên lạc không rõ, không lịch sử), và cả 15 láng giềng Train đều là `no`. Thêm vào đó, `balance=7180` bị cắt về 4.052 nên mất phần thông tin số dư rất cao. Khách chỉ đồng ý sau cuộc gọi dài 15 phút, thông tin mà mô hình trước cuộc gọi không thể có. **Lỗi không thể khử.** |
| 2 | 8224 | **FP tự tin cao** (RF) | 51 tuổi, entrepreneur, `balance=2109`, không vay, `poutcome=success`, `pdays=181`, tháng 4; *duration=119 s* | no | 0.93 / **0.99** / 0.85 | 0.93 | `poutcome=success` (hệ số LR +1.67, đặc trưng mạnh nhất) cộng hồ sơ tài chính tốt khiến cả ba mô hình chắc chắn `yes`; 14/15 láng giềng Train cũng là `yes`. Khách từ chối nhanh (cuộc gọi 2 phút). Đây là **điểm mù dùng chung** của "đường tắt" `poutcome=success`. |
| 3 | 1779 | **FN sát biên** (RF) | 30 tuổi, management, tertiary, `balance=533`, không vay, cellular, tháng 5, `campaign=4`, chưa từng liên hệ | yes | 0.52 / **0.498** / 0.64 | 0.53 | Tín hiệu dương (trẻ, học vấn cao, không vay, liên lạc qua di động) và tín hiệu âm (tháng 5, gọi 4 lần, không lịch sử) gần như triệt tiêu nhau. RF chỉ thiếu 0.002 để đổi nhãn, còn LR và SVC đều đoán đúng. **Lỗi do ngưỡng**: hạ $\tau$ một chút là sửa được. |
| 4 | 7652 | **FP sát biên** (RF) | 54 tuổi, technician, `balance=3323`, housing=yes, loan=yes, cellular, tháng 4, `campaign=3`; *duration=59 s* | no | 0.45 / **0.502** / 0.31 | 0.47 | Số dư cao và tháng 4 (tỉ lệ yes 61%) đẩy xác suất lên, hai khoản vay kéo xuống. RF dừng đúng ở biên, láng giềng chia gần 50/50. LR và SVC đều đoán đúng `no`: ví dụ cho thấy **phương sai của RF ở vùng biên**. |
| 5 | 264 | **FN + nhiễu cắt outlier & điền khuyết** | 55 tuổi, blue-collar, `education=unknown`, `balance=7561`, không vay, `contact=unknown`, tháng 6; *duration=685 s* | yes | 0.38 / **0.10** / 0.28 | 0.13 | Mẫu chịu **ba lớp nhiễu tiền xử lý**: `balance=7561` bị cắt còn 4.052, `education` và `contact` đều bị gộp thành `missing`, và `education=unknown` là nhóm hiếm (4.6% Train). Mức `contact=missing` là tín hiệu âm mạnh, nên dù khách không vay và có số dư rất cao, RF vẫn cho p=0.10. |
| 6 | 9466 | **FP do `poutcome=success`** | 43 tuổi, unemployed, `balance=1854`, không vay, `poutcome=success`, `pdays=197`, tháng 11; *duration=131 s* | no | 0.85 / **0.98** / 0.85 | 0.93 | Cùng cơ chế với mẫu #2. Thêm vào đó `job=unemployed` thuộc nhóm hiếm (3.1% Train) và có tỉ lệ `yes` cao (60%), củng cố thêm dự đoán sai. Thành công ở chiến dịch trước không bảo đảm đăng ký lần này, nhưng mô hình không có thông tin nào khác để phân biệt. |
| 7 | 142 | **FN do `contact=unknown`** | 38 tuổi, admin., divorced, `balance=198`, housing=yes, `contact=unknown`, tháng 5, chưa từng liên hệ; *duration=431 s* | yes | 0.17 / **0.03** / 0.27 | 0.07 | Điển hình cho **điểm mù điền khuyết** ở mục 5.2.4: trong 107 khách `contact=unknown` đăng ký, RF chỉ bắt được 15. Mức `missing` kết hợp tháng 5 và khoản vay tạo ra hồ sơ trùng khớp với nhóm TN (14/15 láng giềng là `no`). |
| 8 | 6365 | **RF sai – LR đúng** | 50 tuổi, management, tertiary, `balance=0`, không vay, `poutcome=other`, `previous=3` (bị cắt về 2.5), `pdays=130`, tháng 8 | no | **0.41** / 0.82 / 0.68 | 0.87 | Vùng "đã từng liên hệ, kết quả khác" thưa dữ liệu, `poutcome=other` là nhóm hiếm (4.8%). RF (sâu 20 tầng) bám theo cụm láng giềng cục bộ toàn `yes` nên **quá khớp cục bộ**. LR làm trơn toàn cục (và `poutcome=other` không có cột riêng trong 32 đặc trưng được chọn) nên ra 0.41 và đúng. Minh họa cho **phương sai cao** của họ Cây. |
| 9 | 4482 | **LR sai – RF đúng** | 31 tuổi, blue-collar, housing=yes, `contact=unknown`, `poutcome=other`, `pdays=782` (bị cắt về 568), tháng 8 | yes | **0.16** / 0.54 / 0.76 | 0.73 | LR cộng dồn hai hình phạt độc lập (`contact_missing` −0.40, khoản vay −0.40) nên ra p=0.16. RF và SVC không bị ràng buộc cộng tính: tín hiệu lịch sử liên hệ (`previous=1`, `pdays>0`) và cụm láng giềng (11/15 là `yes`) có thể lấn át việc thiếu kênh liên lạc. Minh họa cho **giới hạn của giả thiết cộng tính** — mô hình tuyến tính không biểu diễn được tương tác. |
| 10 | 6652 | **Cả 3 mô hình cùng sai** (FP) | 50 tuổi, housemaid, `balance=388`, không vay, `poutcome=success`, `pdays=89`, `previous=2`, tháng 6; *duration=152 s* | no | 0.90 / 0.97 / 0.79 | 0.93 | Lỗi thuộc nhóm 431 mẫu cả ba mô hình đều sai. `poutcome=success` với lần liên hệ gần (89 ngày) là tổ hợp có tỉ lệ `yes` > 90%. Khi cả ba họ thuật toán và cả láng giềng Train (14/15 `yes`) đều đồng thuận sai, nguyên nhân nằm ở **giới hạn thông tin của dữ liệu**, không phải ở cấu trúc mô hình. |

---

## 5.2.7. Tổng kết bản chất lỗi & Khuyến nghị cho Person 4

**Phân loại bản chất lỗi của Random Forest (600 lỗi):**

| Loại lỗi | Ước lượng quy mô | Có thể khắc phục? | Hướng xử lý |
|:---|:---|:---|:---|
| Chồng lấn lớp do thiếu thông tin trước cuộc gọi | ~84% lỗi nằm trong vùng chồng lấn kNN; 72% là lỗi chung của cả 3 mô hình | Rất khó với dữ liệu hiện có | Bổ sung đặc trưng mới có sẵn trước cuộc gọi; chấp nhận như giới hạn của bài toán |
| Lỗi sát biên quyết định | 30% lỗi ($\lvert\hat{p}-0.5\rvert < 0.1$) | **Có**, bằng ngưỡng | Hiệu chỉnh $\tau$ theo chi phí FP/FN (hạ $\tau$ nếu ưu tiên Recall) |
| Thiên lệch do điền khuyết `contact=unknown` | Recall nhóm này chỉ 14% (RF) / 1.9% (LR) | Có một phần | Thay vì gộp `unknown` → `missing`, có thể thêm cờ thiếu riêng hoặc suy luận kênh liên lạc từ thông tin khác |
| Đường tắt `poutcome=success` | 17/17 khách từ chối trong nhóm bị cả 3 mô hình đoán sai | Khó | Ghi nhận như rủi ro; theo dõi khi triển khai |
| Quá khớp cục bộ của Cây ở vùng thưa | Phần lớn trong 137 mẫu "LR đúng – RF sai" | **Có** | Tăng `min_samples_leaf`, giảm `max_depth`, hoặc xem xét ensemble RF + LR |

**Khuyến nghị:**
1. **Random Forest vẫn là lựa chọn Champion hợp lý:** nó vượt Logistic Regression có ý nghĩa thống kê (McNemar $p = 0.041$) nhờ nắm bắt được tương tác. Tuy nhiên nó **không vượt SVC một cách có ý nghĩa** ($p = 0.54$), nên nếu ưu tiên Precision thì SVC là phương án thay thế tương đương về chất lượng.
2. Khi báo cáo hiệu năng, cần nói rõ rằng mức trần ~73% Accuracy chủ yếu đến từ **giới hạn thông tin của bài toán trước cuộc gọi** (FN có cuộc gọi dài gấp đôi TP; riêng `duration` đạt AUC 0.80), không phải do thiếu tinh chỉnh mô hình.
3. Lỗi chủ yếu là **FN (65%)**. Nếu mục tiêu kinh doanh là không bỏ sót khách hàng, nên chọn ngưỡng theo đường cong PR của Person 2 thay vì dùng $\tau = 0.5$ mặc định.
4. Hai điểm mù cần ghi vào phần Hạn chế của báo cáo: (i) khách **chưa từng được liên hệ** (Recall chỉ 50%); (ii) khách có **`contact=unknown`** (Recall 14%).
