# Mục 4 — Đánh giá Mô hình & So sánh Năng lực Tổng quát hóa

> **Tài liệu nộp cho Person 4:** Bản thảo Mục 4.1 và Mục 4.2 do **Lợi (Person 1 - Metrics & Quantitative Evaluation)** biên soạn.  
> **Dữ liệu thực nghiệm:** Tập kiểm thử độc lập `data/processed/X_test.parquet` và `y_test.parquet` (2.233 mẫu, 32 đặc trưng, 1.175 *no* / 1.058 *yes*).  
> **Mã nguồn thực thi:** `src/evaluation/metrics.py`.  
> **Artifact bàn giao:**  
> - Bảng số liệu: `reports/evaluation_benchmark.csv`  
> - Bảng dự đoán hợp nhất cho Person 2 & Person 3: `reports/test_predictions.parquet` và `reports/test_predictions.csv`  

---

## 4.1. Biện luận Khoa học về Lựa chọn Chỉ số Đánh giá

Trong các bài toán phân loại nhị phân ứng dụng trong tiếp thị tài chính ngân hàng (Bank Marketing), việc lựa chọn thước đo định lượng không chỉ dừng lại ở các chỉ số thống kê toán học đơn thuần mà phải phản ánh chính xác cấu trúc dữ liệu và bài toán kinh tế - nghiệp vụ.

### 4.1.1. Giới hạn của Accuracy và Nghịch lý Độ chính xác (Accuracy Paradox)
Độ chính xác tổng thể (Accuracy) được định nghĩa là tỷ lệ số mẫu dự đoán đúng trên toàn bộ tập dữ liệu:
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$
Trong bài toán phân loại tiếp thị, nhãn dữ liệu thường có sự mất cân bằng giữa khách hàng từ chối (*no*) và khách hàng đăng ký (*yes*). Một mô hình cơ sở ngây thơ (Dummy Baseline luôn đoán nhãn đa số *no*) có thể dễ dàng đạt Accuracy trên 52% đến 88% (tùy tỷ lệ mẫu) mà hoàn toàn không có khả năng nhận diện bất kỳ khách hàng tiềm năng nào ($TP = 0, \text{Recall} = 0$). Do đó, tối ưu Accuracy sẽ khiến mô hình bị "thiên vị" sang lớp đa số và triệt tiêu khả năng phát hiện lớp mục tiêu.

### 4.1.2. Đánh đổi Chi phí Nghiệp vụ: Precision vs. Recall
Trong chiến dịch gọi điện tiếp thị tiền gửi có kỳ hạn:
- **False Positive (Dương tính giả - Báo động nhầm):** Mô hình dự đoán khách hàng sẽ đăng ký nhưng thực tế họ từ chối. Hậu quả là nhân viên tư vấn tốn thời gian, phát sinh chi phí cước gọi và có thể gây khó chịu cho khách hàng. Chỉ số kiểm soát lỗi này là **Precision** (Độ chính xác dương):
  $$
  \text{Precision} = \frac{TP}{TP + FP}
  $$
- **False Negative (Âm tính giả - Bỏ sót cơ hội):** Mô hình dự đoán khách hàng không có nhu cầu nhưng thực tế họ sẵn sàng gửi tiết kiệm nếu được tiếp cận. Hậu quả là ngân hàng đánh mất dòng vốn huy động và khách hàng vào tay đối thủ. Chỉ số kiểm soát lỗi này là **Recall** (Độ nhạy / Độ bao phủ):
  $$
  \text{Recall} = \frac{TP}{TP + FN}
  $$

Vì ngân hàng không thể chấp nhận đánh đổi cực đoan (chỉ gọi vài khách hàng chắc chắn nhất để đạt Precision cao, hoặc gọi tràn lan tất cả mọi người để đạt Recall cao), mô hình cần một thước đo dung hòa sự đánh đổi này.

### 4.1.3. Lựa chọn Macro-$F_1$ làm Thước đo Tối ưu Cốt lõi
Điểm $F_1$ là trung bình điều hòa giữa Precision và Recall. Nhóm thống nhất lựa chọn **Macro-$F_1$** làm chỉ số mục tiêu xuyên suốt từ Tuần 3 đến Tuần 4:
$$
\text{Macro-}F_1 = \frac{F_{1, \text{no}} + F_{1, \text{yes}}}{2}
$$
Khác với Micro-$F_1$ (vốn đồng nhất với Accuracy trong bài toán 2 lớp) hay Weighted-$F_1$ (bị chi phối bởi kích thước lớp), **Macro-$F_1$ đánh trọng số bình đẳng (50 - 50) cho cả hai lớp**. Điều này buộc mọi mô hình phải phân loại tốt cả lớp thiểu số lẫn lớp đa số, ngăn chặn hiện tượng mô hình "hy sinh" lớp dương để tối ưu điểm số bề nổi.

### 4.1.4. Đánh giá Độc lập Ngưỡng: ROC-AUC và PR-AUC
Các chỉ số phân loại cứng (Accuracy, Precision, Recall, $F_1$) phụ thuộc chặt chẽ vào ngưỡng phân lớp cố định (ngưỡng mặc định $\tau = 0.5$). Tuy nhiên, trong thực tế triển khai, đội ngũ kinh doanh có thể linh hoạt tăng hoặc giảm ngưỡng tùy theo số lượng nhân sự telesale trong từng tuần:
- **ROC-AUC (Area Under the Receiver Operating Characteristic):** Đo lường năng lực phân tách tổng thể giữa phân bố điểm của hai lớp qua mọi ngưỡng, biểu diễn sự đánh đổi giữa True Positive Rate ($TPR$) và False Positive Rate ($FPR$). ROC-AUC phản ánh xác suất một mẫu dương tính ngẫu nhiên nhận được điểm dự đoán cao hơn một mẫu âm tính ngẫu nhiên.
- **PR-AUC (Precision-Recall AUC / Average Precision):** Đặc biệt hữu ích khi đánh giá chất lượng xếp hạng của lớp mục tiêu (*yes*), tập trung vào độ tập trung của các dự đoán dương tính trong các miền xác suất cao.

*(Ghi chú mở rộng: Đối với các bài toán hồi quy phụ trợ hoặc ước tính giá trị tiền gửi, hệ thống số liệu định lượng chuẩn mực bao gồm RMSE và MAE để định lượng sai số tuyệt đối/bình phương theo đơn vị tiền tệ, $R^2$ để đo lường mức độ giải thích phương sai, và MAPE để kiểm soát sai số phần trăm tương đối).*

---

## 4.2. Bảng Tổng hợp Kết quả Định lượng & Đối chiếu 2 Chiều (Generalization Gap)

Tất cả các mô hình được huấn luyện hoàn toàn trên tập Train (8.929 mẫu, 5-fold CV) ở Tuần 3 và được đánh giá mù (blind evaluation) trên tập Test độc lập (2.233 mẫu) ở Tuần 4. 

Độ chênh lệch tổng quát hóa (**Generalization Gap**) được định nghĩa:
$$
\text{Generalization Gap} = \text{Score}_{\text{CV (Train)}} - \text{Score}_{\text{Test}}
$$
- Nếu $\text{Gap} > 0.05$: Dấu hiệu quá khớp (Overfitting), mô hình ghi nhớ tập huấn luyện nhưng giảm sút rõ rệt ngoài thực tế.
- Nếu $|\text{Gap}| \le 0.02$: Mô hình có năng lực tổng quát hóa xuất sắc, độ tin cậy thực nghiệm cao.

### 4.2.1. Bảng Đối chiếu Hai Chiều Toàn diện

| Nhóm mô hình | Tên mô hình | CV Macro-$F_1$ (Tuần 3) | Test Macro-$F_1$ (Tuần 4) | Generalization Gap | Test Accuracy | Test Precision (*yes*) | Test Recall (*yes*) | Test $F_1$ (*yes*) | Test ROC-AUC | Test PR-AUC | Đánh giá Nguy cơ Quá khớp |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Baseline** | Dummy (Most Frequent) | 0.3448 | 0.3448 | $+0.0000$ | 0.5262 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.4738 | Ngưỡng cơ sở |
| **Linear** | Logistic Regression (Tuned) | 0.6960 | **0.7098** | **$-0.0137$** | 0.7147 | 0.7384 | 0.6163 | 0.6718 | 0.7605 | 0.7654 | Tổng quát hóa rất tốt (Không overfit) |
| **Tree** | Random Forest (Tuned) | **0.7233** | **0.7263** | **$-0.0030$** | **0.7313** | 0.7626 | **0.6285** | **0.6891** | **0.7747** | **0.7774** | Tổng quát hóa xuất sắc (Vững chắc) |
| **Nonlinear** | SVC RBF (Tuned) | 0.7178 | **0.7185** | **$-0.0007$** | 0.7268 | **0.7835** | 0.5851 | 0.6699 | 0.7729 | 0.7727 | Cực kỳ ổn định (Độ chính xác cao) |

*Ghi chú: Giá trị chi tiết được làm tròn 4 chữ số thập phân; dữ liệu gốc nguyên bản được lưu trong file `reports/evaluation_benchmark.csv`.*

---

### 4.2.2. Phân tích Chi tiết Từng Mô hình

1. **Dummy Baseline (Most Frequent):**
   - Dự đoán toàn bộ nhãn là `no` (lớp chiếm đa số). Đạt Accuracy 52.62% đúng bằng tỷ lệ mẫu `no` trên tập Test, nhưng Recall và $F_1$ của lớp `yes` triệt tiêu về $0.0$. Điểm Macro-$F_1$ dừng ở $0.3448$. Đây là mốc chặn dưới tuyệt đối: bất kỳ mô hình nào không vượt trội hơn mốc này đều vô giá trị trong ứng dụng thực tế.

2. **Logistic Regression (Tuned - $C=10$, L1 penalty):**
   - Đạt Test Macro-$F_1 = 0.7098$, **tăng $+0.0137$ so với điểm CV trên tập Train ($0.6960$)**. 
   - Khoảng cách tổng quát hóa mang giá trị âm nhẹ ($\text{Gap} = -0.0137$), chứng tỏ mô hình có tính ổn định cao, hoàn toàn không gặp hiện tượng quá khớp (Overfitting). Regularization L1 đã thành công trong việc loại bỏ các đặc trưng nhiễu và giữ lại vector trọng số thưa gọn, bền vững.

3. **Random Forest (Tuned - $n=300$, $\text{max\_depth}=20$, $\text{min\_samples\_split}=10$):**
   - **Đạt hiệu năng dẫn đầu toàn diện trên tập kiểm thử:** Test Macro-$F_1 = 0.7263$, Test Accuracy = $73.13\%$, Test ROC-AUC = $0.7747$, Test PR-AUC = $0.7774$, và Test Recall lớp `yes` đạt $62.85\%$.
   - **Generalization Gap gần như hoàn hảo:** Chênh lệch giữa CV ($0.7233$) và Test ($0.7263$) chỉ là **$-0.0030$** (chưa tới $0.3\%$). Mặc dù ở Tuần 3 biểu đồ learning curve cho thấy chênh lệch điểm train/val nhất định do cấu trúc cây quyết định, nhưng các ràng buộc siêu tham số $\text{max\_depth}=20$ và $\text{min\_samples\_split}=10$ cùng cơ chế tổng hợp đa số (Ensemble Bagging) đã bảo vệ mô hình tuyệt đối trước rủi ro suy giảm hiệu năng ngoài mẫu.

4. **SVC RBF (Tuned - $C=100$, $\gamma=0.01$):**
   - Đạt Test Macro-$F_1 = 0.7185$, bám sát điểm CV Tuần 3 ($0.7178$) với khoảng cách $\text{Gap} = -0.0007$ (gần như bằng 0).
   - **Đặc trưng nổi bật:** SVC đạt **Precision cao nhất trong tất cả các mô hình ($78.35\%$)**. Điều này đồng nghĩa với việc khi SVC gắn nhãn một khách hàng là "sẽ đăng ký", xác suất chính xác lên tới gần 80%. Nhược điểm là Recall đạt $58.51\%$ (thấp hơn Random Forest ~4.3%), do siêu phẳng lề tối ưu thiên hướng thận trọng đối với các vùng ranh giới quyết định có mật độ mẫu chồng lấn cao.

---

### 4.2.3. Khuyến nghị Bàn giao cho Person 4 (Lựa chọn Champion Model)
- **Ứng viên Champion số 1:** **Random Forest (Tuned)** là mô hình chiến thắng toàn diện trên tập Test ở cả chỉ số mục tiêu cốt lõi (Macro-$F_1 = 0.7263$), khả năng xếp hạng phân loại (ROC-AUC = $0.7747$, PR-AUC = $0.7774$) và khả năng phát hiện khách hàng tiềm năng (Recall = $62.85\%$).
- **Khả năng triển khai bổ trợ:** Trong trường hợp ngân hàng có ngân sách cuộc gọi giới hạn và ưu tiên tỷ lệ chuyển đổi cao nhất cho mỗi cuộc gọi tiếp cận, **SVC RBF** là phương án thay thế khả thi nhờ Precision dẫn đầu ($78.35\%$).
- Toàn bộ dữ liệu dự đoán `y_pred` và xác suất `y_prob` của 4 mô hình đã được xuất sang `reports/test_predictions.parquet` để Person 2 vẽ đồ thị (Confusion Matrices, ROC, PR Curves) và Person 3 thực hiện Error Analysis.
