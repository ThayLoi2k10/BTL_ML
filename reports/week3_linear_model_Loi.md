# Mục 3.2 — Mô hình tuyến tính

## 3.2. Mô hình tuyến tính

Đối với bài toán phân loại khách hàng có khả năng đăng ký tiền gửi, nhóm sử dụng **Logistic Regression** làm mô hình tuyến tính. Mô hình ước lượng xác suất mẫu thuộc lớp dương thông qua hàm sigmoid:

\[
P(y=1\mid x)=\sigma(z)=\frac{1}{1+e^{-z}},
\qquad
z=w^Tx+b
\]

Trong đó \(w\) là vector trọng số của các đặc trưng và \(b\) là hệ số chệch. Mô hình được huấn luyện bằng cách tối ưu hàm mất mát Logistic Loss, đồng thời sử dụng regularization để kiểm soát độ phức tạp của mô hình. Tham số \(C\) điều khiển mức độ regularization: \(C\) càng nhỏ thì regularization càng mạnh.

### Thiết lập thực nghiệm

Nhóm sử dụng ba mốc so sánh:

- **Dummy Baseline:** `DummyClassifier(strategy="most_frequent")`, dùng làm mức hiệu năng cơ sở.
- **Logistic Regression Default:** sử dụng cấu hình mặc định của `LogisticRegression()` trong scikit-learn.
- **Logistic Regression Tuned:** sử dụng framework tuning chung của nhóm trong `tuning.py`, tìm kiếm trên \(C \in \{0.001, 0.01, 0.1, 1, 10, 100\}\) và hai dạng regularization L1/L2. Mô hình tuned sử dụng solver `saga` và tối đa 5000 vòng lặp để bảo đảm quá trình tối ưu có đủ số vòng lặp cần thiết.

Các mô hình được đánh giá bằng **5-fold Stratified Cross-Validation**, sử dụng **F1-Macro** làm thước đo chính. Việc sử dụng cùng cách chia Cross-Validation giúp các kết quả có thể được so sánh trong cùng điều kiện đánh giá.

### Kết quả

| Mô hình | CV F1-Macro | Thiết lập chính |
|---|---:|---|
| Dummy Baseline | 0.3448 | `most_frequent` |
| Logistic Regression (Default) | 0.6990 | Cấu hình mặc định |
| Logistic Regression (Tuned) | 0.6960 | \(C=10\), L1 |

Logistic Regression cho kết quả cao hơn đáng kể so với Dummy Baseline, cho thấy các đặc trưng sau tiền xử lý chứa thông tin hữu ích cho việc phân loại. Mô hình tuned đạt F1-Macro 0.6960 với \(C=10\) và regularization L1; trong phạm vi search space được thiết lập, đây là cấu hình có điểm Cross-Validation cao nhất.

Kết quả Default đạt 0.6990, cao hơn nhẹ so với tuned. Tuy nhiên, cấu hình Default xuất hiện `ConvergenceWarning` do solver `lbfgs` đạt giới hạn mặc định 100 vòng lặp trước khi thỏa điều kiện hội tụ. Vì vậy, chênh lệch nhỏ giữa hai kết quả không nên được diễn giải đơn giản là tuning làm mô hình kém hơn. Kết quả tuned được sử dụng làm **mô hình tuyến tính tối ưu của P1** theo quy trình tuning đã thống nhất của nhóm.

Mô hình tuned được lưu tại `models/best_linear_model.pkl`; hệ số của các đặc trưng được xuất ra `reports/linear_coefficients.csv` để phục vụ phân tích khả năng giải thích của mô hình.
