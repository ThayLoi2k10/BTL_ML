# Mục 3 — Mô hình & Phương pháp (tích hợp Tuần 3)

## 3.1. Thiết lập thực nghiệm chung

Thực nghiệm dùng 8.929 mẫu và 32 đặc trưng trong X_train cùng y_train. Dữ liệu đầu vào đã được pipeline Tuần 2 mã hóa và scale, không có NaN hoặc vô cùng. Tất cả so sánh dùng StratifiedKFold 5 folds, shuffle=True, random_state=42 và F1-macro. Tập held-out của Tuần 4 được giữ riêng và không tham gia fit, tuning, so sánh, learning curves hoặc lựa chọn ứng viên.

## 3.2. Mô hình tuyến tính — Lợi

Phương pháp và diễn giải chi tiết được giữ tại reports/week3_linear_model_Loi.md. Kết quả được kế thừa gồm Dummy most-frequent 0.3448, Logistic Regression default 0.6990 và cấu hình tuned C=10, L1 đạt 0.6960 theo báo cáo. Artifact là models/best_linear_model.pkl.

## 3.3. Random Forest / Ensemble — Trí

Phương pháp và diễn giải chi tiết được giữ tại reports/week3_tree_model_ensemble_Tri.md. Random Forest default đạt 0.7124; cấu hình tuned gồm n_estimators=300, max_depth=20, min_samples_split=10 đạt 0.7233 theo báo cáo. Artifact là models/best_tree_model.pkl và có feature_importances_.

## 3.4. Nonlinear SVM — Danh

Phương pháp và diễn giải chi tiết được giữ tại reports/week3_nonlinear_model.md. SVC default đạt 0.6960 ± 0.0051; RBF SVC tuned với C=100 và gamma=0.01 đạt 0.7178 ± 0.0104 theo báo cáo. Grid Search 33 cấu hình mất xấp xỉ 320 giây. Artifact là models/best_nonlinear_model.pkl.

## 3.5. So sánh chung trên Cross-Validation

Bảng sau tái tính mean và std bằng cùng splitter trên tập Train. Điểm có thể khác ở chữ số cuối so với báo cáo đã làm tròn. N/A nghĩa là thời gian Grid Search ban đầu không được lưu; không thay bằng 0.

| Owner | Model | Configuration | Mean CV score | CV std | Best parameters | Tuning time (s) |
|---|---|---|---|---|---|---|
| Lợi | DummyClassifier | Baseline | 0.3448 | 0.0001 | {"strategy": "most_frequent"} | N/A |
| Lợi | Logistic Regression | Default | 0.6990 | 0.0073 | N/A | N/A |
| Lợi | Logistic Regression | Tuned | 0.6960 | 0.0040 | {"C": 10.0, "penalty": "l1"} | N/A |
| Trí | Random Forest | Default | 0.7124 | 0.0071 | N/A | N/A |
| Trí | Random Forest | Tuned | 0.7233 | 0.0096 | {"n_estimators": 300, "max_depth": 20, "min_samples_split": 10} | N/A |
| Danh | SVC | Default | 0.6960 | 0.0051 | {"kernel": "rbf", "C": 1.0, "gamma": "scale"} | N/A |
| Danh | SVC | Tuned | 0.7178 | 0.0104 | {"kernel": "rbf", "C": 100.0, "gamma": 0.01} | 320.0000 |

Ứng viên có điểm CV trung bình cao nhất trong phép so sánh chung là Random Forest tuned (0.7233). Đây chỉ là ứng viên CV mạnh nhất; chưa phải champion cuối cùng. Quyết định cuối chờ đánh giá held-out ở Tuần 4.


## 3.6. Learning curves — Nhân

Hình figures/05_learning_curves.png trình bày training và validation F1-macro theo kích thước dữ liệu, dùng đúng 5-fold CV trên tập Train.

- Linear — Logistic Regression: train=0.6986, validation=0.6961, gap=0.0024. Hai đường khá gần nhau: variance thấp; mức điểm có thể bị giới hạn bởi bias. Điểm validation gần bão hòa ở hai mốc cuối.
- Tree — Random Forest: train=0.8860, validation=0.7220, gap=0.1641. Khoảng cách train–validation lớn: dấu hiệu variance/overfitting. Điểm validation gần bão hòa ở hai mốc cuối.
- Nonlinear — RBF SVC: train=0.7393, validation=0.7179, gap=0.0214. Hai đường khá gần nhau: variance thấp; mức điểm có thể bị giới hạn bởi bias. Điểm validation còn tăng nhẹ; lợi ích của thêm dữ liệu có thể ở mức vừa.

Các chẩn đoán trên dựa vào khoảng cách hai đường và thay đổi ở hai mốc dữ liệu cuối; chúng mô tả xu hướng quan sát được, không khẳng định hiệu năng ngoài mẫu trước khi có đánh giá Tuần 4.
