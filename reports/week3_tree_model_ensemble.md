# Mục 3.3 — Mô hình cây và Ensemble

## 3.3. Mô hình cây quyết định và Ensemble

Đối với nhóm mô hình phi tuyến, nhóm sử dụng **Random Forest** làm đại diện cho họ mô hình cây và ensemble. Random Forest kết hợp nhiều cây quyết định, trong đó mỗi cây được huấn luyện trên một mẫu bootstrap khác nhau của tập train. Khi dự đoán bài toán phân loại, mô hình tổng hợp kết quả bằng cơ chế bỏ phiếu đa số giữa các cây thành phần.

### Nguyên lý Bagging và giảm phương sai

Random Forest dựa trên nguyên lý **Bagging** (Bootstrap Aggregating). Thay vì huấn luyện một cây quyết định duy nhất, thuật toán tạo nhiều tập dữ liệu con bằng cách lấy mẫu có hoàn lại từ tập train, sau đó huấn luyện một cây trên từng tập con. Các cây quyết định riêng lẻ thường có phương sai cao vì chúng dễ thay đổi mạnh khi dữ liệu huấn luyện thay đổi. Khi kết hợp nhiều cây ít tương quan với nhau, sai số riêng của từng cây có xu hướng được triệt tiêu một phần trong quá trình bỏ phiếu, nhờ đó ensemble giúp **giảm phương sai** và hạn chế overfitting so với một cây đơn lẻ.

Ngoài bootstrap sampling, Random Forest còn chọn ngẫu nhiên một tập con đặc trưng tại mỗi lần tách node. Cơ chế này làm các cây đa dạng hơn, giảm tương quan giữa các cây và tăng hiệu quả tổng hợp của ensemble.

### Thiết lập thực nghiệm

Nhóm đánh giá hai mốc so sánh:

- **Random Forest Default:** `RandomForestClassifier(random_state=42, n_jobs=-1)`.
- **Random Forest Tuned:** sử dụng framework tuning chung trong `src/tuning.py`, tìm kiếm trên các siêu tham số `n_estimators`, `max_depth`, và `min_samples_split`.

Các mô hình được đánh giá bằng **5-fold Stratified Cross-Validation** trên tập train, sử dụng **F1-Macro** làm thước đo chính. Việc chỉ dùng tập train trong Cross-Validation giúp tránh rò rỉ thông tin từ tập test vào quá trình chọn siêu tham số.

Ba siêu tham số được chọn vì chúng kiểm soát trực tiếp độ phức tạp và khả năng overfitting của mô hình:

- `n_estimators`: số lượng cây trong rừng. Số cây lớn hơn thường giúp kết quả ổn định hơn, nhưng làm tăng thời gian huấn luyện.
- `max_depth`: độ sâu tối đa của từng cây. Giới hạn độ sâu giúp cây bớt học quá chi tiết các nhiễu trong dữ liệu.
- `min_samples_split`: số mẫu tối thiểu để tiếp tục phân nhánh một node. Giá trị lớn hơn làm cây thận trọng hơn khi tách nhánh.

### Kết quả Cross-Validation

| Mô hình | CV F1-Macro | Ghi chú / Tham số |
|---|---:|---|
| Random Forest (Default) | 0.7124 | `RandomForestClassifier(random_state=42)` |
| Random Forest (Tuned) | 0.7233 | `{"n_estimators": 300, "max_depth": 20, "min_samples_split": 10}` |

Cấu hình tốt nhất tìm được cho Random Forest là `{"n_estimators": 300, "max_depth": 20, "min_samples_split": 10}`. Mô hình tuned được refit trên toàn bộ tập train thông qua `GridSearchCV` và được lưu tại `models/best_tree_model.pkl`.

Sau khi huấn luyện, nhóm xuất mảng độ quan trọng đặc trưng `feature_importances_` ra `reports/tree_feature_importances.csv`. Bảng này cho biết mức đóng góp tương đối của từng đặc trưng trong quá trình tách nhánh của các cây, hỗ trợ phân tích những biến có ảnh hưởng lớn đến quyết định phân loại của mô hình.
