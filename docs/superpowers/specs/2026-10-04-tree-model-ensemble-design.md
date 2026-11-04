# Thiết kế: Huấn luyện mô hình Cây Quyết định & Ensemble

Ngày: 2026-10-04

## Mục tiêu

Bổ sung phần huấn luyện mô hình họ cây cho bài toán phân loại hiện tại của project. Phần này dùng dữ liệu huấn luyện đã chuẩn hóa trong `data/processed/`, huấn luyện và tối ưu `RandomForestClassifier`, lưu mô hình tốt nhất, xuất độ quan trọng đặc trưng, và tạo nội dung báo cáo tiếng Việt cho mục mô hình cây/ensemble.

## Phạm vi

Triển khai trong một module độc lập `src/models/tree_model.py`, đi theo cấu trúc và phong cách của `src/models/linear_model.py`. Không thay đổi pipeline tiền xử lý, không dùng tập test cho tuning, và không mở rộng sang các mô hình cây khác ngoài Random Forest trong lần triển khai này.

## Thiết kế kỹ thuật

### Dữ liệu đầu vào

Module sẽ nạp:

- `data/processed/X_train.parquet`
- `data/processed/y_train.parquet`

`y_train` sẽ được chuyển từ `DataFrame` một cột sang `Series` nếu cần, giống pattern hiện có trong `linear_model.py`.

### Cross-Validation

Dùng splitter chung từ `src/tuning.py`:

```python
make_cv_splitter(
    task_type="classification",
    n_splits=5,
    random_state=42,
    shuffle=True,
)
```

Metric chính là `f1_macro` để thống nhất với mô hình tuyến tính hiện có.

### Mô hình mặc định

Đánh giá mốc trước tuning bằng:

```python
RandomForestClassifier(random_state=42, n_jobs=-1)
```

Kết quả được tính bằng `cross_validate` trên train set với 5-fold Stratified CV.

### Tuning mô hình

Dùng framework chung `tune_model(...)` trong `src/tuning.py` với `GridSearchCV`. Không gian siêu tham số:

```python
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
}
```

Các tham số này trực tiếp kiểm soát khả năng overfitting:

- `n_estimators`: số lượng cây trong ensemble; nhiều cây hơn thường ổn định hơn nhưng tốn thời gian hơn.
- `max_depth`: giới hạn độ sâu từng cây; giá trị nhỏ hơn làm cây đơn giản hơn.
- `min_samples_split`: yêu cầu số mẫu tối thiểu để tiếp tục tách node; giá trị lớn hơn giúp giảm tách quá chi tiết.

`GridSearchCV` sẽ refit estimator tốt nhất trên toàn bộ tập train, nên estimator trả về có thể lưu trực tiếp.

### Đầu ra

Sau khi chạy:

```bash
python -m src.models.tree_model
```

module sẽ tạo hoặc cập nhật:

- `models/best_tree_model.pkl`: mô hình Random Forest tốt nhất sau tuning.
- `reports/tree_model_benchmark.csv`: bảng so sánh Cross-Validation trước và sau tuning.
- `reports/tree_feature_importances.csv`: bảng `feature_importances_` theo tên đặc trưng, sắp xếp giảm dần.
- `reports/week3_tree_model_ensemble.md`: đoạn báo cáo hoàn chỉnh bằng tiếng Việt cho mục mô hình cây/ensemble.

## Cấu trúc module đề xuất

`src/models/tree_model.py` sẽ có các hàm:

- `load_data(...)`: nạp train features/labels.
- `evaluate_default_tree_model(X_train, y_train, cv)`: đánh giá Random Forest mặc định.
- `tune_tree_model(X_train, y_train, cv)`: tuning Random Forest qua `src.tuning.tune_model`.
- `extract_and_save_feature_importances(model, feature_names, output_path=...)`: xuất độ quan trọng đặc trưng.
- `write_tree_report(comparison_table, tuned_metrics, output_path=...)`: sinh báo cáo tiếng Việt.
- `main()`: điều phối toàn bộ quy trình, in kết quả ra terminal và ghi file.

## Nội dung báo cáo

Báo cáo `reports/week3_tree_model_ensemble.md` sẽ gồm:

1. Nguyên lý Random Forest và Bagging.
2. Cơ chế giảm phương sai khi kết hợp nhiều cây huấn luyện trên bootstrap samples.
3. Ý nghĩa các siêu tham số kiểm soát overfitting.
4. Bảng kết quả Cross-Validation của Random Forest trước và sau tuning.
5. Nhận xét kết quả, best params, vị trí file model và file feature importances.

## Kiểm thử và xác minh

Sau khi triển khai sẽ chạy:

```bash
python -m src.models.tree_model
```

Nếu project có test phù hợp hoặc import check nhanh, sẽ chạy thêm để đảm bảo module mới không phá vỡ code hiện có. Kết quả thành công cần xác nhận các file đầu ra tồn tại và bảng benchmark có đủ hai dòng: default và tuned.

## Ngoài phạm vi

- Không dùng tập test để chọn siêu tham số.
- Không thêm mô hình Gradient Boosting, XGBoost, LightGBM hoặc DecisionTree riêng lẻ.
- Không sửa pipeline tiền xử lý hoặc schema dữ liệu processed.
- Không commit thay đổi nếu chưa có yêu cầu rõ ràng từ người dùng.
