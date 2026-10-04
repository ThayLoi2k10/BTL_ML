"""
tree_model.py
Nhiệm vụ: Huấn luyện Random Forest, đánh giá trước/sau tối ưu
         siêu tham số, lưu mô hình tốt nhất và xuất độ quan trọng
         của các đặc trưng.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

from src.tuning import make_cv_splitter, tune_model


def load_data(
    train_x_path="data/processed/X_train.parquet",
    train_y_path="data/processed/y_train.parquet",
):
    """Nạp dữ liệu huấn luyện đã được tiền xử lý từ data/processed."""
    if not os.path.exists(train_x_path) or not os.path.exists(train_y_path):
        raise FileNotFoundError(
            f"Không tìm thấy file huấn luyện tại "
            f"{train_x_path} hoặc {train_y_path}"
        )

    X_train = pd.read_parquet(train_x_path)
    y_train = pd.read_parquet(train_y_path)

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]

    return X_train, y_train


def evaluate_default_tree_model(X_train, y_train, cv):
    """Đánh giá Random Forest với cấu hình mặc định có random_state cố định."""
    model_default = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )

    scores = cross_validate(
        model_default,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )

    metrics = {
        "Model": "Random Forest (Default)",
        "CV_F1_Macro_Mean": float(np.mean(scores["test_score"])),
    }

    return metrics


def tune_tree_model(X_train, y_train, cv):
    """Tối ưu Random Forest thông qua framework tuning dùng chung."""
    model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    }

    result = tune_model(
        estimator=model,
        param_grid=param_grid,
        cv_splitter=cv,
        scoring_metric="f1_macro",
        X=X_train,
        y=y_train,
    )

    best_params = {
        "n_estimators": result.best_params["n_estimators"],
        "max_depth": result.best_params["max_depth"],
        "min_samples_split": result.best_params["min_samples_split"],
    }

    metrics = {
        "Model": "Random Forest (Tuned)",
        "CV_F1_Macro_Mean": float(result.best_score),
        "Best_Params": best_params,
    }

    return result.best_estimator, metrics


def extract_and_save_feature_importances(
    model,
    feature_names,
    output_path="reports/tree_feature_importances.csv",
):
    """Xuất độ quan trọng đặc trưng của Random Forest để phân tích mô hình."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    }).sort_values(
        by="Importance",
        ascending=False,
    )

    df_importances.to_csv(output_path, index=False)

    print(
        f"-> Đã xuất bảng độ quan trọng đặc trưng tại: "
        f"{output_path}"
    )

    return df_importances


def build_comparison_table(default_metrics, tuned_metrics):
    """Tổng hợp điểm Cross-Validation trước và sau tuning."""
    comparison_table = pd.DataFrame([
        {
            "Mô hình": default_metrics["Model"],
            "CV F1-Macro": round(
                default_metrics["CV_F1_Macro_Mean"],
                4,
            ),
            "Ghi chú / Tham số": "RandomForestClassifier(random_state=42)",
        },
        {
            "Mô hình": tuned_metrics["Model"],
            "CV F1-Macro": round(
                tuned_metrics["CV_F1_Macro_Mean"],
                4,
            ),
            "Ghi chú / Tham số": json.dumps(
                tuned_metrics["Best_Params"],
            ),
        },
    ])

    return comparison_table


def _comparison_table_to_markdown(comparison_table):
    """Render bảng benchmark đơn giản mà không cần tabulate."""
    lines = [
        "| Mô hình | CV F1-Macro | Ghi chú / Tham số |",
        "|---|---:|---|",
    ]
    for row in comparison_table.to_dict(orient="records"):
        lines.append(
            "| "
            f"{row['Mô hình']} | "
            f"{row['CV F1-Macro']:.4f} | "
            f"`{row['Ghi chú / Tham số']}` |"
        )
    return "\n".join(lines)


def write_tree_report(
    comparison_table,
    tuned_metrics,
    output_path="reports/week3_tree_model_ensemble.md",
):
    """Ghi nội dung báo cáo tiếng Việt cho mô hình cây và ensemble."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    best_params = tuned_metrics["Best_Params"]
    markdown_table = _comparison_table_to_markdown(comparison_table)

    content = f"""# Mục 3.3 — Mô hình cây và Ensemble

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

{markdown_table}

Cấu hình tốt nhất tìm được cho Random Forest là `{json.dumps(best_params)}`. Mô hình tuned được refit trên toàn bộ tập train thông qua `GridSearchCV` và được lưu tại `models/best_tree_model.pkl`.

Sau khi huấn luyện, nhóm xuất mảng độ quan trọng đặc trưng `feature_importances_` ra `reports/tree_feature_importances.csv`. Bảng này cho biết mức đóng góp tương đối của từng đặc trưng trong quá trình tách nhánh của các cây, hỗ trợ phân tích những biến có ảnh hưởng lớn đến quyết định phân loại của mô hình.
"""

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(content)

    print(
        f"-> Đã ghi nội dung báo cáo mô hình cây tại: "
        f"{output_path}"
    )


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("=== BẮT ĐẦU HUẤN LUYỆN TREE MODEL & ENSEMBLE ===")

    X_train, y_train = load_data()

    print(
        f"Kích thước tập Train: "
        f"{X_train.shape[0]} mẫu, "
        f"{X_train.shape[1]} đặc trưng."
    )

    cv = make_cv_splitter(
        task_type="classification",
        n_splits=5,
        random_state=42,
        shuffle=True,
    )

    print("\n1. Đang đánh giá Random Forest (Default)...")

    default_metrics = evaluate_default_tree_model(
        X_train,
        y_train,
        cv,
    )

    print("2. Đang tuning Random Forest...")

    best_tree_model, tuned_metrics = tune_tree_model(
        X_train,
        y_train,
        cv,
    )

    model_output_path = "models/best_tree_model.pkl"

    joblib.dump(
        best_tree_model,
        model_output_path,
    )

    print(
        f"-> Đã lưu mô hình tối ưu vào: "
        f"{model_output_path}"
    )

    extract_and_save_feature_importances(
        best_tree_model,
        X_train.columns,
    )

    comparison_table = build_comparison_table(
        default_metrics,
        tuned_metrics,
    )

    comparison_table.to_csv(
        "reports/tree_model_benchmark.csv",
        index=False,
    )

    write_tree_report(
        comparison_table,
        tuned_metrics,
    )

    print("\n=== KẾT QUẢ MÔ HÌNH CÂY & ENSEMBLE ===")
    print(
        comparison_table.to_string(index=False)
    )


if __name__ == "__main__":
    main()
