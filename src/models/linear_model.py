"""
linear_model.py
Phụ trách bởi: Lợi
Nhiệm vụ: Huấn luyện Dummy Classifier và Logistic Regression,
         đánh giá trước/sau tối ưu siêu tham số, lưu mô hình
         tốt nhất và xuất hệ số của các đặc trưng.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from src.tuning import make_cv_splitter, tune_model


def load_data(
    train_x_path="data/processed/X_train.parquet",
    train_y_path="data/processed/y_train.parquet"
):
    """Nạp dữ liệu huấn luyện đã được tiền xử lý từ Tuần 2."""
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


def evaluate_dummy_baseline(X_train, y_train, cv):
    """Đánh giá mô hình cơ sở bằng chiến lược dự đoán lớp phổ biến nhất."""
    dummy = DummyClassifier(strategy="most_frequent")

    scores = cross_validate(
        dummy,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    metrics = {
        "Model": "Dummy Baseline (Most Frequent)",
        "CV_F1_Macro_Mean": np.mean(scores["test_score"])
    }

    return metrics


def evaluate_default_linear_model(X_train, y_train, cv):
    """Đánh giá Logistic Regression với cấu hình mặc định của scikit-learn."""
    model_default = LogisticRegression()

    scores = cross_validate(
        model_default,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    metrics = {
        "Model": "Logistic Regression (Default)",
        "CV_F1_Macro_Mean": np.mean(scores["test_score"])
    }

    return metrics


def tune_linear_model(X_train, y_train, cv):
    """Tối ưu Logistic Regression thông qua framework tuning dùng chung."""

    model = LogisticRegression(
        solver="saga",
        max_iter=5000,
        random_state=42
    )

    # l1_ratio = 0 tương ứng L2; l1_ratio = 1 tương ứng L1.
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "l1_ratio": [0, 1]
    }

    result = tune_model(
        estimator=model,
        param_grid=param_grid,
        cv_splitter=cv,
        scoring_metric="f1_macro",
        X=X_train,
        y=y_train
    )

    best_l1_ratio = result.best_params["l1_ratio"]

    penalty_name = (
        "l1"
        if best_l1_ratio == 1
        else "l2"
    )

    # Chuẩn hóa cách biểu diễn kết quả để thuận tiện cho báo cáo.
    best_params = {
        "C": result.best_params["C"],
        "penalty": penalty_name
    }

    metrics = {
        "Model": "Logistic Regression (Tuned)",
        "CV_F1_Macro_Mean": result.best_score,
        "Best_Params": best_params
    }

    return result.best_estimator, metrics


def extract_and_save_coefficients(
    model,
    feature_names,
    output_path="reports/linear_coefficients.csv"
):
    """Xuất hệ số mô hình và độ lớn tuyệt đối để phân tích đặc trưng."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    coefs = model.coef_[0]

    df_coef = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs,
        "Abs_Impact": np.abs(coefs)
    }).sort_values(
        by="Abs_Impact",
        ascending=False
    )

    df_coef.to_csv(output_path, index=False)

    print(
        f"-> Đã xuất bảng trọng số đặc trưng tại: "
        f"{output_path}"
    )

    return df_coef


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print(
        "=== [PERSON 1] BẮT ĐẦU HUẤN LUYỆN "
        "LINEAR MODEL & BASELINE ==="
    )

    # Nạp tập Train đã được cố định từ quy trình tiền xử lý Tuần 2.
    X_train, y_train = load_data()

    print(
        f"Kích thước tập Train: "
        f"{X_train.shape[0]} mẫu, "
        f"{X_train.shape[1]} đặc trưng."
    )

    # Sử dụng cùng cấu hình 5-fold Stratified CV của framework P4.
    cv = make_cv_splitter(
        task_type="classification",
        n_splits=5,
        random_state=42,
        shuffle=True
    )

    print("\n1. Đang đánh giá Dummy Baseline...")

    dummy_metrics = evaluate_dummy_baseline(
        X_train,
        y_train,
        cv
    )

    print("2. Đang đánh giá Logistic Regression (Default)...")

    default_metrics = evaluate_default_linear_model(
        X_train,
        y_train,
        cv
    )

    print("3. Đang tuning Logistic Regression...")

    best_linear_model, tuned_metrics = tune_linear_model(
        X_train,
        y_train,
        cv
    )

    # Lưu estimator tốt nhất đã được refit trên toàn bộ tập Train.
    model_output_path = "models/best_linear_model.pkl"

    joblib.dump(
        best_linear_model,
        model_output_path
    )

    print(
        f"-> Đã lưu mô hình tối ưu vào: "
        f"{model_output_path}"
    )

    # Xuất hệ số của estimator tốt nhất để phục vụ phân tích mô hình.
    extract_and_save_coefficients(
        best_linear_model,
        X_train.columns
    )

    # Tổng hợp các mốc hiệu năng của riêng P1.
    comparison_table = pd.DataFrame([
        {
            "Mô hình": dummy_metrics["Model"],
            "CV F1-Macro": round(
                dummy_metrics["CV_F1_Macro_Mean"],
                4
            ),
            "Ghi chú / Tham số": (
                "strategy='most_frequent'"
            )
        },
        {
            "Mô hình": default_metrics["Model"],
            "CV F1-Macro": round(
                default_metrics["CV_F1_Macro_Mean"],
                4
            ),
            "Ghi chú / Tham số": (
                "LogisticRegression()"
            )
        },
        {
            "Mô hình": tuned_metrics["Model"],
            "CV F1-Macro": round(
                tuned_metrics["CV_F1_Macro_Mean"],
                4
            ),
            "Ghi chú / Tham số": json.dumps(
                tuned_metrics["Best_Params"]
            )
        }
    ])

    comparison_table.to_csv(
        "reports/linear_model_benchmark.csv",
        index=False
    )

    print("\n=== KẾT QUẢ RIÊNG CỦA P1 ===")
    print(
        comparison_table.to_string(index=False)
    )


if __name__ == "__main__":
    main()