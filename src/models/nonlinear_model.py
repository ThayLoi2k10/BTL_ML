

import os
import json
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC

from src.models.linear_model import load_data
from src.tuning import cv_results_dataframe, make_cv_splitter, tune_model


SCORING = "f1_macro"
RANDOM_STATE = 42

# Kernel linear không dùng gamma, nên tách thành hai không gian con
# để tránh lặp lại các tổ hợp tương đương.
PARAM_GRID = [
    {
        "kernel": ["linear"],
        "C": [0.1, 1.0, 10.0],
    },
    {
        "kernel": ["rbf"],
        # C = 300, 1000 kiểm tra xem cực đại có nằm ngoài biên C = 100 không.
        "C": [0.1, 1.0, 10.0, 100.0, 300.0, 1000.0],
        "gamma": ["scale", 0.001, 0.01, 0.1, 1.0],
    },
]


def evaluate_default_svm(X_train, y_train, cv):
    """Đánh giá SVC với cấu hình mặc định (kernel='rbf', C=1, gamma='scale')."""
    model_default = SVC(random_state=RANDOM_STATE)

    scores = cross_validate(
        model_default,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        n_jobs=-1
    )

    metrics = {
        "Model": "SVM - SVC (Default)",
        "CV_F1_Macro_Mean": np.mean(scores["test_score"]),
        "CV_F1_Macro_Std": np.std(scores["test_score"]),
        "Params": {"kernel": "rbf", "C": 1.0, "gamma": "scale"}
    }

    return metrics


def tune_svm(X_train, y_train, cv):
    """Quét siêu tham số SVC qua framework tuning dùng chung (GridSearchCV)."""
    model = SVC(random_state=RANDOM_STATE)

    # refit=False: mô hình cuối được huấn luyện lại thủ công ở bước sau
    # để đo riêng thời gian huấn luyện trên toàn bộ tập Train.
    result = tune_model(
        estimator=model,
        param_grid=PARAM_GRID,
        cv_splitter=cv,
        scoring_metric=SCORING,
        search_type="grid",
        X=X_train,
        y=y_train,
        refit=False
    )

    cv_table = cv_results_dataframe(result)
    best_row = cv_table.iloc[0]

    metrics = {
        "Model": "SVM - SVC (Tuned)",
        "CV_F1_Macro_Mean": result.best_score,
        "CV_F1_Macro_Std": float(best_row["std_test_score"]),
        "Params": result.best_params,
        "Search_Seconds": result.elapsed_seconds,
        "N_Candidates": len(cv_table)
    }

    return result, cv_table, metrics


def train_final_model(best_params, X_train, y_train):
    """Huấn luyện SVC tối ưu trên toàn bộ tập Train và đo thời gian."""
    final_model = clone(SVC(random_state=RANDOM_STATE)).set_params(**best_params)

    started_at = perf_counter()
    final_model.fit(X_train, y_train)
    train_seconds = perf_counter() - started_at

    return final_model, train_seconds


def save_cv_results(cv_table, output_path="reports/nonlinear_cv_results.csv"):
    """Lưu toàn bộ kết quả Grid Search (mỗi tổ hợp siêu tham số một dòng)."""
    table = cv_table.copy()
    table["params"] = table["params"].apply(json.dumps)
    table = table.round(4)
    table.to_csv(output_path, index=False)

    print(f"-> Đã xuất bảng kết quả Grid Search tại: {output_path}")
    return table


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("=== BẮT ĐẦU HUẤN LUYỆN NONLINEAR MODEL (SVM) ===")

    # Dùng chung tập Train đã cố định từ quy trình tiền xử lý Tuần 2.
    X_train, y_train = load_data()

    print(
        f"Kích thước tập Train: "
        f"{X_train.shape[0]} mẫu, "
        f"{X_train.shape[1]} đặc trưng."
    )

    # Cùng cấu hình 5-fold Stratified CV với mô hình tuyến tính.
    cv = make_cv_splitter(
        task_type="classification",
        n_splits=5,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    print("\n1. Đang đánh giá SVC (Default)...")
    default_metrics = evaluate_default_svm(X_train, y_train, cv)

    print("2. Đang tuning SVC bằng GridSearchCV...")
    _, cv_table, tuned_metrics = tune_svm(X_train, y_train, cv)
    print(
        f"   Đã quét {tuned_metrics['N_Candidates']} tổ hợp trong "
        f"{tuned_metrics['Search_Seconds']:.1f} giây."
    )
    print(f"   Bộ siêu tham số tốt nhất: {tuned_metrics['Params']}")

    save_cv_results(cv_table)

    print("3. Đang huấn luyện SVC tối ưu trên toàn bộ tập Train...")
    best_model, train_seconds = train_final_model(
        tuned_metrics["Params"],
        X_train,
        y_train
    )
    print(
        f"   Thời gian huấn luyện: {train_seconds:.3f} giây "
        f"({best_model.n_support_.sum()} support vectors)."
    )

    model_output_path = "models/best_nonlinear_model.pkl"
    joblib.dump(best_model, model_output_path)
    print(f"-> Đã lưu mô hình tối ưu vào: {model_output_path}")

    comparison_table = pd.DataFrame([
        {
            "Mô hình": metrics["Model"],
            "CV F1-Macro (mean)": round(metrics["CV_F1_Macro_Mean"], 4),
            "CV F1-Macro (std)": round(metrics["CV_F1_Macro_Std"], 4),
            "Tham số": json.dumps(metrics["Params"])
        }
        for metrics in (default_metrics, tuned_metrics)
    ])
    comparison_table["Thời gian huấn luyện full Train (s)"] = [
        None,
        round(train_seconds, 3)
    ]

    comparison_table.to_csv(
        "reports/nonlinear_model_benchmark.csv",
        index=False
    )

    print("\n=== KẾT QUẢ MÔ HÌNH PHI TUYẾN ===")
    print(comparison_table.to_string(index=False))


if __name__ == "__main__":
    main()
