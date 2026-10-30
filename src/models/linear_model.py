"""Mô-đun huấn luyện, đánh giá mô hình cơ sở và tối ưu hóa hồi quy Logistic Regression."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from src.config import (
    CV_SPLITS,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SCORING_METRIC,
)
from src.tuning import make_cv_splitter, tune_model

DEFAULT_TRAIN_X_PATH = DATA_PROCESSED_DIR / "X_train.parquet"
DEFAULT_TRAIN_Y_PATH = DATA_PROCESSED_DIR / "y_train.parquet"
DEFAULT_OUTPUT_MODEL_PATH = MODELS_DIR / "best_linear_model.pkl"
DEFAULT_COEF_OUTPUT_PATH = REPORTS_DIR / "linear_coefficients.csv"
DEFAULT_BENCHMARK_OUTPUT_PATH = REPORTS_DIR / "linear_model_benchmark.csv"


def load_data(
    train_x_path: str | Path = DEFAULT_TRAIN_X_PATH,
    train_y_path: str | Path = DEFAULT_TRAIN_Y_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Nạp dữ liệu đặc trưng và nhãn huấn luyện đã tiền xử lý từ tệp Parquet."""
    train_x_path = Path(train_x_path)
    train_y_path = Path(train_y_path)
    if not train_x_path.exists() or not train_y_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file huấn luyện tại {train_x_path} hoặc {train_y_path}."
        )

    X_train = pd.read_parquet(train_x_path)
    y_train = pd.read_parquet(train_y_path)
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]

    return X_train, y_train


def evaluate_dummy_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: Any,
) -> dict[str, Any]:
    """Đánh giá mô hình cơ sở Dummy Classifier bằng chiến lược dự đoán nhãn đa số."""
    dummy = DummyClassifier(strategy="most_frequent")
    scores = cross_validate(
        dummy,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING_METRIC,
        n_jobs=-1,
    )
    return {
        "Model": "Dummy Baseline (Most Frequent)",
        "CV_F1_Macro_Mean": float(np.mean(scores["test_score"])),
    }


def evaluate_default_linear_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: Any,
) -> dict[str, Any]:
    """Đánh giá mô hình hồi quy Logistic Regression với cấu hình tham số mặc định."""
    model_default = LogisticRegression()
    scores = cross_validate(
        model_default,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING_METRIC,
        n_jobs=-1,
    )
    return {
        "Model": "Logistic Regression (Default)",
        "CV_F1_Macro_Mean": float(np.mean(scores["test_score"])),
    }


def tune_linear_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: Any,
) -> tuple[LogisticRegression, dict[str, Any]]:
    """Tối ưu hóa siêu tham số regularization C và penalty cho Logistic Regression qua Grid Search."""
    model = LogisticRegression(solver="saga", max_iter=5000, random_state=RANDOM_STATE)
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "l1_ratio": [0, 1],
    }

    result = tune_model(
        estimator=model,
        param_grid=param_grid,
        cv_splitter=cv,
        scoring_metric=SCORING_METRIC,
        X=X_train,
        y=y_train,
    )

    best_l1_ratio = result.best_params["l1_ratio"]
    penalty_name = "l1" if best_l1_ratio == 1 else "l2"
    best_params = {
        "C": result.best_params["C"],
        "penalty": penalty_name,
    }

    metrics = {
        "Model": "Logistic Regression (Tuned)",
        "CV_F1_Macro_Mean": float(result.best_score),
        "Best_Params": best_params,
    }
    return result.best_estimator, metrics


def extract_and_save_coefficients(
    model: LogisticRegression,
    feature_names: Sequence[str],
    output_path: str | Path = DEFAULT_COEF_OUTPUT_PATH,
) -> pd.DataFrame:
    """Trích xuất trọng số các đặc trưng của mô hình tuyến tính và lưu ra tệp CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coefs = model.coef_[0]
    df_coef = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs,
        "Abs_Impact": np.abs(coefs),
    }).sort_values(by="Abs_Impact", ascending=False)

    df_coef.to_csv(output_path, index=False)
    print(f"-> Đã xuất bảng trọng số đặc trưng tại: {output_path}")
    return df_coef


def main() -> None:
    """Thực thi toàn bộ quy trình huấn luyện, tối ưu và lưu mô hình tuyến tính."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== BẮT ĐẦU HUẤN LUYỆN LINEAR MODEL & BASELINE ===")
    X_train, y_train = load_data()
    print(f"Kích thước tập Train: {X_train.shape[0]} mẫu, {X_train.shape[1]} đặc trưng.")

    cv = make_cv_splitter(
        task_type="classification",
        n_splits=CV_SPLITS,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    print("\n1. Đang đánh giá Dummy Baseline...")
    dummy_metrics = evaluate_dummy_baseline(X_train, y_train, cv)

    print("2. Đang đánh giá Logistic Regression (Default)...")
    default_metrics = evaluate_default_linear_model(X_train, y_train, cv)

    print("3. Đang tuning Logistic Regression...")
    best_linear_model, tuned_metrics = tune_linear_model(X_train, y_train, cv)

    joblib.dump(best_linear_model, DEFAULT_OUTPUT_MODEL_PATH)
    print(f"-> Đã lưu mô hình tối ưu vào: {DEFAULT_OUTPUT_MODEL_PATH}")

    extract_and_save_coefficients(best_linear_model, X_train.columns)

    comparison_table = pd.DataFrame([
        {
            "Mô hình": dummy_metrics["Model"],
            "CV F1-Macro": round(dummy_metrics["CV_F1_Macro_Mean"], 4),
            "Ghi chú / Tham số": "strategy='most_frequent'",
        },
        {
            "Mô hình": default_metrics["Model"],
            "CV F1-Macro": round(default_metrics["CV_F1_Macro_Mean"], 4),
            "Ghi chú / Tham số": "LogisticRegression()",
        },
        {
            "Mô hình": tuned_metrics["Model"],
            "CV F1-Macro": round(tuned_metrics["CV_F1_Macro_Mean"], 4),
            "Ghi chú / Tham số": json.dumps(tuned_metrics["Best_Params"]),
        },
    ])

    comparison_table.to_csv(DEFAULT_BENCHMARK_OUTPUT_PATH, index=False)
    print("\n=== KẾT QUẢ HUẤN LUYỆN LINEAR MODEL ===")
    print(comparison_table.to_string(index=False))


if __name__ == "__main__":
    main()
