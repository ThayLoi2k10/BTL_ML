"""Mô-đun huấn luyện, đánh giá và tối ưu hóa siêu tham số mô hình phi tuyến Support Vector Machine (SVC)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC

from src.config import (
    CV_SPLITS,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SCORING_METRIC,
)
from src.models.linear_model import load_data
from src.tuning import cv_results_dataframe, make_cv_splitter, tune_model

DEFAULT_OUTPUT_MODEL_PATH = MODELS_DIR / "best_nonlinear_model.pkl"
DEFAULT_CV_RESULTS_PATH = REPORTS_DIR / "nonlinear_cv_results.csv"
DEFAULT_BENCHMARK_PATH = REPORTS_DIR / "nonlinear_model_benchmark.csv"

PARAM_GRID = [
    {
        "kernel": ["linear"],
        "C": [0.1, 1.0, 10.0],
    },
    {
        "kernel": ["rbf"],
        "C": [0.1, 1.0, 10.0, 100.0, 300.0, 1000.0],
        "gamma": ["scale", 0.001, 0.01, 0.1, 1.0],
    },
]


def evaluate_default_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: Any,
) -> dict[str, Any]:
    """Đánh giá mô hình SVC với cấu hình tham số mặc định RBF."""
    model_default = SVC(random_state=RANDOM_STATE)
    scores = cross_validate(
        model_default,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING_METRIC,
        n_jobs=-1,
    )
    return {
        "Model": "SVM - SVC (Default)",
        "CV_F1_Macro_Mean": float(np.mean(scores["test_score"])),
        "CV_F1_Macro_Std": float(np.std(scores["test_score"])),
        "Params": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
    }


def tune_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: Any,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """Quét tìm kiếm siêu tham số tối ưu kernel, C và gamma cho mô hình SVC."""
    model = SVC(random_state=RANDOM_STATE)
    result = tune_model(
        estimator=model,
        param_grid=PARAM_GRID,
        cv_splitter=cv,
        scoring_metric=SCORING_METRIC,
        search_type="grid",
        X=X_train,
        y=y_train,
        refit=False,
    )

    cv_table = cv_results_dataframe(result)
    best_row = cv_table.iloc[0]

    metrics = {
        "Model": "SVM - SVC (Tuned)",
        "CV_F1_Macro_Mean": float(result.best_score),
        "CV_F1_Macro_Std": float(best_row["std_test_score"]),
        "Params": result.best_params,
        "Search_Seconds": float(result.elapsed_seconds),
        "N_Candidates": len(cv_table),
    }
    return result, cv_table, metrics


def train_final_model(
    best_params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[SVC, float]:
    """Huấn luyện lại mô hình SVC tối ưu trên toàn bộ tập Train và đo lường thời gian."""
    final_model = clone(SVC(random_state=RANDOM_STATE)).set_params(**best_params)

    started_at = perf_counter()
    final_model.fit(X_train, y_train)
    train_seconds = perf_counter() - started_at

    return final_model, float(train_seconds)


def save_cv_results(
    cv_table: pd.DataFrame,
    output_path: str | Path = DEFAULT_CV_RESULTS_PATH,
) -> pd.DataFrame:
    """Lưu toàn bộ bảng kết quả kiểm định chéo của các tổ hợp siêu tham số ra tệp CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table = cv_table.copy()
    table["params"] = table["params"].apply(json.dumps)
    table = table.round(4)
    table.to_csv(output_path, index=False)
    print(f"-> Đã xuất bảng kết quả Grid Search tại: {output_path}")
    return table


def main() -> None:
    """Thực thi toàn bộ quy trình huấn luyện, tối ưu và lưu mô hình phi tuyến SVM."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== BẮT ĐẦU HUẤN LUYỆN NONLINEAR MODEL (SVM) ===")
    X_train, y_train = load_data()
    print(f"Kích thước tập Train: {X_train.shape[0]} mẫu, {X_train.shape[1]} đặc trưng.")

    cv = make_cv_splitter(
        task_type="classification",
        n_splits=CV_SPLITS,
        random_state=RANDOM_STATE,
        shuffle=True,
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
        y_train,
    )
    print(
        f"   Thời gian huấn luyện: {train_seconds:.3f} giây "
        f"({best_model.n_support_.sum()} support vectors)."
    )

    joblib.dump(best_model, DEFAULT_OUTPUT_MODEL_PATH)
    print(f"-> Đã lưu mô hình tối ưu vào: {DEFAULT_OUTPUT_MODEL_PATH}")

    comparison_table = pd.DataFrame([
        {
            "Mô hình": metrics["Model"],
            "CV F1-Macro (mean)": round(metrics["CV_F1_Macro_Mean"], 4),
            "CV F1-Macro (std)": round(metrics["CV_F1_Macro_Std"], 4),
            "Tham số": json.dumps(metrics["Params"]),
        }
        for metrics in (default_metrics, tuned_metrics)
    ])
    comparison_table["Thời gian huấn luyện full Train (s)"] = [
        None,
        round(train_seconds, 3),
    ]

    comparison_table.to_csv(DEFAULT_BENCHMARK_PATH, index=False)
    print("\n=== KẾT QUẢ MÔ HÌNH PHI TUYẾN ===")
    print(comparison_table.to_string(index=False))


if __name__ == "__main__":
    main()
