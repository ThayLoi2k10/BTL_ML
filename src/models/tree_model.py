"""Mô-đun huấn luyện, đánh giá và tối ưu hóa mô hình cây quyết định Random Forest Ensemble."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
DEFAULT_OUTPUT_MODEL_PATH = MODELS_DIR / "best_tree_model.pkl"
DEFAULT_IMPORTANCE_OUTPUT_PATH = REPORTS_DIR / "tree_feature_importances.csv"
DEFAULT_BENCHMARK_OUTPUT_PATH = REPORTS_DIR / "tree_model_benchmark.csv"
DEFAULT_REPORT_PATH = REPORTS_DIR / "week3_tree_model_ensemble.md"


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


def evaluate_default_tree_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: Any,
) -> dict[str, Any]:
    """Đánh giá mô hình Random Forest với cấu hình siêu tham số mặc định."""
    model_default = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    scores = cross_validate(
        model_default,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING_METRIC,
        n_jobs=-1,
    )
    return {
        "Model": "Random Forest (Default)",
        "CV_F1_Macro_Mean": float(np.mean(scores["test_score"])),
    }


def tune_tree_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: Any,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """Tối ưu hóa các siêu tham số n_estimators, max_depth và min_samples_split cho Random Forest."""
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    }

    result = tune_model(
        estimator=model,
        param_grid=param_grid,
        cv_splitter=cv,
        scoring_metric=SCORING_METRIC,
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
    model: RandomForestClassifier,
    feature_names: Sequence[str],
    output_path: str | Path = DEFAULT_IMPORTANCE_OUTPUT_PATH,
) -> pd.DataFrame:
    """Trích xuất độ quan trọng đặc trưng của Random Forest và lưu ra tệp CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    }).sort_values(by="Importance", ascending=False)

    df_importances.to_csv(output_path, index=False)
    print(f"-> Đã xuất bảng độ quan trọng đặc trưng tại: {output_path}")
    return df_importances


def build_comparison_table(
    default_metrics: dict[str, Any],
    tuned_metrics: dict[str, Any],
) -> pd.DataFrame:
    """Tổng hợp bảng so sánh điểm F1-Macro kiểm định chéo trước và sau khi tối ưu."""
    return pd.DataFrame([
        {
            "Mô hình": default_metrics["Model"],
            "CV F1-Macro": round(default_metrics["CV_F1_Macro_Mean"], 4),
            "Ghi chú / Tham số": f"RandomForestClassifier(random_state={RANDOM_STATE})",
        },
        {
            "Mô hình": tuned_metrics["Model"],
            "CV F1-Macro": round(tuned_metrics["CV_F1_Macro_Mean"], 4),
            "Ghi chú / Tham số": json.dumps(tuned_metrics["Best_Params"]),
        },
    ])


def _comparison_table_to_markdown(comparison_table: pd.DataFrame) -> str:
    """Chuyển đổi bảng kết quả so sánh sang định dạng chuỗi bảng Markdown."""
    lines = [
        "| Mô hình | CV F1-Macro | Ghi chú / Tham số |",
        "|---|---:|---|",
    ]
    for row in comparison_table.to_dict(orient="records"):
        lines.append(
            f"| {row['Mô hình']} | {row['CV F1-Macro']:.4f} | `{row['Ghi chú / Tham số']}` |"
        )
    return "\n".join(lines)


def write_tree_report(
    comparison_table: pd.DataFrame,
    tuned_metrics: dict[str, Any],
    output_path: str | Path = DEFAULT_REPORT_PATH,
) -> None:
    """Ghi tóm tắt kết quả huấn luyện mô hình cây và ensemble ra tệp báo cáo Markdown."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_params = tuned_metrics["Best_Params"]
    markdown_table = _comparison_table_to_markdown(comparison_table)

    content = f"""# Mô hình cây quyết định và Ensemble (Random Forest)

## Kết quả Cross-Validation trước và sau Tuning

{markdown_table}

- Cấu hình tốt nhất: `{json.dumps(best_params)}`.
- Mô hình được lưu tại: `{DEFAULT_OUTPUT_MODEL_PATH}`.
"""
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(content)


def main() -> None:
    """Thực thi toàn bộ quy trình huấn luyện, tối ưu và lưu mô hình Random Forest."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== BẮT ĐẦU HUẤN LUYỆN TREE MODEL & ENSEMBLE ===")
    X_train, y_train = load_data()
    print(f"Kích thước tập Train: {X_train.shape[0]} mẫu, {X_train.shape[1]} đặc trưng.")

    cv = make_cv_splitter(
        task_type="classification",
        n_splits=CV_SPLITS,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    print("\n1. Đang đánh giá Random Forest (Default)...")
    default_metrics = evaluate_default_tree_model(X_train, y_train, cv)

    print("2. Đang tuning Random Forest...")
    best_tree_model, tuned_metrics = tune_tree_model(X_train, y_train, cv)

    joblib.dump(best_tree_model, DEFAULT_OUTPUT_MODEL_PATH)
    print(f"-> Đã lưu mô hình tối ưu vào: {DEFAULT_OUTPUT_MODEL_PATH}")

    extract_and_save_feature_importances(best_tree_model, X_train.columns)

    comparison_table = build_comparison_table(default_metrics, tuned_metrics)
    comparison_table.to_csv(DEFAULT_BENCHMARK_OUTPUT_PATH, index=False)
    write_tree_report(comparison_table, tuned_metrics)

    print("\n=== KẾT QUẢ MÔ HÌNH CÂY & ENSEMBLE ===")
    print(comparison_table.to_string(index=False))


if __name__ == "__main__":
    main()
