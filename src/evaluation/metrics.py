"""Mô-đun đo lường định lượng và so sánh năng lực tổng quát hóa giữa các mô hình."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from src.config import (
    BEST_HYPERPARAMETERS_PATH,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    PREPROCESSING_PIPELINE_PATH,
    REPORTS_DIR,
)

PREDICTIONS_PARQUET_PATH = REPORTS_DIR / "test_predictions.parquet"
PREDICTIONS_CSV_PATH = REPORTS_DIR / "test_predictions.csv"
BENCHMARK_SUMMARY_PATH = REPORTS_DIR / "evaluation_benchmark.csv"


def load_preprocessing_pipeline(
    pipeline_path: str | Path = PREPROCESSING_PIPELINE_PATH,
) -> Any:
    """Nạp đường ống tiền xử lý đã huấn luyện kèm bộ đệm tương thích unpickle."""
    pipeline_path = Path(pipeline_path)
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Không tìm thấy pipeline tiền xử lý tại: {pipeline_path}")

    import sklearn.compose._column_transformer as ct

    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        ct._RemainderColsList = _RemainderColsList

    return joblib.load(pipeline_path)


def load_test_data(
    processed_dir: str | Path = DATA_PROCESSED_DIR,
    raw_csv_path: str | Path | None = None,
    pipeline: Any | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Nạp ma trận đặc trưng và nhãn thực tế của tập kiểm thử độc lập."""
    processed_dir = Path(processed_dir)
    test_x_path = processed_dir / "X_test.parquet"
    test_y_path = processed_dir / "y_test.parquet"

    if test_x_path.exists() and test_y_path.exists():
        X_test = pd.read_parquet(test_x_path)
        y_test_df = pd.read_parquet(test_y_path)
        y_test = y_test_df.iloc[:, 0] if isinstance(y_test_df, pd.DataFrame) else y_test_df
        return X_test, y_test

    if raw_csv_path is not None and pipeline is not None:
        raw_df = pd.read_csv(raw_csv_path)
        target_col = "deposit"
        y_test = raw_df[target_col]
        X_raw = raw_df.drop(columns=[target_col])
        X_test = pipeline.transform(X_raw)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        return X_test, y_test

    raise FileNotFoundError(f"Không tìm thấy tập dữ liệu kiểm thử tại: {processed_dir}")


def load_all_models(
    models_dir: str | Path = MODELS_DIR,
    train_x_path: str | Path = DATA_PROCESSED_DIR / "X_train.parquet",
    train_y_path: str | Path = DATA_PROCESSED_DIR / "y_train.parquet",
) -> dict[str, Any]:
    """Nạp tất cả các mô hình đã lưu cùng mô hình cơ sở Dummy Classifier."""
    models_dir = Path(models_dir)
    linear_path = models_dir / "best_linear_model.pkl"
    tree_path = models_dir / "best_tree_model.pkl"
    nonlinear_path = models_dir / "best_nonlinear_model.pkl"

    for p in (linear_path, tree_path, nonlinear_path):
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy file mô hình: {p}")

    linear_model = joblib.load(linear_path)
    tree_model = joblib.load(tree_path)
    nonlinear_model = joblib.load(nonlinear_path)

    dummy_model = DummyClassifier(strategy="most_frequent")
    if Path(train_x_path).exists() and Path(train_y_path).exists():
        X_train = pd.read_parquet(train_x_path)
        y_train = pd.read_parquet(train_y_path).iloc[:, 0]
        dummy_model.fit(X_train, y_train)

    return {
        "dummy": dummy_model,
        "linear": linear_model,
        "tree": tree_model,
        "nonlinear": nonlinear_model,
    }


def predict_model_outputs(
    model: Any,
    X_test: pd.DataFrame,
    pos_label: str = "yes",
) -> tuple[np.ndarray, np.ndarray]:
    """Thực thi dự đoán nhãn lớp và xác suất tương ứng cho một mô hình ước lượng."""
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            if pos_label in classes:
                pos_idx = classes.index(pos_label)
                y_prob = probs[:, pos_idx]
            else:
                y_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        else:
            y_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_test)
        y_prob = 1.0 / (1.0 + np.exp(-decision))
    else:
        y_prob = (pd.Series(y_pred) == pos_label).astype(float).to_numpy()

    return np.asarray(y_pred), np.asarray(y_prob)


def generate_predictions_dataframe(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    pos_label: str = "yes",
    output_parquet: str | Path | None = PREDICTIONS_PARQUET_PATH,
    output_csv: str | Path | None = PREDICTIONS_CSV_PATH,
) -> pd.DataFrame:
    """Tạo và lưu trữ DataFrame hợp nhất chứa nhãn thực tế, dự đoán và xác suất của tất cả mô hình."""
    y_test_clean = y_test.reset_index(drop=True)
    df_pred = pd.DataFrame({
        "y_true": y_test_clean,
        "y_true_binary": (y_test_clean == pos_label).astype(int),
    })

    for model_key, model in models.items():
        preds, probs = predict_model_outputs(model, X_test, pos_label=pos_label)
        df_pred[f"y_pred_{model_key}"] = preds
        df_pred[f"y_prob_{model_key}"] = probs

    if output_parquet is not None:
        Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
        df_pred.to_parquet(output_parquet, index=False)

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df_pred.to_csv(output_csv, index=False)

    return df_pred


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray | None = None,
    pos_label: str = "yes",
) -> dict[str, float]:
    """Tính toán toàn bộ các chỉ số định lượng phân loại trên tập kiểm thử."""
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    y_pred_series = pd.Series(y_pred).reset_index(drop=True)

    y_true_binary = (y_true_series == pos_label).astype(int)
    y_pred_binary = (y_pred_series == pos_label).astype(int)

    acc = float(accuracy_score(y_true_series, y_pred_series))
    prec = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
    rec = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
    macro_f1 = float(f1_score(y_true_series, y_pred_series, average="macro"))
    f1_pos = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))

    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true_binary, y_prob))
        except Exception:
            roc_auc = 0.5
        try:
            pr_auc = float(average_precision_score(y_true_binary, y_prob))
        except Exception:
            pr_auc = 0.0
    else:
        roc_auc = 0.5
        pr_auc = 0.0

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Macro_F1": macro_f1,
        "F1_Positive": f1_pos,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
    }


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Tính toán các chỉ số hồi quy (RMSE, MAE, R2, MAPE) phục vụ bài toán liên tục."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    rmse = float(root_mean_squared_error(y_true_arr, y_pred_arr))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    mape = float(mean_absolute_percentage_error(y_true_arr, y_pred_arr))

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape,
    }


def load_week3_cv_scores(
    hyperparameters_json: str | Path = BEST_HYPERPARAMETERS_PATH,
) -> dict[str, float]:
    """Nạp điểm kiểm định chéo F1-Macro đã ghi nhận trong quá trình tuning ở Tuần 3."""
    json_path = Path(hyperparameters_json)
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "dummy": 0.3448,
            "linear": float(data.get("linear", {}).get("best_cv_score", 0.6960)),
            "tree": float(data.get("tree", {}).get("best_cv_score", 0.7233)),
            "nonlinear": float(data.get("nonlinear", {}).get("best_cv_score", 0.7178)),
        }
    return {
        "dummy": 0.3448,
        "linear": 0.6960,
        "tree": 0.7233,
        "nonlinear": 0.7178,
    }


def build_evaluation_summary(
    df_predictions: pd.DataFrame,
    cv_scores: dict[str, float] | None = None,
    output_csv: str | Path | None = BENCHMARK_SUMMARY_PATH,
) -> pd.DataFrame:
    """Lập bảng đối chiếu hai chiều giữa điểm CV Tuần 3 và điểm Test Tuần 4 để tính Generalization Gap."""
    if cv_scores is None:
        cv_scores = load_week3_cv_scores()

    model_display_names = {
        "dummy": "Dummy Baseline (Most Frequent)",
        "linear": "Logistic Regression (Tuned)",
        "tree": "Random Forest (Tuned)",
        "nonlinear": "SVC RBF (Tuned)",
    }

    records = []
    y_true = df_predictions["y_true"]

    for key, display_name in model_display_names.items():
        pred_col = f"y_pred_{key}"
        prob_col = f"y_prob_{key}"

        if pred_col not in df_predictions.columns:
            continue

        y_pred = df_predictions[pred_col]
        y_prob = df_predictions[prob_col] if prob_col in df_predictions.columns else None

        metrics = compute_classification_metrics(y_true, y_pred, y_prob)

        cv_score = cv_scores.get(key, np.nan)
        test_score = metrics["Macro_F1"]
        gen_gap = cv_score - test_score
        abs_gap = abs(gen_gap)

        if np.isnan(cv_score):
            risk = "N/A"
        elif gen_gap > 0.05:
            risk = "Nguy cơ Overfitting cao"
        elif gen_gap > 0.02:
            risk = "Nguy cơ Overfitting vừa phải"
        elif gen_gap < -0.02:
            risk = "Khái quát hóa mạnh / Điểm Test cao hơn Train"
        else:
            risk = "Chênh lệch tối thiểu / Khái quát hóa xuất sắc"

        records.append({
            "Model_Key": key,
            "Model_Name": display_name,
            "CV_Macro_F1 (W3)": cv_score,
            "Test_Macro_F1 (W4)": test_score,
            "Generalization_Gap": gen_gap,
            "Abs_Gap": abs_gap,
            "Test_Accuracy": metrics["Accuracy"],
            "Test_Precision": metrics["Precision"],
            "Test_Recall": metrics["Recall"],
            "Test_F1_Positive": metrics["F1_Positive"],
            "Test_ROC_AUC": metrics["ROC_AUC"],
            "Test_PR_AUC": metrics["PR_AUC"],
            "Overfitting_Risk": risk,
        })

    summary_df = pd.DataFrame(records)
    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_csv, index=False)

    return summary_df


def evaluate_all_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thực thi toàn bộ quy trình đánh giá định lượng đầu cuối và xuất các bảng kết quả."""
    print("Nạp dữ liệu kiểm thử...")
    X_test, y_test = load_test_data()

    print("Nạp các mô hình đã huấn luyện...")
    models = load_all_models()

    print("Sinh dự đoán và phân bố xác suất cho tất cả các mô hình...")
    df_pred = generate_predictions_dataframe(models, X_test, y_test)
    print(f"Đã lưu kết quả dự đoán tại: {PREDICTIONS_CSV_PATH}")

    print("Xây dựng bảng đối chiếu hai chiều CV vs Test...")
    summary_df = build_evaluation_summary(df_pred)
    print(f"Đã lưu bảng tổng hợp đánh giá tại: {BENCHMARK_SUMMARY_PATH}")

    return df_pred, summary_df


def main() -> None:
    """Hàm thực thi chính khi chạy module từ dòng lệnh."""
    preds, summary = evaluate_all_models()
    print("\n--- BẢNG ĐỐI CHIẾU ĐỊNH LƯỢNG CV (TUẦN 3) VS TEST (TUẦN 4) ---")
    print(summary[[
        "Model_Name",
        "CV_Macro_F1 (W3)",
        "Test_Macro_F1 (W4)",
        "Generalization_Gap",
        "Test_Accuracy",
        "Test_ROC_AUC",
        "Test_PR_AUC",
        "Overfitting_Risk",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
