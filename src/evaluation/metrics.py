"""Quantitative evaluation and generalization gap comparison for Week 4.

Author: Lợi (Person 1 - Evaluation & Metrics)
Responsibilities:
1. Load preprocessing pipeline and evaluation test data (leakage-free).
2. Load all saved models from Week 3 (Linear, Tree, Nonlinear) + Dummy baseline.
3. Generate predictions (y_pred) and probabilities (y_prob) into a shared DataFrame
   to hand over to Person 2 (plots.py) and Person 3 (error_analysis.py).
4. Compute comprehensive classification (Accuracy, Precision, Recall, Macro-F1,
   ROC-AUC, PR-AUC) and regression metrics (RMSE, MAE, R2, MAPE).
5. Build 2-way comparison table: CV Score (Week 3) <-> Test Score (Week 4)
   and determine Generalization Gap / Overfitting risk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_preprocessing_pipeline(
    pipeline_path: str | Path = MODELS_DIR / "preprocessing_pipeline.pkl",
) -> Any:
    """Load the fitted preprocessing pipeline with unpickling compatibility guards."""
    pipeline_path = Path(pipeline_path)
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Preprocessing pipeline not found at {pipeline_path}")

    # Compatibility shim for scikit-learn version differences
    import sklearn.compose._column_transformer as ct

    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        ct._RemainderColsList = _RemainderColsList

    return joblib.load(pipeline_path)


def load_test_data(
    processed_dir: str | Path = PROCESSED_DATA_DIR,
    raw_csv_path: str | Path | None = None,
    pipeline: Any | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load test features and ground-truth labels.

    If data/processed/X_test.parquet exists, it loads pre-transformed features
    directly (already processed by Week 2 pipeline).
    If a raw CSV is provided, it transforms raw data using the fitted pipeline.
    """
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

    raise FileNotFoundError(
        f"Could not find processed test data in {processed_dir} and no raw CSV provided."
    )


def load_all_models(
    models_dir: str | Path = MODELS_DIR,
    train_x_path: str | Path = PROCESSED_DATA_DIR / "X_train.parquet",
    train_y_path: str | Path = PROCESSED_DATA_DIR / "y_train.parquet",
) -> dict[str, Any]:
    """Load all saved models from Week 3 and instantiate the Dummy baseline.

    Returns
    -------
    dict with keys: 'dummy', 'linear', 'tree', 'nonlinear'
    """
    models_dir = Path(models_dir)
    linear_path = models_dir / "best_linear_model.pkl"
    tree_path = models_dir / "best_tree_model.pkl"
    nonlinear_path = models_dir / "best_nonlinear_model.pkl"

    for p in (linear_path, tree_path, nonlinear_path):
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")

    linear_model = joblib.load(linear_path)
    tree_model = joblib.load(tree_path)
    nonlinear_model = joblib.load(nonlinear_path)

    # Fit DummyClassifier baseline on X_train, y_train
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
    """Generate predictions and positive class probabilities for an estimator.

    For SVC without predict_proba, uses the logistic sigmoid of decision_function:
    P(y=1|x) = 1 / (1 + exp(-decision_function(x))).
    """
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
        # Platt/logistic sigmoid mapping: monotonic, maps R -> (0, 1)
        y_prob = 1.0 / (1.0 + np.exp(-decision))
    else:
        # Fallback to binary predicted values as pseudo-probability
        y_prob = (pd.Series(y_pred) == pos_label).astype(float).to_numpy()

    return np.asarray(y_pred), np.asarray(y_prob)


def generate_predictions_dataframe(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    pos_label: str = "yes",
    output_parquet: str | Path | None = REPORTS_DIR / "test_predictions.parquet",
    output_csv: str | Path | None = REPORTS_DIR / "test_predictions.csv",
) -> pd.DataFrame:
    """Generate and save unified predictions DataFrame for Person 2 and Person 3.

    Columns include:
    - y_true: actual string label ('no', 'yes')
    - y_true_binary: actual binary integer (0, 1)
    - y_pred_<model>: predicted label
    - y_prob_<model>: positive class probability
    """
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
    """Compute all required quantitative classification metrics on the test set."""
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
    """Compute regression metrics (RMSE, MAE, R2, MAPE) for full specification compliance."""
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
    hyperparameters_json: str | Path = MODELS_DIR / "best_hyperparameters.json",
) -> dict[str, float]:
    """Load CV scores recorded during Week 3 model tuning."""
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
    output_csv: str | Path | None = REPORTS_DIR / "evaluation_benchmark.csv",
) -> pd.DataFrame:
    """Build the 2-way comparison table: CV Score (Week 3) <-> Test Score (Week 4).

    Calculates:
    - Generalization Gap = CV_Score - Test_Macro_F1
    - Overfitting Risk classification
    """
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
            risk = "High Overfitting Risk"
        elif gen_gap > 0.02:
            risk = "Moderate Overfitting Risk"
        elif gen_gap < -0.02:
            risk = "Underfitting on Train / Strong Generalization"
        else:
            risk = "Minimal Gap / Well Generalized"

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
    """Execute end-to-end evaluation pipeline and export all artifacts."""
    print("Loading test data...")
    X_test, y_test = load_test_data()

    print("Loading Week 3 models...")
    models = load_all_models()

    print("Generating predictions and probability outputs for Person 2 & Person 3...")
    df_pred = generate_predictions_dataframe(models, X_test, y_test)
    print(f"Predictions saved to {REPORTS_DIR / 'test_predictions.parquet'} and .csv")

    print("Computing 2-way evaluation summary and generalization gaps...")
    summary_df = build_evaluation_summary(df_pred)
    print(f"Summary saved to {REPORTS_DIR / 'evaluation_benchmark.csv'}")

    return df_pred, summary_df


if __name__ == "__main__":
    preds, summary = evaluate_all_models()
    print("\n--- QUANTITATIVE EVALUATION BENCHMARK: CV (WEEK 3) VS TEST (WEEK 4) ---")
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
