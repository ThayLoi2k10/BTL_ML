"""Command-line inference script for predicting bank deposit subscription.

Author: Lợi (Person 1 - Deploy & Inference)
Usage:
    python src/predict.py --input data/sample_test.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# Safe relative paths resolution based on project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PIPELINE_PATH = PROJECT_ROOT / "models" / "preprocessing_pipeline.pkl"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "champion_model.pkl"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "sample_test.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "predictions.csv"


def ensure_unpickle_compatibility() -> None:
    """Ensure compatibility across scikit-learn versions during unpickling."""
    import sklearn.compose._column_transformer as ct

    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        ct._RemainderColsList = _RemainderColsList


def load_artifacts(
    pipeline_path: Path | str = DEFAULT_PIPELINE_PATH,
    model_path: Path | str = DEFAULT_MODEL_PATH,
) -> tuple[Any, Any]:
    """Load fitted preprocessing pipeline and champion predictive model."""
    ensure_unpickle_compatibility()
    pipeline_path = Path(pipeline_path)
    model_path = Path(model_path)

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Preprocessing pipeline not found at '{pipeline_path}'."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Champion model not found at '{model_path}'."
        )

    pipeline = joblib.load(pipeline_path)
    model = joblib.load(model_path)
    return pipeline, model


def predict(
    input_data: pd.DataFrame,
    pipeline: Any,
    model: Any,
    pos_label: str = "yes",
) -> pd.DataFrame:
    """Transform input features and generate predictions with confidence probabilities."""
    df_raw = input_data.copy()

    # If ground-truth 'deposit' exists in the input, separate it to avoid pipeline guard error
    actual_labels = None
    if "deposit" in df_raw.columns:
        actual_labels = df_raw["deposit"].copy()
        features_to_transform = df_raw.drop(columns=["deposit"])
    else:
        features_to_transform = df_raw

    # Transform raw data through Week 2 preprocessing pipeline
    X_transformed = pipeline.transform(features_to_transform)

    # Predict class labels
    predictions = model.predict(X_transformed)

    # Compute prediction probabilities and confidence scores
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_transformed)
        classes = list(getattr(model, "classes_", ["no", "yes"]))
        pos_idx = classes.index(pos_label) if pos_label in classes else 1
        neg_idx = 1 - pos_idx

        prob_pos = probabilities[:, pos_idx]
        prob_neg = probabilities[:, neg_idx]
        confidence = np.max(probabilities, axis=1)
    elif hasattr(model, "decision_function"):
        decisions = model.decision_function(X_transformed)
        # Platt sigmoid mapping for decision values
        prob_pos = 1.0 / (1.0 + np.exp(-decisions))
        prob_neg = 1.0 - prob_pos
        confidence = np.maximum(prob_pos, prob_neg)
    else:
        prob_pos = (pd.Series(predictions) == pos_label).astype(float).to_numpy()
        prob_neg = 1.0 - prob_pos
        confidence = np.ones(len(predictions), dtype=float)

    result_df = df_raw.copy()
    result_df["predicted_deposit"] = predictions
    result_df["confidence_score"] = np.round(confidence, 4)
    result_df["probability_yes"] = np.round(prob_pos, 4)
    result_df["probability_no"] = np.round(prob_neg, 4)

    if actual_labels is not None:
        result_df["actual_deposit"] = actual_labels
        result_df["is_correct"] = result_df["predicted_deposit"] == actual_labels

    return result_df


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch inference using Week 4 Champion Model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the input CSV file containing raw customer data.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path where output predictions CSV will be saved.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default=str(DEFAULT_PIPELINE_PATH),
        help="Path to fitted preprocessing pipeline artifact.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to trained champion model artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    input_path = Path(args.input)
    output_path = Path(args.output)
    pipeline_path = Path(args.pipeline)
    model_path = Path(args.model)

    print("=" * 70)
    print(" BANK MARKETING TERM DEPOSIT - BATCH INFERENCE SYSTEM")
    print("=" * 70)
    print(f"[*] Input Data       : {input_path}")
    print(f"[*] Output File      : {output_path}")
    print(f"[*] Pipeline Artifact: {pipeline_path}")
    print(f"[*] Champion Model   : {model_path}")
    print("-" * 70)

    if not input_path.exists():
        print(f"[!] Error: Input file '{input_path}' does not exist.")
        sys.exit(1)

    print("[*] Loading input data...")
    input_df = pd.read_csv(input_path)
    print(f"    Loaded {len(input_df)} records across {len(input_df.columns)} columns.")

    print("[*] Loading ML artifacts...")
    pipeline, model = load_artifacts(pipeline_path, model_path)
    print(f"    Loaded Pipeline: {pipeline.__class__.__name__}")
    print(f"    Loaded Model   : {model.__class__.__name__}")

    print("[*] Running inference pipeline...")
    results_df = predict(input_df, pipeline, model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"[+] Predictions saved successfully to: {output_path}")
    print("-" * 70)

    # Display preview in terminal
    preview_cols = [
        col
        for col in [
            "age",
            "job",
            "marital",
            "balance",
            "predicted_deposit",
            "confidence_score",
            "probability_yes",
            "actual_deposit",
            "is_correct",
        ]
        if col in results_df.columns
    ]

    print("[*] INFERENCE RESULTS PREVIEW:")
    print(results_df[preview_cols].to_string(index=True))
    print("-" * 70)

    counts = results_df["predicted_deposit"].value_counts().to_dict()
    print(f"[*] Summary Statistics: {counts}")
    mean_conf = results_df["confidence_score"].mean()
    print(f"[*] Average Confidence: {mean_conf:.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
