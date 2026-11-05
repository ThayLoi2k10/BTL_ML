"""Read-only scientific acceptance checks; emits a separate JSON evidence file.

Run from the extracted project: python scripts/verify_submission.py --output evidence.json
No model fitting, artifact rewriting, or external dataset downloads occur here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from src.encoding_scaling import split_data
from src.pipeline import get_fitted_pipeline_summary, load_correct_dataset
from src.predict import load_artifacts, predict


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    protected = [*sorted((ROOT / "models").glob("*.pkl")),
                 *sorted((ROOT / "data/processed").glob("*.parquet"))]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in protected}
    raw = load_correct_dataset()
    Xtr, Xte, ytr, yte = split_data(raw)
    pipeline, champion = load_artifacts()
    assert set(Xtr.index).isdisjoint(Xte.index)
    np.testing.assert_array_equal(pipeline.named_steps["schema_guard"].fit_indices_, Xtr.index)
    for name, X, y in [("train", Xtr, ytr), ("test", Xte, yte)]:
        transformed = pipeline.transform(X).reset_index(drop=True)
        saved = pd.read_parquet(ROOT / f"data/processed/X_{name}.parquet")
        pd.testing.assert_frame_equal(transformed, saved, check_dtype=False)
        stored_y = pd.read_parquet(ROOT / f"data/processed/y_{name}.parquet").iloc[:, 0]
        np.testing.assert_array_equal(y.to_numpy(), stored_y.to_numpy())
        assert np.isfinite(saved.to_numpy(dtype=float)).all()
        assert not any("duration" in c or "deposit" in c for c in saved.columns)
    X = pd.read_parquet(ROOT / "data/processed/X_test.parquet")
    metric_table = pd.read_csv(ROOT / "reports/evaluation_benchmark.csv").set_index("Model_Key")
    metrics = {}
    for family in ("linear", "tree", "nonlinear"):
        model = joblib.load(ROOT / f"models/best_{family}_model.pkl")
        assert list(model.feature_names_in_) == list(X.columns)
        pred = model.predict(X)
        score = (model.predict_proba(X)[:, list(model.classes_).index("yes")]
                 if hasattr(model, "predict_proba") else model.decision_function(X))
        row = {
            "macro_f1": f1_score(yte, pred, average="macro"),
            "accuracy": accuracy_score(yte, pred),
            "precision_yes": precision_score(yte, pred, pos_label="yes"),
            "recall_yes": recall_score(yte, pred, pos_label="yes"),
            "roc_auc": roc_auc_score(yte == "yes", score),
            "average_precision": average_precision_score(yte == "yes", score),
        }
        for key, col in [("macro_f1", "Test_Macro_F1 (W4)"),
                         ("accuracy", "Test_Accuracy"), ("precision_yes", "Test_Precision"),
                         ("recall_yes", "Test_Recall"), ("roc_auc", "Test_ROC_AUC"),
                         ("average_precision", "Test_PR_AUC")]:
            np.testing.assert_allclose(row[key], metric_table.loc[family, col], atol=1e-12)
        metrics[family] = row
    raw_prediction = predict(Xte, pipeline, champion)
    np.testing.assert_array_equal(raw_prediction.predicted_deposit, champion.predict(X))
    assert digest(ROOT / "models/champion_model.pkl") == digest(ROOT / "models/best_tree_model.pkl")
    assert hashes == {p.relative_to(ROOT).as_posix(): digest(p) for p in protected}
    missing = raw.isna().sum().to_dict()
    return {
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "pandas": pd.__version__, "numpy": np.__version__,
        "raw_sha256": digest(ROOT / "data/raw/bank.csv"),
        "raw_shape": list(raw.shape), "class_counts": raw.deposit.value_counts().to_dict(),
        "duplicates": int(raw.duplicated().sum()), "missing_values": missing,
        "unknown_counts": {c: int(raw[c].eq("unknown").sum()) for c in raw.select_dtypes(exclude="number")},
        "train_shape": [len(Xtr), X.shape[1]], "test_shape": list(X.shape),
        "pipeline_summary": get_fitted_pipeline_summary(pipeline),
        "heldout_transform_equals_saved": True, "raw_cli_prediction_equals_saved": True,
        "metrics": metrics, "protected_sha256": hashes,
        "scientific_limitations": [
            "Historical CV runs on preprocessing and MI selection fitted on all training rows; not fold-local.",
            "CV/test gap is descriptive, not proof against overfitting or deployment drift.",
            "SVC decision_function is a margin, not a calibrated probability.",
            "Champion is provisional; held-out set has been used for model comparison and error analysis.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("PASS: raw/saved schema, row alignment, all metrics, artifact integrity and predictions")
