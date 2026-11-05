"""Regression tests for the final raw-input deployment contract (no model fit)."""
import numpy as np
import pandas as pd
import pytest

from src.predict import load_artifacts, predict


def test_raw_prediction_ignores_target_and_post_call_duration():
    pipeline, model = load_artifacts()
    raw = pd.read_csv("data/sample_test.csv")
    expected = predict(raw, pipeline, model)
    changed = raw.copy()
    changed["duration"] = 999999
    changed["deposit"] = "not_a_training_label"
    actual = predict(changed, pipeline, model)
    np.testing.assert_array_equal(actual.predicted_deposit, expected.predicted_deposit)
    np.testing.assert_allclose(actual.probability_yes, expected.probability_yes)
    assert (actual.score_kind == "uncalibrated_predict_proba").all()


def test_raw_schema_rejects_missing_required_feature():
    pipeline, model = load_artifacts()
    raw = pd.read_csv("data/sample_test.csv").drop(columns="age")
    with pytest.raises(ValueError):
        predict(raw, pipeline, model)


def test_margin_is_not_exported_as_probability():
    class Identity:
        def transform(self, X):
            return X

    class MarginOnly:
        def predict(self, X):
            return np.array(["no", "yes"])

        def decision_function(self, X):
            return np.array([-2.0, 3.0])

    out = predict(pd.DataFrame({"x": [0, 1]}), Identity(), MarginOnly())
    assert out.probability_yes.isna().all()
    assert out.confidence_score.isna().all()
    np.testing.assert_array_equal(out.decision_score, [-2.0, 3.0])


def test_orchestrator_selects_file_backend_without_gui(monkeypatch):
    import importlib
    import run_pipeline

    monkeypatch.delenv("MPLBACKEND", raising=False)
    importlib.reload(run_pipeline)
    import os
    assert os.environ["MPLBACKEND"] == "Agg"
