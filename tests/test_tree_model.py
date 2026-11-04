import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models import tree_model


def test_load_data_returns_features_and_series_target(tmp_path):
    X = pd.DataFrame({"feature_a": [0.1, 0.2], "feature_b": [1.0, 2.0]})
    y = pd.DataFrame({"target": [0, 1]})
    x_path = tmp_path / "X_train.parquet"
    y_path = tmp_path / "y_train.parquet"
    X.to_parquet(x_path)
    y.to_parquet(y_path)

    loaded_X, loaded_y = tree_model.load_data(str(x_path), str(y_path))

    pd.testing.assert_frame_equal(loaded_X, X)
    assert isinstance(loaded_y, pd.Series)
    assert loaded_y.tolist() == [0, 1]
    assert loaded_y.name == "target"


def test_load_data_raises_when_file_missing(tmp_path):
    missing_x = tmp_path / "missing_X.parquet"
    missing_y = tmp_path / "missing_y.parquet"

    try:
        tree_model.load_data(str(missing_x), str(missing_y))
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")

    assert "Không tìm thấy file huấn luyện" in message


def test_evaluate_default_tree_model_returns_cv_mean(monkeypatch):
    X = pd.DataFrame({"feature_a": [0, 1, 0, 1], "feature_b": [1, 1, 0, 0]})
    y = pd.Series([0, 1, 0, 1], name="target")
    fake_cv = object()
    captured = {}

    def fake_cross_validate(estimator, X_train, y_train, cv, scoring, n_jobs):
        captured["estimator"] = estimator
        captured["X_train"] = X_train
        captured["y_train"] = y_train
        captured["cv"] = cv
        captured["scoring"] = scoring
        captured["n_jobs"] = n_jobs
        return {"test_score": [0.60, 0.80]}

    monkeypatch.setattr(tree_model, "cross_validate", fake_cross_validate)

    metrics = tree_model.evaluate_default_tree_model(X, y, fake_cv)

    assert metrics == {
        "Model": "Random Forest (Default)",
        "CV_F1_Macro_Mean": 0.70,
    }
    assert captured["X_train"] is X
    assert captured["y_train"] is y
    assert captured["cv"] is fake_cv
    assert captured["scoring"] == "f1_macro"
    assert captured["n_jobs"] == -1
    assert captured["estimator"].random_state == 42
    assert captured["estimator"].n_jobs == -1


def test_tune_tree_model_uses_expected_search_space(monkeypatch):
    X = pd.DataFrame({"feature_a": [0, 1, 0, 1], "feature_b": [1, 1, 0, 0]})
    y = pd.Series([0, 1, 0, 1], name="target")
    fake_cv = object()
    best_estimator = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )

    class FakeResult:
        best_score = 0.8123
        best_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 10,
        }

    FakeResult.best_estimator = best_estimator
    captured = {}

    def fake_tune_model(
        estimator,
        param_grid,
        cv_splitter,
        scoring_metric,
        X,
        y,
    ):
        captured["estimator"] = estimator
        captured["param_grid"] = param_grid
        captured["cv_splitter"] = cv_splitter
        captured["scoring_metric"] = scoring_metric
        captured["X"] = X
        captured["y"] = y
        return FakeResult()

    monkeypatch.setattr(tree_model, "tune_model", fake_tune_model)

    model, metrics = tree_model.tune_tree_model(X, y, fake_cv)

    assert model is best_estimator
    assert metrics == {
        "Model": "Random Forest (Tuned)",
        "CV_F1_Macro_Mean": 0.8123,
        "Best_Params": {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 10,
        },
    }
    assert captured["estimator"].random_state == 42
    assert captured["estimator"].n_jobs == -1
    assert captured["param_grid"] == {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    }
    assert captured["cv_splitter"] is fake_cv
    assert captured["scoring_metric"] == "f1_macro"
    assert captured["X"] is X
    assert captured["y"] is y


def test_extract_and_save_feature_importances_sorts_descending(tmp_path):
    class FakeModel:
        feature_importances_ = [0.2, 0.5, 0.3]

    output_path = tmp_path / "tree_feature_importances.csv"

    df = tree_model.extract_and_save_feature_importances(
        FakeModel(),
        ["feature_a", "feature_b", "feature_c"],
        str(output_path),
    )

    assert output_path.exists()
    assert df["Feature"].tolist() == ["feature_b", "feature_c", "feature_a"]
    assert df["Importance"].tolist() == [0.5, 0.3, 0.2]

    persisted = pd.read_csv(output_path)
    assert persisted["Feature"].tolist() == ["feature_b", "feature_c", "feature_a"]
    assert persisted["Importance"].tolist() == [0.5, 0.3, 0.2]


def test_build_comparison_table_formats_default_and_tuned_rows():
    default_metrics = {
        "Model": "Random Forest (Default)",
        "CV_F1_Macro_Mean": 0.70123,
    }
    tuned_metrics = {
        "Model": "Random Forest (Tuned)",
        "CV_F1_Macro_Mean": 0.81234,
        "Best_Params": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
        },
    }

    table = tree_model.build_comparison_table(default_metrics, tuned_metrics)

    assert table.to_dict(orient="records") == [
        {
            "Mô hình": "Random Forest (Default)",
            "CV F1-Macro": 0.7012,
            "Ghi chú / Tham số": "RandomForestClassifier(random_state=42)",
        },
        {
            "Mô hình": "Random Forest (Tuned)",
            "CV F1-Macro": 0.8123,
            "Ghi chú / Tham số": '{"n_estimators": 200, "max_depth": 10, "min_samples_split": 5}',
        },
    ]


def test_write_tree_report_contains_required_vietnamese_sections(tmp_path):
    comparison_table = pd.DataFrame([
        {
            "Mô hình": "Random Forest (Default)",
            "CV F1-Macro": 0.7012,
            "Ghi chú / Tham số": "RandomForestClassifier(random_state=42)",
        },
        {
            "Mô hình": "Random Forest (Tuned)",
            "CV F1-Macro": 0.8123,
            "Ghi chú / Tham số": '{"n_estimators": 200, "max_depth": 10, "min_samples_split": 5}',
        },
    ])
    tuned_metrics = {
        "Best_Params": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
        }
    }
    output_path = tmp_path / "week3_tree_model_ensemble.md"

    tree_model.write_tree_report(
        comparison_table,
        tuned_metrics,
        str(output_path),
    )

    content = output_path.read_text(encoding="utf-8")
    assert "# Mục 3.3 — Mô hình cây và Ensemble" in content
    assert "Bagging" in content
    assert "giảm phương sai" in content
    assert "Random Forest (Default)" in content
    assert "Random Forest (Tuned)" in content
    assert "models/best_tree_model.pkl" in content
    assert "reports/tree_feature_importances.csv" in content
