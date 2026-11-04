# Tree Model Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible Random Forest training module that loads processed train data, evaluates default and tuned models, saves the best tree model, exports feature importances, and writes the Vietnamese report section.

**Architecture:** Implement one focused module, `src/models/tree_model.py`, mirroring the established structure in `src/models/linear_model.py`. The module will consume the shared CV/tuning utilities from `src/tuning.py`, keep tuning train-only, and produce model/report artifacts under `models/` and `reports/`.

**Tech Stack:** Python, pandas, numpy, scikit-learn `RandomForestClassifier`, `cross_validate`, project-local `src.tuning`, joblib, pytest.

## Global Constraints

- Use `RandomForestClassifier` because the project is solving a classification task.
- Load training data only from `data/processed/X_train.parquet` and `data/processed/y_train.parquet` for tuning/training.
- Use 5-fold Stratified Cross-Validation with `random_state=42` and `shuffle=True`.
- Use `f1_macro` as the primary scoring metric, matching the existing linear model work.
- Tune only `n_estimators`, `max_depth`, and `min_samples_split` for overfitting control.
- Save the tuned estimator to `models/best_tree_model.pkl`.
- Export feature importances from the final tuned model.
- Write the Vietnamese ensemble/tree report section for Person 4.
- Do not use the held-out test set for hyperparameter tuning.
- Do not modify the preprocessing pipeline or processed data schema.
- Do not add new third-party dependencies.

---

## File Structure

- Create `src/models/tree_model.py`: contains all Random Forest loading, evaluation, tuning, feature-importance export, report writing, and CLI entry point.
- Create `tests/test_tree_model.py`: unit tests for the new module using tiny synthetic DataFrames and monkeypatching to avoid expensive CV during tests.
- Output at runtime `models/best_tree_model.pkl`: serialized best `RandomForestClassifier` after tuning.
- Output at runtime `reports/tree_model_benchmark.csv`: default-vs-tuned Cross-Validation comparison table.
- Output at runtime `reports/tree_feature_importances.csv`: feature importance ranking.
- Output at runtime `reports/week3_tree_model_ensemble.md`: Vietnamese report text for section 3.3.

---

### Task 1: Data Loading and Default Random Forest Evaluation

**Files:**
- Create: `src/models/tree_model.py`
- Create: `tests/test_tree_model.py`

**Interfaces:**
- Consumes: `src.tuning.make_cv_splitter(task_type: str, n_splits: int, random_state: int | None, shuffle: bool)`
- Produces:
  - `load_data(train_x_path: str = "data/processed/X_train.parquet", train_y_path: str = "data/processed/y_train.parquet") -> tuple[pd.DataFrame, pd.Series]`
  - `evaluate_default_tree_model(X_train: pd.DataFrame, y_train: pd.Series, cv) -> dict[str, float | str]`

- [ ] **Step 1: Write failing tests for loading data and default evaluation**

Create `tests/test_tree_model.py` with this content:

```python
import pandas as pd

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
```

- [ ] **Step 2: Run tests to verify they fail because module does not exist**

Run:

```bash
cd /d/261/ML/BTL_ML && pytest tests/test_tree_model.py -v
```

Expected: FAIL with `ImportError` or `ModuleNotFoundError` for `src.models.tree_model`.

- [ ] **Step 3: Implement the module header, imports, `load_data`, and default evaluation**

Create `src/models/tree_model.py` with this content:

```python
"""
tree_model.py
Nhiệm vụ: Huấn luyện Random Forest, đánh giá trước/sau tối ưu
         siêu tham số, lưu mô hình tốt nhất và xuất độ quan trọng
         của các đặc trưng.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

from src.tuning import make_cv_splitter, tune_model


def load_data(
    train_x_path="data/processed/X_train.parquet",
    train_y_path="data/processed/y_train.parquet",
):
    """Nạp dữ liệu huấn luyện đã được tiền xử lý từ data/processed."""
    if not os.path.exists(train_x_path) or not os.path.exists(train_y_path):
        raise FileNotFoundError(
            f"Không tìm thấy file huấn luyện tại "
            f"{train_x_path} hoặc {train_y_path}"
        )

    X_train = pd.read_parquet(train_x_path)
    y_train = pd.read_parquet(train_y_path)

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]

    return X_train, y_train


def evaluate_default_tree_model(X_train, y_train, cv):
    """Đánh giá Random Forest với cấu hình mặc định có random_state cố định."""
    model_default = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )

    scores = cross_validate(
        model_default,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )

    metrics = {
        "Model": "Random Forest (Default)",
        "CV_F1_Macro_Mean": float(np.mean(scores["test_score"])),
    }

    return metrics
```

- [ ] **Step 4: Run tests to verify Task 1 passes**

Run:

```bash
cd /d/261/ML/BTL_ML && pytest tests/test_tree_model.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
cd /d/261/ML/BTL_ML && git add src/models/tree_model.py tests/test_tree_model.py && git commit -m "Add Random Forest data loading and default evaluation

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

Expected: commit succeeds.

---

### Task 2: Random Forest Hyperparameter Tuning and Feature Importances

**Files:**
- Modify: `src/models/tree_model.py`
- Modify: `tests/test_tree_model.py`

**Interfaces:**
- Consumes:
  - `src.tuning.tune_model(estimator, param_grid, cv_splitter, scoring_metric, *, X, y, ...)`
  - `evaluate_default_tree_model(...)` from Task 1
- Produces:
  - `tune_tree_model(X_train: pd.DataFrame, y_train: pd.Series, cv) -> tuple[RandomForestClassifier, dict[str, object]]`
  - `extract_and_save_feature_importances(model, feature_names, output_path: str = "reports/tree_feature_importances.csv") -> pd.DataFrame`

- [ ] **Step 1: Append failing tests for tuning and feature importances**

Append this content to `tests/test_tree_model.py`:

```python

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
        best_estimator = best_estimator
        best_score = 0.8123
        best_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 10,
        }

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
```

Also add this import near the top of `tests/test_tree_model.py`:

```python
from sklearn.ensemble import RandomForestClassifier
```

- [ ] **Step 2: Run tests to verify new tests fail because functions are missing**

Run:

```bash
cd /d/261/ML/BTL_ML && pytest tests/test_tree_model.py -v
```

Expected: FAIL with `AttributeError` for `tune_tree_model` and `extract_and_save_feature_importances`.

- [ ] **Step 3: Add tuning and feature importance functions**

Append this code to `src/models/tree_model.py` after `evaluate_default_tree_model`:

```python

def tune_tree_model(X_train, y_train, cv):
    """Tối ưu Random Forest thông qua framework tuning dùng chung."""
    model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    }

    result = tune_model(
        estimator=model,
        param_grid=param_grid,
        cv_splitter=cv,
        scoring_metric="f1_macro",
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
    model,
    feature_names,
    output_path="reports/tree_feature_importances.csv",
):
    """Xuất độ quan trọng đặc trưng của Random Forest để phân tích mô hình."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    }).sort_values(
        by="Importance",
        ascending=False,
    )

    df_importances.to_csv(output_path, index=False)

    print(
        f"-> Đã xuất bảng độ quan trọng đặc trưng tại: "
        f"{output_path}"
    )

    return df_importances
```

- [ ] **Step 4: Run tests to verify Task 2 passes**

Run:

```bash
cd /d/261/ML/BTL_ML && pytest tests/test_tree_model.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
cd /d/261/ML/BTL_ML && git add src/models/tree_model.py tests/test_tree_model.py && git commit -m "Add Random Forest tuning and feature importances

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

Expected: commit succeeds.

---

### Task 3: Benchmark Table, Vietnamese Report, and CLI Entry Point

**Files:**
- Modify: `src/models/tree_model.py`
- Modify: `tests/test_tree_model.py`

**Interfaces:**
- Consumes:
  - `load_data(...) -> tuple[pd.DataFrame, pd.Series]`
  - `evaluate_default_tree_model(...) -> dict[str, float | str]`
  - `tune_tree_model(...) -> tuple[RandomForestClassifier, dict[str, object]]`
  - `extract_and_save_feature_importances(...) -> pd.DataFrame`
- Produces:
  - `build_comparison_table(default_metrics: dict, tuned_metrics: dict) -> pd.DataFrame`
  - `write_tree_report(comparison_table: pd.DataFrame, tuned_metrics: dict, output_path: str = "reports/week3_tree_model_ensemble.md") -> None`
  - `main() -> None`

- [ ] **Step 1: Append failing tests for comparison table and report writing**

Append this content to `tests/test_tree_model.py`:

```python

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
```

- [ ] **Step 2: Run tests to verify new functions are missing**

Run:

```bash
cd /d/261/ML/BTL_ML && pytest tests/test_tree_model.py -v
```

Expected: FAIL with `AttributeError` for `build_comparison_table` and `write_tree_report`.

- [ ] **Step 3: Add comparison table, report writer, and main function**

Append this code to `src/models/tree_model.py` after `extract_and_save_feature_importances`:

```python

def build_comparison_table(default_metrics, tuned_metrics):
    """Tổng hợp điểm Cross-Validation trước và sau tuning."""
    comparison_table = pd.DataFrame([
        {
            "Mô hình": default_metrics["Model"],
            "CV F1-Macro": round(
                default_metrics["CV_F1_Macro_Mean"],
                4,
            ),
            "Ghi chú / Tham số": "RandomForestClassifier(random_state=42)",
        },
        {
            "Mô hình": tuned_metrics["Model"],
            "CV F1-Macro": round(
                tuned_metrics["CV_F1_Macro_Mean"],
                4,
            ),
            "Ghi chú / Tham số": json.dumps(
                tuned_metrics["Best_Params"],
            ),
        },
    ])

    return comparison_table


def write_tree_report(
    comparison_table,
    tuned_metrics,
    output_path="reports/week3_tree_model_ensemble.md",
):
    """Ghi nội dung báo cáo tiếng Việt cho mô hình cây và ensemble."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    best_params = tuned_metrics["Best_Params"]
    markdown_table = comparison_table.to_markdown(index=False)

    content = f"""# Mục 3.3 — Mô hình cây và Ensemble

## 3.3. Mô hình cây quyết định và Ensemble

Đối với nhóm mô hình phi tuyến, nhóm sử dụng **Random Forest** làm đại diện cho họ mô hình cây và ensemble. Random Forest kết hợp nhiều cây quyết định, trong đó mỗi cây được huấn luyện trên một mẫu bootstrap khác nhau của tập train. Khi dự đoán bài toán phân loại, mô hình tổng hợp kết quả bằng cơ chế bỏ phiếu đa số giữa các cây thành phần.

### Nguyên lý Bagging và giảm phương sai

Random Forest dựa trên nguyên lý **Bagging** (Bootstrap Aggregating). Thay vì huấn luyện một cây quyết định duy nhất, thuật toán tạo nhiều tập dữ liệu con bằng cách lấy mẫu có hoàn lại từ tập train, sau đó huấn luyện một cây trên từng tập con. Các cây quyết định riêng lẻ thường có phương sai cao vì chúng dễ thay đổi mạnh khi dữ liệu huấn luyện thay đổi. Khi kết hợp nhiều cây ít tương quan với nhau, sai số riêng của từng cây có xu hướng được triệt tiêu một phần trong quá trình bỏ phiếu, nhờ đó ensemble giúp **giảm phương sai** và hạn chế overfitting so với một cây đơn lẻ.

Ngoài bootstrap sampling, Random Forest còn chọn ngẫu nhiên một tập con đặc trưng tại mỗi lần tách node. Cơ chế này làm các cây đa dạng hơn, giảm tương quan giữa các cây và tăng hiệu quả tổng hợp của ensemble.

### Thiết lập thực nghiệm

Nhóm đánh giá hai mốc so sánh:

- **Random Forest Default:** `RandomForestClassifier(random_state=42, n_jobs=-1)`.
- **Random Forest Tuned:** sử dụng framework tuning chung trong `src/tuning.py`, tìm kiếm trên các siêu tham số `n_estimators`, `max_depth`, và `min_samples_split`.

Các mô hình được đánh giá bằng **5-fold Stratified Cross-Validation** trên tập train, sử dụng **F1-Macro** làm thước đo chính. Việc chỉ dùng tập train trong Cross-Validation giúp tránh rò rỉ thông tin từ tập test vào quá trình chọn siêu tham số.

Ba siêu tham số được chọn vì chúng kiểm soát trực tiếp độ phức tạp và khả năng overfitting của mô hình:

- `n_estimators`: số lượng cây trong rừng. Số cây lớn hơn thường giúp kết quả ổn định hơn, nhưng làm tăng thời gian huấn luyện.
- `max_depth`: độ sâu tối đa của từng cây. Giới hạn độ sâu giúp cây bớt học quá chi tiết các nhiễu trong dữ liệu.
- `min_samples_split`: số mẫu tối thiểu để tiếp tục phân nhánh một node. Giá trị lớn hơn làm cây thận trọng hơn khi tách nhánh.

### Kết quả Cross-Validation

{markdown_table}

Cấu hình tốt nhất tìm được cho Random Forest là `{json.dumps(best_params)}`. Mô hình tuned được refit trên toàn bộ tập train thông qua `GridSearchCV` và được lưu tại `models/best_tree_model.pkl`.

Sau khi huấn luyện, nhóm xuất mảng độ quan trọng đặc trưng `feature_importances_` ra `reports/tree_feature_importances.csv`. Bảng này cho biết mức đóng góp tương đối của từng đặc trưng trong quá trình tách nhánh của các cây, hỗ trợ phân tích những biến có ảnh hưởng lớn đến quyết định phân loại của mô hình.
"""

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(content)

    print(
        f"-> Đã ghi nội dung báo cáo mô hình cây tại: "
        f"{output_path}"
    )


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("=== BẮT ĐẦU HUẤN LUYỆN TREE MODEL & ENSEMBLE ===")

    X_train, y_train = load_data()

    print(
        f"Kích thước tập Train: "
        f"{X_train.shape[0]} mẫu, "
        f"{X_train.shape[1]} đặc trưng."
    )

    cv = make_cv_splitter(
        task_type="classification",
        n_splits=5,
        random_state=42,
        shuffle=True,
    )

    print("\n1. Đang đánh giá Random Forest (Default)...")

    default_metrics = evaluate_default_tree_model(
        X_train,
        y_train,
        cv,
    )

    print("2. Đang tuning Random Forest...")

    best_tree_model, tuned_metrics = tune_tree_model(
        X_train,
        y_train,
        cv,
    )

    model_output_path = "models/best_tree_model.pkl"

    joblib.dump(
        best_tree_model,
        model_output_path,
    )

    print(
        f"-> Đã lưu mô hình tối ưu vào: "
        f"{model_output_path}"
    )

    extract_and_save_feature_importances(
        best_tree_model,
        X_train.columns,
    )

    comparison_table = build_comparison_table(
        default_metrics,
        tuned_metrics,
    )

    comparison_table.to_csv(
        "reports/tree_model_benchmark.csv",
        index=False,
    )

    write_tree_report(
        comparison_table,
        tuned_metrics,
    )

    print("\n=== KẾT QUẢ MÔ HÌNH CÂY & ENSEMBLE ===")
    print(
        comparison_table.to_string(index=False)
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify Task 3 passes**

Run:

```bash
cd /d/261/ML/BTL_ML && pytest tests/test_tree_model.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Run all existing tests**

Run:

```bash
cd /d/261/ML/BTL_ML && pytest tests -v
```

Expected: all tests PASS, including `tests/test_encoding_scaling.py` and `tests/test_tree_model.py`.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
cd /d/261/ML/BTL_ML && git add src/models/tree_model.py tests/test_tree_model.py && git commit -m "Add Random Forest report and training entry point

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

Expected: commit succeeds.

---

### Task 4: Execute Training and Verify Required Artifacts

**Files:**
- Runtime create/modify: `models/best_tree_model.pkl`
- Runtime create/modify: `reports/tree_model_benchmark.csv`
- Runtime create/modify: `reports/tree_feature_importances.csv`
- Runtime create/modify: `reports/week3_tree_model_ensemble.md`

**Interfaces:**
- Consumes: `python -m src.models.tree_model` CLI entry point from Task 3.
- Produces: all required project deliverables for the tree/ensemble section.

- [ ] **Step 1: Run the training module**

Run:

```bash
cd /d/261/ML/BTL_ML && python -m src.models.tree_model
```

Expected: command finishes successfully and prints:

```text
=== BẮT ĐẦU HUẤN LUYỆN TREE MODEL & ENSEMBLE ===
...
=== KẾT QUẢ MÔ HÌNH CÂY & ENSEMBLE ===
```

- [ ] **Step 2: Verify required artifact files exist**

Run:

```bash
cd /d/261/ML/BTL_ML && python - <<'PY'
from pathlib import Path

required = [
    Path('models/best_tree_model.pkl'),
    Path('reports/tree_model_benchmark.csv'),
    Path('reports/tree_feature_importances.csv'),
    Path('reports/week3_tree_model_ensemble.md'),
]

missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f'Missing required artifacts: {missing}')

for path in required:
    print(f'OK {path}')
PY
```

Expected: prints `OK` for all four required paths.

- [ ] **Step 3: Verify benchmark table contains default and tuned rows**

Run:

```bash
cd /d/261/ML/BTL_ML && python - <<'PY'
import pandas as pd

benchmark = pd.read_csv('reports/tree_model_benchmark.csv')
models = benchmark['Mô hình'].tolist()
assert models == ['Random Forest (Default)', 'Random Forest (Tuned)'], models
assert benchmark['CV F1-Macro'].notna().all()
print(benchmark.to_string(index=False))
PY
```

Expected: prints a two-row table with non-empty CV F1-Macro values.

- [ ] **Step 4: Verify saved model and feature importances are usable**

Run:

```bash
cd /d/261/ML/BTL_ML && python - <<'PY'
import joblib
import pandas as pd

model = joblib.load('models/best_tree_model.pkl')
assert hasattr(model, 'feature_importances_')

importances = pd.read_csv('reports/tree_feature_importances.csv')
assert list(importances.columns) == ['Feature', 'Importance']
assert len(importances) > 0
assert importances['Importance'].notna().all()
print(importances.head(10).to_string(index=False))
PY
```

Expected: prints top 10 feature importances and raises no assertion errors.

- [ ] **Step 5: Run all tests after artifact generation**

Run:

```bash
cd /d/261/ML/BTL_ML && pytest tests -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit Task 4 artifacts**

Run:

```bash
cd /d/261/ML/BTL_ML && git add models/best_tree_model.pkl reports/tree_model_benchmark.csv reports/tree_feature_importances.csv reports/week3_tree_model_ensemble.md && git commit -m "Add trained Random Forest artifacts and report

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

Expected: commit succeeds.

---

## Self-Review

- Spec coverage: The plan covers loading processed train data, training `RandomForestClassifier`, tuning through `src/tuning.py`, searching `n_estimators`, `max_depth`, `min_samples_split`, saving `models/best_tree_model.pkl`, exporting `feature_importances_`, writing CV results before/after tuning, and producing the Vietnamese Bagging/Ensemble report text.
- Placeholder scan: The plan contains no `TBD`, `TODO`, incomplete sections, or references to undefined functions. Each code-changing step includes concrete code.
- Type consistency: Function names and signatures are consistent across tasks: `load_data`, `evaluate_default_tree_model`, `tune_tree_model`, `extract_and_save_feature_importances`, `build_comparison_table`, `write_tree_report`, and `main`.
