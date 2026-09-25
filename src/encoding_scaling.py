"""
Encoding, scaling and data splitting utilities for the Bank Marketing dataset.

This module keeps the preprocessing configuration independent from model
training so it can be reused in notebooks, scripts and sklearn Pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

TARGET_COLUMN = "deposit"
RANDOM_STATE = 42
TEST_SIZE = 0.2

ORDINAL_COLUMNS = ["education"]
EDUCATION_ORDER = ["primary", "secondary", "tertiary"]

ONEHOT_COLUMNS = [
    "job",
    "marital",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

ROBUST_SCALING_COLUMNS = [
    "balance",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

STANDARD_SCALING_COLUMNS = ["age", "day"]


def split_data(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a dataset into train/test sets with optional target stratification.

    Parameters
    ----------
    data:
        Full input dataframe containing features and the target column.
    target_column:
        Name of the target column to separate from feature columns.
    test_size:
        Fraction of rows reserved for the test set. Default 0.2 gives 80/20.
    random_state:
        Fixed random seed for reproducible splits.
    stratify:
        When True, pass `stratify=y` to preserve the target class ratio in
        train and test sets. This is recommended for classification tasks.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        `X_train, X_test, y_train, y_test`.
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    X = data.drop(columns=[target_column])
    y = data[target_column]
    stratify_target = y if stratify else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )


def get_transformer_configurations() -> List[Tuple[str, object, List[str]]]:
    """Return ColumnTransformer-compatible preprocessing configurations.

    Each item follows sklearn's `(name, transformer, columns)` convention.
    """
    return [
        (
            "ordinal_education",
            OrdinalEncoder(
                categories=[EDUCATION_ORDER],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            ORDINAL_COLUMNS,
        ),
        (
            "onehot_nominal",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ONEHOT_COLUMNS,
        ),
        (
            "robust_numeric",
            RobustScaler(),
            ROBUST_SCALING_COLUMNS,
        ),
        (
            "standard_numeric",
            StandardScaler(),
            STANDARD_SCALING_COLUMNS,
        ),
    ]


def build_preprocessor(
    transformer_configurations: Sequence[Tuple[str, object, List[str]]] | None = None,
) -> ColumnTransformer:
    """Build the sklearn ColumnTransformer for encoding and scaling features."""
    configurations = list(transformer_configurations or get_transformer_configurations())
    return ColumnTransformer(transformers=configurations, remainder="drop")


def get_feature_configuration_summary() -> pd.DataFrame:
    """Return a human-readable summary of selected encoders/scalers."""
    return pd.DataFrame(
        [
            {
                "Nhóm cột": "Biến phân loại có thứ bậc",
                "Cột": ", ".join(ORDINAL_COLUMNS),
                "Bộ chuyển đổi": "OrdinalEncoder",
                "Lý do": "education có thứ tự tự nhiên primary < secondary < tertiary.",
            },
            {
                "Nhóm cột": "Biến phân loại danh nghĩa",
                "Cột": ", ".join(ONEHOT_COLUMNS),
                "Bộ chuyển đổi": "OneHotEncoder(handle_unknown='ignore')",
                "Lý do": "Không áp đặt quan hệ thứ tự giả; bỏ qua nhãn lạ ở tập test để tránh lỗi.",
            },
            {
                "Nhóm cột": "Biến số lệch/dao động lớn",
                "Cột": ", ".join(ROBUST_SCALING_COLUMNS),
                "Bộ chuyển đổi": "RobustScaler",
                "Lý do": "Các cột như balance, duration, campaign, pdays, previous dễ có ngoại lai nên dùng median và IQR bền vững hơn.",
            },
            {
                "Nhóm cột": "Biến số ổn định hơn",
                "Cột": ", ".join(STANDARD_SCALING_COLUMNS),
                "Bộ chuyển đổi": "StandardScaler",
                "Lý do": "age và day có miền giá trị rõ hơn, phù hợp chuẩn hóa về mean=0 và std=1.",
            },
        ]
    )


def _load_default_dataset() -> pd.DataFrame:
    """Load the project dataset when this file is executed directly."""
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "raw" / "dataset.csv"
    return pd.read_csv(dataset_path)


def main() -> None:
    """Run a small smoke test showing the split and preprocessing setup."""
    data = _load_default_dataset()
    X_train, X_test, y_train, y_test = split_data(data)

    print("Encoding, Scaling & Splitting smoke test")
    print("=" * 50)
    print(f"Dataset shape: {data.shape}")
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape} | y_test: {y_test.shape}")
    print("\nTarget distribution:")
    print(pd.DataFrame({"train": y_train.value_counts(normalize=True), "test": y_test.value_counts(normalize=True)}))

    print("\nTransformer configurations:")
    for name, transformer, columns in get_transformer_configurations():
        print(f"- {name}: {transformer.__class__.__name__} -> {columns}")

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    print("\nTransformed shapes:")
    print(f"X_train_transformed: {X_train_transformed.shape}")
    print(f"X_test_transformed: {X_test_transformed.shape}")


if __name__ == "__main__":
    main()
