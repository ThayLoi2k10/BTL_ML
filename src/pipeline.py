"""Mô-đun tích hợp đường ống tiền xử lý chống rò rỉ dữ liệu cho tập dữ liệu Bank Marketing."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.utils.validation import check_is_fitted

from src.cleaning import MissingHandler, OutlierClipper, StringCleaner
from src.config import (
    DATA_PROCESSED_DIR,
    LEAKAGE_COLUMNS,
    MODEL_INPUT_COLUMNS,
    PREPROCESSING_PIPELINE_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.encoding_scaling import (
    build_preprocessor,
    get_transformer_configurations,
)
from src.feature_engineering import (
    BankMarketingFeatureEngineer,
    ENGINEERED_FEATURES,
    MutualInformationSelector,
)

DEFAULT_DATA_PATH = RAW_DATA_PATH
DEFAULT_MODEL_PATH = PREPROCESSING_PIPELINE_PATH
DEFAULT_PROCESSED_DIR = DATA_PROCESSED_DIR

ENGINEERED_CATEGORICAL_COLUMNS = ("age_group",)
ENGINEERED_CONTINUOUS_COLUMNS = (
    "age_squared",
    "balance_positive_log",
    "balance_debt_log",
    "balance_per_age",
    "prior_contact_recency_log",
    "previous_contacts_log",
    "previous_contact_intensity",
    "campaign_contacts_log",
    "campaign_vs_previous_ratio",
    "successful_prior_contact_recency",
    "contact_month_sin",
    "contact_month_cos",
)
ENGINEERED_BINARY_COLUMNS = (
    "has_negative_balance",
    "any_loan",
    "both_loans",
    "credit_obligation_count",
    "financial_stress_flag",
    "loan_with_negative_balance",
    "prior_contacted",
    "prior_contact_recent_30d",
    "campaign_repeat_contact",
    "failed_prior_campaign_recontact",
    "contact_known",
)


class PreCallSchemaGuard(BaseEstimator, TransformerMixin):
    """Xác thực lược đồ dữ liệu đầu vào và chủ động loại bỏ cột thời lượng cuộc gọi rò rỉ."""

    def __init__(
        self,
        required_columns: tuple[str, ...] = MODEL_INPUT_COLUMNS,
        leakage_columns: tuple[str, ...] = LEAKAGE_COLUMNS,
        target_column: str = TARGET_COLUMN,
    ):
        """Khởi tạo danh sách các cột hợp lệ bắt buộc và cột rò rỉ cần loại trừ."""
        self.required_columns = required_columns
        self.leakage_columns = leakage_columns
        self.target_column = target_column

    def fit(self, X: pd.DataFrame, y=None) -> PreCallSchemaGuard:
        """Kiểm tra tính hợp lệ của dữ liệu huấn luyện và ghi nhận các cột đầu ra hợp lệ."""
        X = self._validate(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.fit_row_count_ = len(X)
        self.fit_indices_ = np.asarray(X.index)
        self.leakage_columns_seen_ = tuple(
            column for column in self.leakage_columns if column in X.columns
        )
        self.feature_names_out_ = np.asarray(self.required_columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Trích xuất và trả về đúng tập hợp các đặc trưng hợp lệ trước thời điểm cuộc gọi."""
        check_is_fitted(self, "feature_names_out_")
        X = self._validate(X)
        return X.loc[:, self.required_columns].copy()

    def get_feature_names_out(self, input_features: Sequence[str] | None = None):
        """Trả về danh sách tên các đặc trưng hợp lệ sau bước bảo vệ lược đồ."""
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()

    def _validate(self, X) -> pd.DataFrame:
        """Kiểm tra kiểu dữ liệu DataFrame, ngăn chặn rò rỉ nhãn và kiểm tra cột thiếu hoặc lạ."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Pipeline yêu cầu đầu vào là pandas DataFrame.")
        if self.target_column in X.columns:
            raise ValueError(f"Cột mục tiêu '{self.target_column}' phải được tách riêng trước khi xử lý.")

        required = set(self.required_columns)
        allowed = required.union(self.leakage_columns)
        missing = sorted(required.difference(X.columns))
        unexpected = sorted(set(X.columns).difference(allowed))
        if missing or unexpected:
            raise ValueError(
                f"Lược đồ đặc trưng không khớp. Thiếu={missing}; Thừa={unexpected}."
            )
        return X


def get_integrated_transformer_configurations():
    """Mở rộng cấu hình ColumnTransformer kết hợp biến gốc đã lọc và 24 biến kỹ nghệ mới."""
    configurations = []
    available = set(MODEL_INPUT_COLUMNS).union(ENGINEERED_FEATURES)

    for name, transformer, columns in get_transformer_configurations():
        retained_columns = [column for column in columns if column in available]
        if retained_columns:
            configurations.append((name, clone(transformer), retained_columns))

    configurations.extend(
        [
            (
                "onehot_engineered",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(ENGINEERED_CATEGORICAL_COLUMNS),
            ),
            (
                "robust_engineered",
                RobustScaler(),
                list(ENGINEERED_CONTINUOUS_COLUMNS),
            ),
            (
                "engineered_binary",
                "passthrough",
                list(ENGINEERED_BINARY_COLUMNS),
            ),
        ]
    )
    return configurations


def build_preprocessing_pipeline(
    selected_feature_count: int = 32,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Xây dựng chuỗi Pipeline tiền xử lý hoàn chỉnh từ kiểm tra lược đồ đến chọn lọc Mutual Information."""
    encoder = build_preprocessor(get_integrated_transformer_configurations())
    encoder.set_output(transform="pandas")

    return Pipeline(
        steps=[
            ("schema_guard", PreCallSchemaGuard()),
            ("string_cleaning", StringCleaner()),
            ("missing_values", MissingHandler()),
            ("outlier_clipping", OutlierClipper()),
            ("feature_engineering", BankMarketingFeatureEngineer()),
            ("encoding_scaling", encoder),
            (
                "feature_selection",
                MutualInformationSelector(
                    k=selected_feature_count,
                    random_state=random_state,
                ),
            ),
        ]
    )


def load_correct_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Nạp và kiểm tra tính toàn vẹn của tệp dữ liệu gốc bank.csv."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu tại: {path}")

    data = pd.read_csv(path)
    expected_columns = [*RAW_FEATURE_COLUMNS, TARGET_COLUMN]
    if data.columns.tolist() != expected_columns:
        raise ValueError(
            f"Lược đồ tập dữ liệu không khớp. Mong đợi={expected_columns}; Thực tế={data.columns.tolist()}."
        )
    if set(data[TARGET_COLUMN].dropna().unique()) != {"no", "yes"}:
        raise ValueError("Cột mục tiêu 'deposit' phải chứa chính xác hai nhãn {'no', 'yes'}.")
    return data


def get_fitted_pipeline_summary(pipeline: Pipeline) -> dict:
    """Trích xuất tóm tắt số lượng chiều đặc trưng và danh sách tên cột qua từng bước của pipeline."""
    guard = pipeline.named_steps["schema_guard"]
    engineer = pipeline.named_steps["feature_engineering"]
    encoder = pipeline.named_steps["encoding_scaling"]
    selector = pipeline.named_steps["feature_selection"]

    check_is_fitted(guard, "fit_row_count_")
    check_is_fitted(engineer, "feature_names_out_")
    check_is_fitted(selector, "selected_feature_names_")

    encoded_names = np.asarray(encoder.get_feature_names_out(), dtype=object)
    return {
        "raw_feature_count": int(guard.n_features_in_),
        "leakage_safe_original_count": len(guard.feature_names_out_),
        "engineered_feature_count": len(ENGINEERED_FEATURES),
        "after_feature_engineering_count": len(engineer.feature_names_out_),
        "encoded_candidate_count": len(encoded_names),
        "after_feature_selection_count": len(selector.selected_feature_names_),
        "created_features": list(ENGINEERED_FEATURES),
        "encoded_candidate_features": encoded_names.tolist(),
        "selected_features": selector.selected_feature_names_.tolist(),
        "removed_leakage_features": list(guard.leakage_columns_seen_),
    }


def validate_transformed_splits(
    X_train_transformed: pd.DataFrame,
    X_test_transformed: pd.DataFrame,
    expected_train_rows: int | None = None,
    expected_test_rows: int | None = None,
) -> dict:
    """Kiểm tra tính đồng nhất, hữu hạn và không chứa rò rỉ nhãn giữa hai tập Train và Test."""
    if not isinstance(X_train_transformed, pd.DataFrame) or not isinstance(
        X_test_transformed, pd.DataFrame
    ):
        raise TypeError("Dữ liệu biến đổi phải ở định dạng pandas DataFrame.")
    if list(X_train_transformed.columns) != list(X_test_transformed.columns):
        raise ValueError("Lược đồ cột của tập Train và Test không đồng nhất.")
    if TARGET_COLUMN in X_train_transformed.columns:
        raise ValueError("Cột mục tiêu bị rò rỉ vào ma trận đặc trưng.")
    if any("duration" in str(column).lower() for column in X_train_transformed.columns):
        raise ValueError("Biến rò rỉ duration vẫn còn tồn tại trong ma trận đặc trưng.")
    if expected_train_rows is not None and len(X_train_transformed) != expected_train_rows:
        raise ValueError("Số lượng dòng tập Train biến đổi không khớp kỳ vọng.")
    if expected_test_rows is not None and len(X_test_transformed) != expected_test_rows:
        raise ValueError("Số lượng dòng tập Test biến đổi không khớp kỳ vọng.")

    train_values = X_train_transformed.to_numpy(dtype=float)
    test_values = X_test_transformed.to_numpy(dtype=float)
    if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
        raise ValueError("Ma trận đặc trưng biến đổi chứa NaN hoặc vô cùng.")

    return {
        "train_shape": tuple(X_train_transformed.shape),
        "test_shape": tuple(X_test_transformed.shape),
        "same_schema": True,
        "train_nan_count": int(X_train_transformed.isna().sum().sum()),
        "test_nan_count": int(X_test_transformed.isna().sum().sum()),
        "train_inf_count": int(np.isinf(train_values).sum()),
        "test_inf_count": int(np.isinf(test_values).sum()),
    }


def save_preprocessing_artifact(
    pipeline: Pipeline,
    path: str | Path = DEFAULT_MODEL_PATH,
) -> Path:
    """Lưu trữ đối tượng Pipeline tiền xử lý đã fit thành tệp tin nhị phân joblib."""
    check_is_fitted(pipeline.named_steps["feature_selection"], "selected_indices_")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def save_processed_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    directory: str | Path = DEFAULT_PROCESSED_DIR,
) -> dict[str, Path]:
    """Lưu trữ các tập dữ liệu đã biến đổi thành các tệp Parquet nhằm tăng tốc độ nạp."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "X_train": directory / "X_train.parquet",
        "X_test": directory / "X_test.parquet",
        "y_train": directory / "y_train.parquet",
        "y_test": directory / "y_test.parquet",
    }

    X_train.reset_index(drop=True).to_parquet(paths["X_train"], index=False)
    X_test.reset_index(drop=True).to_parquet(paths["X_test"], index=False)
    y_train.rename(TARGET_COLUMN).reset_index(drop=True).to_frame().to_parquet(
        paths["y_train"], index=False
    )
    y_test.rename(TARGET_COLUMN).reset_index(drop=True).to_frame().to_parquet(
        paths["y_test"], index=False
    )
    return paths


def load_processed_splits(
    directory: str | Path = DEFAULT_PROCESSED_DIR,
) -> dict[str, pd.DataFrame]:
    """Nạp lại 4 tập dữ liệu huấn luyện và kiểm thử dạng Parquet đã tiền xử lý."""
    directory = Path(directory)
    return {
        name: pd.read_parquet(directory / f"{name}.parquet")
        for name in ("X_train", "X_test", "y_train", "y_test")
    }


__all__ = [
    "DEFAULT_DATA_PATH",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_PROCESSED_DIR",
    "ENGINEERED_BINARY_COLUMNS",
    "ENGINEERED_CATEGORICAL_COLUMNS",
    "ENGINEERED_CONTINUOUS_COLUMNS",
    "LEAKAGE_COLUMNS",
    "MODEL_INPUT_COLUMNS",
    "PreCallSchemaGuard",
    "RAW_FEATURE_COLUMNS",
    "build_preprocessing_pipeline",
    "get_fitted_pipeline_summary",
    "get_integrated_transformer_configurations",
    "load_correct_dataset",
    "load_processed_splits",
    "save_preprocessing_artifact",
    "save_processed_splits",
    "validate_transformed_splits",
]
