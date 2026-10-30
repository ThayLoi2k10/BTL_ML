"""Mô-đun tiền xử lý làm sạch dữ liệu thô và cắt tỉa giá trị ngoại lai."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


class StringCleaner(BaseEstimator, TransformerMixin):
    """Chuẩn hóa định dạng chuỗi và đồng nhất các nhãn phân loại."""

    def __init__(self, columns: list[str] | None = None):
        """Khởi tạo danh sách các cột phân loại cần chuẩn hóa chuỗi."""
        self.columns = columns if columns is not None else CATEGORICAL_COLUMNS

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> StringCleaner:
        """Ghi nhận trạng thái cấu hình của bộ chuẩn hóa chuỗi."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Thực hiện xóa khoảng trắng thừa, chuyển chữ thường và chuẩn hóa nhãn nghề nghiệp."""
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].astype("string").str.strip().str.lower()

        if "job" in self.columns and "job" in X.columns:
            X["job"] = X["job"].replace("admin.", "admin")

        return X


class MissingHandler(BaseEstimator, TransformerMixin):
    """Xử lý giá trị khuyết thiếu cho cả biến phân loại và biến số."""

    def __init__(
        self,
        numeric_strategy: dict[str, str] | None = None,
        categorical_missing_value: str = "missing",
    ):
        """Khởi tạo chiến lược điền giá trị thiếu cho biến số và biến phân loại."""
        self.numeric_strategy = numeric_strategy
        self.categorical_missing_value = categorical_missing_value

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> MissingHandler:
        """Tính toán giá trị thống kê (mean hoặc median) trên tập huấn luyện cho từng biến số."""
        if self.numeric_strategy is None:
            self.numeric_strategy_ = {
                "age": "median",
                "balance": "median",
                "day": "mean",
                "duration": "median",
                "campaign": "median",
                "pdays": "median",
                "previous": "median",
            }
        else:
            self.numeric_strategy_ = self.numeric_strategy

        self.fill_values_ = {}
        for col in NUMERICAL_COLUMNS:
            if col not in X.columns:
                continue

            series = X[col]
            if col == "pdays":
                series = series[series >= 0]

            if self.numeric_strategy_.get(col) == "mean":
                self.fill_values_[col] = float(series.mean())
            elif self.numeric_strategy_.get(col) == "median":
                self.fill_values_[col] = float(series.median())
            else:
                raise ValueError(f"Chiến lược không hợp lệ cho cột '{col}'.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Thay thế giá trị 'unknown'/NaN bằng nhãn thiếu hoặc giá trị thống kê đã học."""
        X = X.copy()

        for col in CATEGORICAL_COLUMNS:
            if col in X.columns:
                X[col] = (
                    X[col]
                    .astype("string")
                    .replace("unknown", self.categorical_missing_value)
                    .fillna(self.categorical_missing_value)
                )

        for col in NUMERICAL_COLUMNS:
            if col in X.columns and col in self.fill_values_:
                X[col] = X[col].fillna(self.fill_values_[col])

        return X


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Cắt tỉa các giá trị ngoại lai của biến số dựa trên khoảng tứ phân vị IQR."""

    def __init__(self, factor: float = 1.5):
        """Khởi tạo hệ số nhân khoảng IQR để xác định biên ngoại lai."""
        self.factor = factor

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> OutlierClipper:
        """Tính toán ngưỡng cận trên và cận dưới theo IQR trên tập huấn luyện."""
        candidate_columns = [
            "age",
            "balance",
            "duration",
            "campaign",
            "previous",
            "pdays",
        ]
        self.clip_columns_ = [col for col in candidate_columns if col in X.columns]
        self.non_negative_columns_ = ["duration", "campaign", "previous", "pdays"]
        self.bounds_ = {}

        for col in self.clip_columns_:
            series = X[col]
            if col == "pdays":
                series = series[series >= 0]

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            lower = q1 - self.factor * iqr
            upper = q3 + self.factor * iqr

            if col in self.non_negative_columns_:
                lower = max(0.0, lower)

            self.bounds_[col] = (lower, upper)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Giới hạn các giá trị của biến số nằm trong khoảng biên đã tính toán."""
        X = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            if col not in X.columns:
                continue

            if col == "pdays":
                mask = X["pdays"] >= 0
                X.loc[mask, "pdays"] = X.loc[mask, "pdays"].clip(lower=lower, upper=upper)
            else:
                X[col] = X[col].clip(lower=lower, upper=upper)

        return X
