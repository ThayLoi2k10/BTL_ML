import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


CATEGORICAL_COLUMNS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

NUMERICAL_COLUMNS = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]


class StringCleaner(BaseEstimator, TransformerMixin):
    """Chuẩn hóa các cột categorical."""

    def __init__(self, columns=None):
        self.columns = columns if columns is not None else CATEGORICAL_COLUMNS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for col in self.columns:
            X[col] = (
                X[col].astype("string").str.strip().str.lower()
            )

        # Chuẩn hóa nhãn bị phát hiện trong EDA
        if "job" in self.columns:
            X["job"] = X["job"].replace("admin.", "admin")

        return X


class MissingHandler(BaseEstimator, TransformerMixin):
    """
    Xử lý missing values.

    - categorical: 'unknown' / NaN -> 'missing'
    - numerical: điền mean hoặc median theo strategy
    - pdays = -1 được giữ nguyên
    """

    def __init__(
        self,
        numeric_strategy=None,
        categorical_missing_value="missing",
    ):
        self.numeric_strategy = numeric_strategy
        self.categorical_missing_value = categorical_missing_value

    def fit(self, X, y=None):
        # Strategy mặc định dựa trên EDA
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

        # Học giá trị thay thế từ tập train
        self.fill_values_ = {}

        for col in NUMERICAL_COLUMNS:
            series = X[col]

            # pdays = -1 là sentinel, không dùng để tính statistic
            if col == "pdays":
                series = series[series >= 0]

            if self.numeric_strategy_[col] == "mean":
                self.fill_values_[col] = series.mean()
            elif self.numeric_strategy_[col] == "median":
                self.fill_values_[col] = series.median()
            else:
                raise ValueError(
                    f"Strategy không hợp lệ cho '{col}': "
                    f"{self.numeric_strategy_[col]}"
                )

        return self

    def transform(self, X):
        X = X.copy()

        # Categorical: unknown / NaN -> missing
        for col in CATEGORICAL_COLUMNS:
            X[col] = (
                X[col]
                .astype("string")
                .replace("unknown", self.categorical_missing_value)
                .fillna(self.categorical_missing_value)
            )

        # Numerical: NaN -> statistic đã học từ train
        for col in NUMERICAL_COLUMNS:
            X[col] = X[col].fillna(self.fill_values_[col])

        return X


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Cắt tỉa outlier bằng ngưỡng 1.5 × IQR."""

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        # Các biến cần clipping
        self.clip_columns_ = [
            "age",
            "balance",
            "duration",
            "campaign",
            "previous",
            "pdays",
        ]

        # Các biến không được phép âm
        self.non_negative_columns_ = [
            "duration",
            "campaign",
            "previous",
            "pdays",
        ]

        self.bounds_ = {}

        for col in self.clip_columns_:
            series = X[col]

            # pdays = -1 là sentinel, không tham gia tính IQR
            if col == "pdays":
                series = series[series >= 0]

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - self.factor * iqr
            upper = q3 + self.factor * iqr

            # Các biến này không thể có giá trị âm
            if col in self.non_negative_columns_:
                lower = max(0, lower)

            self.bounds_[col] = (lower, upper)

        return self

    def transform(self, X):
        X = X.copy()

        for col, (lower, upper) in self.bounds_.items():

            if col == "pdays":
                # Chỉ clip giá trị pdays >= 0.
                # pdays = -1 được giữ nguyên.
                mask = X["pdays"] >= 0
                X.loc[mask, "pdays"] = (
                    X.loc[mask, "pdays"]
                    .clip(lower=lower, upper=upper)
                )

            else:
                X[col] = X[col].clip(lower=lower, upper=upper,)

        return X



# Ghép các bước cleaning thành một chuỗi xử lý thống nhất
# để có thể fit/transform dữ liệu theo chuẩn Scikit-Learn.
""" from sklearn.pipeline import Pipeline

cleaning_pipeline = Pipeline([
    ("string", StringCleaner()),
    ("missing", MissingHandler()),
    ("outlier", OutlierClipper()),
]) """