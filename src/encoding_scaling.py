"""Mô-đun mã hóa đặc trưng phân loại, co giãn biến số và phân chia tập dữ liệu."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from src.config import (
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)

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
    """Phân chia tập dữ liệu thành tập huấn luyện và kiểm thử có phân tầng nhãn mục tiêu."""
    if target_column not in data.columns:
        raise ValueError(f"Không tìm thấy cột mục tiêu '{target_column}' trong DataFrame.")

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
    """Trả về danh sách cấu hình bộ biến đổi mã hóa và co giãn cho ColumnTransformer."""
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
    """Khởi tạo đối tượng ColumnTransformer áp dụng các bộ mã hóa và chuẩn hóa lên dữ liệu."""
    configurations = list(transformer_configurations or get_transformer_configurations())
    return ColumnTransformer(transformers=configurations, remainder="drop")


def get_feature_configuration_summary() -> pd.DataFrame:
    """Tạo bảng tóm tắt lý do lựa chọn các phương pháp mã hóa và co giãn đặc trưng."""
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
                "Lý do": "Tránh áp đặt quan hệ thứ bậc giả; bỏ qua nhãn mới ở tập test.",
            },
            {
                "Nhóm cột": "Biến số có ngoại lai và đuôi lệch",
                "Cột": ", ".join(ROBUST_SCALING_COLUMNS),
                "Bộ chuyển đổi": "RobustScaler",
                "Lý do": "Dùng trung vị và khoảng tứ phân vị IQR bền vững với giá trị ngoại lai.",
            },
            {
                "Nhóm cột": "Biến số phân bố tương đối đều",
                "Cột": ", ".join(STANDARD_SCALING_COLUMNS),
                "Bộ chuyển đổi": "StandardScaler",
                "Lý do": "Chuẩn hóa về trung bình 0 và phương sai 1 cho biến tuổi và ngày.",
            },
        ]
    )


def _load_default_dataset() -> pd.DataFrame:
    """Nạp tệp dữ liệu gốc từ đường dẫn cấu hình mặc định."""
    return pd.read_csv(RAW_DATA_PATH)


def main() -> None:
    """Thực thi kiểm tra mẫu quy trình phân tách và tiền xử lý mã hóa dữ liệu."""
    data = _load_default_dataset()
    X_train, X_test, y_train, y_test = split_data(data)

    print("Kiểm thử phân tách dữ liệu và cấu hình biến đổi:")
    print(f"Kích thước tập Train: {X_train.shape} | Tập Test: {X_test.shape}")
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    print(f"Kích thước sau biến đổi: {X_train_trans.shape}")


if __name__ == "__main__":
    main()
