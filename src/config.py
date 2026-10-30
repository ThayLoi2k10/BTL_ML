"""Cấu hình tập trung toàn bộ đường dẫn, tham số và lược đồ dữ liệu của dự án."""

from pathlib import Path

# Đường dẫn thư mục gốc và các thư mục con trong dự án
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR = DATA_PROCESSED_DIR
RAW_DATA_DIR = DATA_RAW_DIR
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Đường dẫn các tệp dữ liệu và artifact mô hình
RAW_DATA_PATH = DATA_RAW_DIR / "bank.csv"
SAMPLE_TEST_PATH = DATA_DIR / "sample_test.csv"
PREPROCESSING_PIPELINE_PATH = MODELS_DIR / "preprocessing_pipeline.pkl"
CHAMPION_MODEL_PATH = MODELS_DIR / "champion_model.pkl"
BEST_HYPERPARAMETERS_PATH = MODELS_DIR / "best_hyperparameters.json"

# Cấu hình chia tập và thực nghiệm
TARGET_COLUMN = "deposit"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
SCORING_METRIC = "f1_macro"

# Danh sách cột dữ liệu thô ban đầu
RAW_FEATURE_COLUMNS = (
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
)

# Cột gây rò rỉ thông tin trong bối cảnh phân loại trước cuộc gọi
LEAKAGE_COLUMNS = ("duration",)

# Lược đồ đặc trưng hợp lệ đưa vào mô hình trước cuộc gọi
MODEL_INPUT_COLUMNS = tuple(col for col in RAW_FEATURE_COLUMNS if col not in LEAKAGE_COLUMNS)

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
