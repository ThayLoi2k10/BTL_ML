"""Kịch bản suy luận dòng lệnh (CLI) phân loại khả năng đăng ký tiền gửi ngân hàng."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CHAMPION_MODEL_PATH,
    PREPROCESSING_PIPELINE_PATH,
    SAMPLE_TEST_PATH,
)

warnings.filterwarnings("ignore", category=UserWarning)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PIPELINE_PATH = PREPROCESSING_PIPELINE_PATH
DEFAULT_MODEL_PATH = CHAMPION_MODEL_PATH
DEFAULT_INPUT_PATH = SAMPLE_TEST_PATH
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "predictions.csv"


def ensure_unpickle_compatibility() -> None:
    """Đảm bảo tính tương thích khi unpickle mô hình giữa các phiên bản scikit-learn."""
    import sklearn.compose._column_transformer as ct

    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        ct._RemainderColsList = _RemainderColsList


def load_artifacts(
    pipeline_path: Path | str = DEFAULT_PIPELINE_PATH,
    model_path: Path | str = DEFAULT_MODEL_PATH,
) -> tuple[Any, Any]:
    """Nạp pipeline tiền xử lý và mô hình champion từ tệp nhị phân."""
    ensure_unpickle_compatibility()
    pipeline_path = Path(pipeline_path)
    model_path = Path(model_path)

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Không tìm thấy pipeline tiền xử lý tại: '{pipeline_path}'.")
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình champion tại: '{model_path}'.")

    pipeline = joblib.load(pipeline_path)
    model = joblib.load(model_path)
    return pipeline, model


def predict(
    input_data: pd.DataFrame,
    pipeline: Any,
    model: Any,
    pos_label: str = "yes",
) -> pd.DataFrame:
    """Chuyển hóa dữ liệu thô qua pipeline và sinh nhãn dự đoán kèm điểm độ tin cậy."""
    df_raw = input_data.copy()

    actual_labels = None
    if "deposit" in df_raw.columns:
        actual_labels = df_raw["deposit"].copy()
        features_to_transform = df_raw.drop(columns=["deposit"])
    else:
        features_to_transform = df_raw

    X_transformed = pipeline.transform(features_to_transform)
    predictions = model.predict(X_transformed)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_transformed)
        classes = list(getattr(model, "classes_", ["no", "yes"]))
        pos_idx = classes.index(pos_label) if pos_label in classes else 1
        neg_idx = 1 - pos_idx

        prob_pos = probabilities[:, pos_idx]
        prob_neg = probabilities[:, neg_idx]
        confidence = np.max(probabilities, axis=1)
    elif hasattr(model, "decision_function"):
        decisions = model.decision_function(X_transformed)
        prob_pos = 1.0 / (1.0 + np.exp(-decisions))
        prob_neg = 1.0 - prob_pos
        confidence = np.maximum(prob_pos, prob_neg)
    else:
        prob_pos = (pd.Series(predictions) == pos_label).astype(float).to_numpy()
        prob_neg = 1.0 - prob_pos
        confidence = np.ones(len(predictions), dtype=float)

    result_df = df_raw.copy()
    result_df["predicted_deposit"] = predictions
    result_df["confidence_score"] = np.round(confidence, 4)
    result_df["probability_yes"] = np.round(prob_pos, 4)
    result_df["probability_no"] = np.round(prob_neg, 4)

    if actual_labels is not None:
        result_df["actual_deposit"] = actual_labels
        result_df["is_correct"] = result_df["predicted_deposit"] == actual_labels

    return result_df


def parse_arguments(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Thiết lập và phân tích các đối số dòng lệnh CLI đầu vào."""
    parser = argparse.ArgumentParser(
        description="Thực thi dự đoán hàng loạt bằng mô hình Champion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Đường dẫn đến file CSV chứa dữ liệu khách hàng cần dự đoán.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Đường dẫn lưu tệp kết quả dự đoán CSV.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default=str(DEFAULT_PIPELINE_PATH),
        help="Đường dẫn đến pipeline tiền xử lý.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Đường dẫn đến mô hình champion.",
    )
    return parser.parse_args(args)


def main(argv: Sequence[str] | None = None) -> None:
    """Thực thi toàn bộ luồng dự đoán dòng lệnh hàng loạt từ tệp CSV."""
    args = parse_arguments(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)
    pipeline_path = Path(args.pipeline)
    model_path = Path(args.model)

    print("=" * 70)
    print(" HỆ THỐNG DỰ ĐOÁN TIỀN GỬI NGÂN HÀNG - DÒNG LỆNH (BATCH CLI)")
    print("=" * 70)
    print(f"[*] Dữ liệu đầu vào : {input_path}")
    print(f"[*] Tệp kết quả lưu  : {output_path}")
    print(f"[*] Pipeline artifact: {pipeline_path}")
    print(f"[*] Mô hình champion : {model_path}")
    print("-" * 70)

    if not input_path.exists():
        print(f"[!] Lỗi: Tệp đầu vào '{input_path}' không tồn tại.")
        sys.exit(1)

    print("[*] Đang nạp dữ liệu đầu vào...")
    input_df = pd.read_csv(input_path)
    print(f"    Đã nạp {len(input_df)} bản ghi.")

    print("[*] Đang nạp mô hình và pipeline...")
    pipeline, model = load_artifacts(pipeline_path, model_path)

    print("[*] Đang thực thi suy luận dự đoán...")
    results_df = predict(input_df, pipeline, model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"[+] Đã lưu kết quả thành công tại: {output_path}")
    print("-" * 70)

    preview_cols = [
        col
        for col in [
            "age",
            "job",
            "marital",
            "balance",
            "predicted_deposit",
            "confidence_score",
            "probability_yes",
            "actual_deposit",
            "is_correct",
        ]
        if col in results_df.columns
    ]

    print("[*] XEM TRƯỚC KẾT QUẢ DỰ ĐOÁN:")
    print(results_df[preview_cols].to_string(index=True))
    print("-" * 70)

    counts = results_df["predicted_deposit"].value_counts().to_dict()
    print(f"[*] Thống kê kết quả : {counts}")
    mean_conf = results_df["confidence_score"].mean()
    print(f"[*] Độ tin cậy TB     : {mean_conf:.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
