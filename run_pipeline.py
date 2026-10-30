"""Kịch bản điều phối thực thi toàn bộ quy trình Machine Learning từ dữ liệu thô đến triển khai."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Đảm bảo đường dẫn gốc project luôn nằm trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_PROCESSED_DIR,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)
from src.encoding_scaling import split_data
from src.evaluation.benchmark import run_week4_nhan
from src.evaluation.error_analysis import run_error_analysis
from src.evaluation.metrics import evaluate_all_models
from src.evaluation.plots import generate_all_diagnostic_plots
from src.models import linear_model, nonlinear_model, tree_model
from src.pipeline import (
    build_preprocessing_pipeline,
    load_correct_dataset,
    save_preprocessing_artifact,
    save_processed_splits,
    validate_transformed_splits,
)
from src.predict import main as predict_main


def run_preprocessing_stage() -> None:
    """Thực thi toàn bộ luồng tiền xử lý, phân chia dữ liệu và lưu trữ các tập parquet."""
    print("\n" + "=" * 70)
    print(" GIAI ĐOẠN 1: TIỀN XỬ LÝ DỮ LIỆU & BẢO VỆ CHỐNG RÒ RỈ THÔNG TIN")
    print("=" * 70)
    raw_df = load_correct_dataset(RAW_DATA_PATH)
    print(f"[*] Đã nạp tập dữ liệu gốc: {raw_df.shape}")

    X_train, X_test, y_train, y_test = split_data(raw_df, target_column=TARGET_COLUMN)
    print(f"[*] Phân chia tập dữ liệu: Train={X_train.shape}, Test={X_test.shape}")

    pipeline = build_preprocessing_pipeline()
    print("[*] Đang fit pipeline tiền xử lý trên tập Train...")
    X_train_trans = pipeline.fit_transform(X_train, y_train)
    X_test_trans = pipeline.transform(X_test)

    validate_transformed_splits(X_train_trans, X_test_trans)
    save_preprocessing_artifact(pipeline)
    save_processed_splits(X_train_trans, X_test_trans, y_train, y_test, directory=DATA_PROCESSED_DIR)
    print("[+] Hoàn thành giai đoạn tiền xử lý và lưu trữ artifact.")


def run_training_stage() -> None:
    """Thực thi huấn luyện và tối ưu siêu tham số độc lập cho 3 họ thuật toán."""
    print("\n" + "=" * 70)
    print(" GIAI ĐOẠN 2: HUẤN LUYỆN & TỐI ƯU SIÊU THAM SỐ CÁC MÔ HÌNH ỨNG VIÊN")
    print("=" * 70)

    print("\n--- 1/3: Mô hình Tuyến tính (Logistic Regression) ---")
    linear_model.main()

    print("\n--- 2/3: Mô hình Cây & Ensemble (Random Forest) ---")
    tree_model.main()

    print("\n--- 3/3: Mô hình Phi tuyến (Support Vector Machine) ---")
    nonlinear_model.main()

    print("[+] Hoàn thành giai đoạn huấn luyện và tối ưu hóa 3 mô hình.")


def run_evaluation_stage() -> None:
    """Thực thi đánh giá định lượng tập Test, trực quan hóa, phân tích lỗi và chốt Champion."""
    print("\n" + "=" * 70)
    print(" GIAI ĐOẠN 3: ĐÁNH GIÁ ĐỊNH LƯỢNG, TRỰC QUAN HÓA & LỰA CHỌN CHAMPION")
    print("=" * 70)

    print("\n[*] 1. Tính toán metric định lượng và Generalization Gap...")
    evaluate_all_models()

    print("\n[*] 2. Xuất bản các đồ thị chẩn đoán (Confusion Matrix, ROC, PR)...")
    generate_all_diagnostic_plots(save_vector=False)

    print("\n[*] 3. Thực thi phân tích lỗi sâu (Error Analysis)...")
    run_error_analysis()

    print("\n[*] 4. Đo kiểm benchmark hệ thống và sao chép Champion Model...")
    run_week4_nhan()

    print("[+] Hoàn thành giai đoạn đánh giá toàn diện.")


def run_inference_stage() -> None:
    """Thực thi kiểm thử suy luận dòng lệnh trên tập dữ liệu mẫu."""
    print("\n" + "=" * 70)
    print(" GIAI ĐOẠN 4: KIỂM THỬ SUY LUẬN DÒNG LỆNH (CLI INFERENCE)")
    print("=" * 70)
    predict_main([])


def parse_args() -> argparse.Namespace:
    """Thiết lập và phân tích đối số lựa chọn chế độ thực thi kịch bản."""
    parser = argparse.ArgumentParser(
        description="Kịch bản điều phối quy trình Machine Learning đầu cuối.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["preprocess", "train", "evaluate", "predict", "all"],
        default="evaluate",
        help="Lựa chọn giai đoạn thực thi trong quy trình.",
    )
    return parser.parse_args()


def main() -> None:
    """Điều phối thực thi các giai đoạn dựa trên tham số dòng lệnh."""
    args = parse_args()

    if args.mode == "preprocess":
        run_preprocessing_stage()
    elif args.mode == "train":
        run_training_stage()
    elif args.mode == "evaluate":
        run_evaluation_stage()
    elif args.mode == "predict":
        run_inference_stage()
    elif args.mode == "all":
        run_preprocessing_stage()
        run_training_stage()
        run_evaluation_stage()
        run_inference_stage()


if __name__ == "__main__":
    main()
