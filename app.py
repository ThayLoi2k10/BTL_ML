"""Streamlit Web Application for Bank Marketing Deposit Prediction.

Author: Lợi (Person 1 - Deploy & Inference)
Usage:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PIPELINE_PATH = PROJECT_ROOT / "models" / "preprocessing_pipeline.pkl"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "champion_model.pkl"
DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "data" / "sample_test.csv"


def ensure_unpickle_compatibility() -> None:
    """Ensure compatibility across scikit-learn versions during unpickling."""
    import sklearn.compose._column_transformer as ct

    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        ct._RemainderColsList = _RemainderColsList


@st.cache_resource(show_spinner="Đang nạp mô hình và pipeline tiền xử lý...")
def load_artifacts() -> tuple[Any, Any]:
    """Load fitted preprocessing pipeline and champion model once."""
    ensure_unpickle_compatibility()
    if not DEFAULT_PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy pipeline tại {DEFAULT_PIPELINE_PATH}")
    if not DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình tại {DEFAULT_MODEL_PATH}")

    pipeline = joblib.load(DEFAULT_PIPELINE_PATH)
    model = joblib.load(DEFAULT_MODEL_PATH)
    return pipeline, model


def predict_dataframe(df_input: pd.DataFrame, pipeline: Any, model: Any) -> pd.DataFrame:
    """Run preprocessing and inference on a DataFrame."""
    df_raw = df_input.copy()
    actual_labels = None
    if "deposit" in df_raw.columns:
        actual_labels = df_raw["deposit"].copy()
        features = df_raw.drop(columns=["deposit"])
    else:
        features = df_raw

    X_transformed = pipeline.transform(features)
    preds = model.predict(X_transformed)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_transformed)
        classes = list(getattr(model, "classes_", ["no", "yes"]))
        pos_idx = classes.index("yes") if "yes" in classes else 1
        prob_yes = probs[:, pos_idx]
        confidence = np.max(probs, axis=1)
    else:
        decisions = model.decision_function(X_transformed)
        prob_yes = 1.0 / (1.0 + np.exp(-decisions))
        confidence = np.maximum(prob_yes, 1.0 - prob_yes)

    result_df = df_raw.copy()
    result_df["predicted_deposit"] = preds
    result_df["probability_yes"] = np.round(prob_yes, 4)
    result_df["confidence_score"] = np.round(confidence, 4)
    if actual_labels is not None:
        result_df["actual_deposit"] = actual_labels
    return result_df


def main() -> None:
    st.set_page_config(
        page_title="Dự Đoán Gửi Tiết Kiệm Ngân Hàng",
        page_icon="🏦",
        layout="wide",
    )

    st.title("🏦 Hệ Thống Dự Đoán Khách Hàng Gửi Tiết Kiệm (Bank Marketing)")
    st.markdown(
        """
        Ứng dụng hỗ trợ đội ngũ tư vấn tài chính phân loại khách hàng tiềm năng tham gia **Tiền gửi có kỳ hạn**.
        Hệ thống sử dụng **Mô hình Vô địch (Random Forest Tuned)** kết hợp **Pipeline Tiền xử lý tự động**.
        """
    )

    try:
        pipeline, model = load_artifacts()
    except Exception as e:
        st.error(f"Lỗi khi nạp mô hình: {e}")
        st.stop()

    tab1, tab2 = st.tabs(["📋 Dự đoán Khách hàng Đơn lẻ", "📁 Dự đoán Theo lô (Batch CSV)"])

    # ---------------- TAB 1: SINGLE PREDICTION ----------------
    with tab1:
        st.subheader("Nhập thông tin khách hàng")
        with st.form("single_customer_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Thông tin định danh & Nhân khẩu:**")
                age = st.number_input("Tuổi (age)", min_value=18, max_value=100, value=35)
                job = st.selectbox(
                    "Nghề nghiệp (job)",
                    [
                        "admin.", "blue-collar", "entrepreneur", "housemaid",
                        "management", "retired", "self-employed", "services",
                        "student", "technician", "unemployed", "unknown"
                    ],
                    index=0,
                )
                marital = st.selectbox("Tình trạng hôn nhân (marital)", ["married", "single", "divorced"], index=1)
                education = st.selectbox("Trình độ học vấn (education)", ["primary", "secondary", "tertiary", "unknown"], index=2)

            with col2:
                st.markdown("**Tài chính & Tín dụng:**")
                default = st.selectbox("Tín dụng xấu (default)", ["no", "yes"], index=0)
                balance = st.number_input("Số dư tài khoản trung bình hàng năm (balance - EUR)", value=1500)
                housing = st.selectbox("Có khoản vay mua nhà? (housing)", ["yes", "no"], index=1)
                loan = st.selectbox("Có khoản vay cá nhân tiêu dùng? (loan)", ["no", "yes"], index=0)

            with col3:
                st.markdown("**Lịch sử tương tác & Chiến dịch:**")
                contact = st.selectbox("Hình thức liên hệ (contact)", ["cellular", "telephone", "unknown"], index=0)
                month = st.selectbox("Tháng liên hệ cuối (month)", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], index=7)
                day = st.number_input("Ngày trong tháng (day)", min_value=1, max_value=31, value=15)
                campaign = st.number_input("Số lần liên hệ trong chiến dịch này (campaign)", min_value=1, max_value=50, value=1)
                pdays = st.number_input("Số ngày từ lần liên hệ chiến dịch trước (pdays, -1 nếu chưa từng)", value=-1)
                previous = st.number_input("Số lần liên hệ trước chiến dịch này (previous)", min_value=0, max_value=50, value=0)
                poutcome = st.selectbox("Kết quả chiến dịch trước (poutcome)", ["unknown", "failure", "other", "success"], index=0)
                duration = st.number_input("Thời lượng cuộc gọi gần nhất (duration - giây)", min_value=0, max_value=5000, value=180, help="Được kiểm soát và loại trừ rò rỉ (leakage-free)")

            submit_btn = st.form_submit_button("🚀 Dự Đoán Ngay", use_container_width=True)

        if submit_btn:
            single_dict = {
                "age": [age],
                "job": [job],
                "marital": [marital],
                "education": [education],
                "default": [default],
                "balance": [balance],
                "housing": [housing],
                "loan": [loan],
                "contact": [contact],
                "day": [day],
                "month": [month],
                "duration": [duration],
                "campaign": [campaign],
                "pdays": [pdays],
                "previous": [previous],
                "poutcome": [poutcome],
            }
            df_single = pd.DataFrame(single_dict)

            with st.spinner("Đang tính toán suy luận..."):
                res = predict_dataframe(df_single, pipeline, model)
                pred_label = res["predicted_deposit"].iloc[0]
                prob_yes = res["probability_yes"].iloc[0]
                confidence = res["confidence_score"].iloc[0]

            st.markdown("---")
            st.subheader("Kết Quả Dự Đoán:")
            r_col1, r_col2, r_col3 = st.columns(3)

            with r_col1:
                if pred_label == "yes":
                    st.success("### 🎯 Dự đoán: SẼ ĐĂNG KÝ (YES)")
                else:
                    st.error("### 🛑 Dự đoán: TỪ CHỐI (NO)")

            with r_col2:
                st.metric("Xác suất Đăng ký (P_yes)", f"{prob_yes:.2%}")
                st.progress(float(prob_yes))

            with r_col3:
                st.metric("Độ tin cậy của mô hình", f"{confidence:.2%}")

            if pred_label == "yes":
                st.info("💡 **Khuyến nghị nghiệp vụ:** Khách hàng có xác suất chuyển đổi cao. Ưu tiên nhân viên tư vấn gọi điện chăm sóc và giới thiệu các gói lãi suất ưu đãi!")
            else:
                st.warning("💡 **Khuyến nghị nghiệp vụ:** Xác suất tham gia thấp. Cân nhắc tiếp cận qua kênh tiếp thị số chi phí thấp (Email/SMS) thay vì tốn chi phí gọi telesale trực tiếp.")

    # ---------------- TAB 2: BATCH PREDICTION ----------------
    with tab2:
        st.subheader("Tải lên file CSV chứa danh sách khách hàng")
        st.markdown(
            "Tải lên file CSV thô với định dạng giống `data/sample_test.csv` để nhận kết quả phân loại hàng loạt."
        )

        uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])
        use_sample = st.checkbox("Hoặc sử dụng dữ liệu mẫu có sẵn (data/sample_test.csv)", value=False)

        batch_df = None
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
        elif use_sample and DEFAULT_SAMPLE_PATH.exists():
            batch_df = pd.read_csv(DEFAULT_SAMPLE_PATH)

        if batch_df is not None:
            st.write(f"Đã nạp **{len(batch_df)}** bản ghi:")
            st.dataframe(batch_df.head(5), use_container_width=True)

            if st.button("⚡ Thực thi dự đoán hàng loạt", use_container_width=True):
                with st.spinner("Đang xử lý dữ liệu qua pipeline và suy luận..."):
                    res_batch = predict_dataframe(batch_df, pipeline, model)

                st.success("Hoàn tất dự đoán!")

                c1, c2, c3 = st.columns(3)
                yes_count = int((res_batch["predicted_deposit"] == "yes").sum())
                no_count = int((res_batch["predicted_deposit"] == "no").sum())
                c1.metric("Tổng số khách hàng", len(res_batch))
                c2.metric("Số lượng Tiềm năng (YES)", yes_count, delta=f"{yes_count/len(res_batch):.1%}")
                c3.metric("Số lượng Từ chối (NO)", no_count, delta=f"{no_count/len(res_batch):.1%}", delta_color="inverse")

                st.subheader("Bảng Kết Quả Chi Tiết:")
                display_cols = [
                    c for c in [
                        "age", "job", "balance", "predicted_deposit",
                        "probability_yes", "confidence_score", "actual_deposit"
                    ] if c in res_batch.columns
                ]
                st.dataframe(res_batch[display_cols], use_container_width=True)

                csv_data = res_batch.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Tải xuống toàn bộ kết quả (CSV)",
                    data=csv_data,
                    file_name="predictions_batch.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
