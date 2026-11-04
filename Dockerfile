# ==============================================================================
# Dockerfile: Môi trường chuẩn hóa cho Dự án Machine Learning (BTL_ML)
# ==============================================================================
# Lưu ý: Thư viện pandas==3.0.5 và numpy==2.5.3 yêu cầu Python >= 3.11.
# Do đó, base image python:3.12-slim được sử dụng để tương thích 100% với requirements.txt.
# ==============================================================================

FROM python:3.12-slim

# Thiết lập biến môi trường tối ưu cho container Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Cài đặt các gói hệ thống tối thiểu nếu cần
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Sao chép và cài đặt danh sách dependencies trước (tận dụng Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn, cấu hình, mô hình và dữ liệu mẫu
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY reports/ ./reports/
COPY figures/ ./figures/
COPY run_pipeline.py .
COPY app.py .

# Tạo thư mục output nếu chưa tồn tại
RUN mkdir -p data

# Thiết lập entrypoint kiểm thử tự động suy luận mô hình Champion với tập mẫu
ENTRYPOINT ["python", "src/predict.py"]
CMD ["--input", "data/sample_test.csv", "--output", "data/predictions_test.csv"]

