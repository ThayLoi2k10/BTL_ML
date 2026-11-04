# BÀI TẬP LỚN MACHINE LEARNING
## Dự Đoán Khách Hàng Đăng Ký Tiền Gửi Tiết Kiệm Ngân Hàng (Bank Marketing Term Deposit Prediction)

> **Môn học:** Machine Learning  
> **Bộ dữ liệu:** Bank Marketing Dataset (11,162 dòng · 16 thuộc tính dự báo · Biến mục tiêu: `deposit` nhị phân `yes`/`no`)  
> **Bối cảnh nghiệp vụ:** Phân loại và chấm điểm tiềm năng khách hàng trước cuộc gọi (Pre-call scoring) nhằm tối ưu hóa chi phí telesale và nâng cao tỷ lệ chuyển đổi.

---

## 1. Thông Tin Thành Viên & Phân Công Vai Trò

| Thành viên | Phân công vai trò | Trách nhiệm chính trong dự án |
|:---|:---:|:---|
| **Phan Ngọc Xuân Lợi**<br>MSSV: `2411969` | **Person 1**<br>*(Leader)* | • Khảo sát phân tích dữ liệu khám phá (EDA).<br>• Xây dựng và tối ưu mô hình Tuyến tính (**Logistic Regression**).<br>• Đo kiểm định lượng hiệu năng trên tập Test & Phân tích chênh lệch tổng quát hóa (**Generalization Gap**).<br>• Xây dựng CLI suy luận (`src/predict.py`) và ứng dụng Web tương tác (`app.py`). |
| **Ngô Minh Trí**<br>MSSV: `2313598` | **Person 2** | • Kiểm định cấu trúc và chất lượng dữ liệu thô (**Data Quality & Inspection**).<br>• Xây dựng bộ biến đổi, mã hóa One-Hot và chuẩn hóa thang đo (**Encoding & Scaling**).<br>• Huấn luyện, đánh giá và tinh chỉnh mô hình Cây & Ensemble (**Random Forest**).<br>• Trực quan hóa chẩn đoán hiệu năng: Confusion Matrix, Đường cong ROC và Precision-Recall (`src/evaluation/plots.py`). |
| **Đỗ Phúc Danh**<br>MSSV: `2310409` | **Person 3** | • Phân tích phân bố đơn biến và đa biến của các trường dữ liệu (**Distributions**).<br>• Xây dựng và tối ưu siêu tham số mô hình Phi tuyến (**Support Vector Machine - SVM RBF**).<br>• Phân tích lỗi chuyên sâu (**Error Analysis**): nhận diện mẫu sát biên quyết định, vùng chồng lấn nhãn kNN và điểm mù cấu trúc.<br>• Soạn thảo kịch bản thuyết trình và slide báo cáo (`docs/Slides.pptx`, `docs/Kich_ban_thuyet_trinh.md`). |
| **Phan Ký Nhân**<br>MSSV: `2412432` | **Person 4** | • Phân tích tương quan đa biến và thiết lập cơ chế bảo vệ chống rò rỉ thông tin (**Data Leakage Guard** - loại bỏ cột `duration`).<br>• Thiết kế và tích hợp toàn diện **Preprocessing Pipeline** theo chuẩn scikit-learn (`src/pipeline.py`).<br>• Khảo sát đường cong học tập (**Learning Curves**) phân tích độ lệch - phương sai (Bias-Variance tradeoff).<br>• Đo kiểm Benchmark thời gian/bộ nhớ hệ thống và lập luận chọn lựa **Champion Model** (`src/evaluation/benchmark.py`). |

---

## 2. Cấu Trúc Cây Thư Mục Dự Án (Directory Structure)

```text
BTL_ML/
├── .dockerignore                     # Danh sách file/thư mục loại trừ khi đóng gói Docker image
├── .gitignore                        # Cấu hình bỏ qua tệp tạm, môi trường ảo và cache của Git
├── Dockerfile                        # Tệp định nghĩa container hóa môi trường trên nền Python 3.12-slim
├── README.md                         # Tài liệu kỹ thuật dự án và hướng dẫn tái hiện thực nghiệm
├── requirements.txt                  # Danh mục dependencies runtime cốt lõi được import trong src/
├── requirements-dev.txt              # Danh mục mở rộng cho kiểm thử (pytest, ipykernel, streamlit)
├── run_pipeline.py                   # Script điều phối toàn bộ quy trình: Preprocess -> Train -> Evaluate -> Predict
├── app.py                            # Ứng dụng web Streamlit hỗ trợ dự đoán đơn lẻ và hàng loạt (Batch CSV)
├── predictions.csv                   # Tệp kết quả dự đoán mặc định khi chạy CLI
│
├── data/                             # Quản lý toàn bộ dữ liệu dự án
│   ├── sample_test.csv               # Tệp CSV 10 mẫu dữ liệu thô phục vụ kiểm thử suy luận nhanh
│   ├── predictions_test.csv          # Kết quả dự đoán xuất ra từ sample_test.csv
│   ├── raw/                          # Dữ liệu gốc ban đầu
│   │   ├── bank.csv                  # Bộ dữ liệu gốc Bank Marketing chuẩn (11,162 bản ghi)
│   │   ├── dataset.csv               # Dữ liệu đối chiếu
│   │   └── archive.zip               # Tệp nén sao lưu dữ liệu gốc
│   └── processed/                    # Dữ liệu sạch sau tiền xử lý, lưu định dạng Apache Parquet
│       ├── X_train.parquet           # Tập đặc trưng huấn luyện (8,929 dòng × 32 cột)
│       ├── X_test.parquet            # Tập đặc trưng kiểm thử (2,233 dòng × 32 cột)
│       ├── y_train.parquet           # Nhãn mục tiêu tập huấn luyện (deposit: yes/no)
│       └── y_test.parquet            # Nhãn mục tiêu tập kiểm thử
│
├── docs/                             # Tài liệu trình chiếu và kịch bản bảo vệ đề tài
│   ├── Kich_ban_thuyet_trinh.md      # Kịch bản chi tiết từng phút cho 4 thành viên nhóm
│   ├── Slides.pptx                   # Bản trình chiếu PowerPoint chính thức
│   └── Slides.pdf                    # Bản xuất PDF của slide thuyết trình
│
├── figures/                          # Toàn bộ biểu đồ phân tích và chẩn đoán mô hình
│   ├── fig01_data_quality.png        # Phân tích tỷ lệ khuyết thiếu và bất thường
│   ├── fig02_distributions.png       # Phân bố tần suất các biến số lượng và phân loại
│   ├── fig03_correlations.png        # Ma trận tương quan và kiểm định rò rỉ dữ liệu
│   ├── fig04_feature_engineering.png # Đánh giá thông tin tương hỗ (Mutual Information)
│   ├── fig05_learning_curves.png     # Đường cong học tập chẩn đoán Bias - Variance
│   ├── fig06_confusion_matrices.png  # Ma trận nhầm lẫn đối chiếu trực quan 4 mô hình
│   ├── fig07_roc_curves.png          # Đường cong ROC và chỉ số AUC trên tập Test
│   └── fig08_precision_recall_curves.png # Đường cong Precision-Recall tối ưu điểm cắt kinh doanh
│
├── models/                           # Lưu trữ các artifact mô hình đã được huấn luyện và tối ưu
│   ├── best_hyperparameters.json     # Siêu tham số tối ưu tìm được qua 5-Fold Stratified CV
│   ├── preprocessing_pipeline.pkl    # Pipeline tiền xử lý và kỹ thuật đặc trưng đã fit trên Train
│   ├── best_linear_model.pkl         # Mô hình hồi quy Logistic tốt nhất
│   ├── best_tree_model.pkl           # Mô hình Random Forest tối ưu
│   ├── best_nonlinear_model.pkl      # Mô hình SVM RBF tối ưu
│   └── champion_model.pkl            # Mô hình được phê duyệt triển khai (Champion: Random Forest Tuned)
│
├── notebooks/                        # Jupyter Notebooks phục vụ EDA và nghiên cứu tuần tự
│   ├── 00_EDA_Final.ipynb            # Khảo sát dữ liệu khám phá tích hợp
│   ├── 01_data_quality.ipynb         # Kiểm định chất lượng dữ liệu thô (Trí)
│   ├── 02_distributions.ipynb        # Phân tích phân bố dữ liệu (Danh)
│   ├── 03_relationships.ipynb        # Tương quan và rò rỉ dữ liệu (Nhân)
│   ├── 04_features_pipeline.ipynb    # Kỹ thuật đặc trưng và pipeline (Nhân & Danh)
│   └── 06_evaluation.ipynb           # Đánh giá và so sánh mô hình (Lợi & Cả nhóm)
│
├── reports/                          # Báo cáo kỹ thuật chi tiết theo từng tuần và phân tích lỗi
│   ├── clean_environment_test_report.md # Biên bản kiểm thử môi trường sạch và tính tái hiện
│   ├── evaluation_benchmark.csv      # Bảng tổng hợp các chỉ số đánh giá 2 chiều (CV vs Test)
│   ├── test_predictions.csv          # Bảng hợp nhất dự đoán và xác suất của 4 mô hình trên tập Test
│   ├── test_predictions.parquet      # Bản lưu trữ parquet tốc độ cao của bảng dự đoán
│   ├── linear_coefficients.csv       # Trọng số hệ số của mô hình Logistic Regression
│   ├── tree_feature_importances.csv  # Mức độ quan trọng đặc trưng trích xuất từ Random Forest
│   ├── week4_champion_recommendation.json # Quyết định phê duyệt mô hình Champion
│   ├── week4_system_benchmark.json   # Đo kiểm hiệu năng bộ nhớ và thông lượng suy luận
│   ├── week4_tradeoff_matrix.json    # Ma trận đánh đổi kỹ thuật và nghiệp vụ
│   ├── week4_diagnostic_plots_Tri.md # Báo cáo chẩn đoán đồ họa
│   ├── week4_error_analysis_Danh.md  # Báo cáo phân tích chuyên sâu các ca dự đoán sai
│   ├── week4_metrics_and_evaluation_Loi.md # Báo cáo định lượng và chỉ số F1-Macro
│   └── error_analysis/               # Trích xuất dữ liệu chi tiết các ca ngoại lệ
│       ├── overlap_cases.csv         # Các mẫu rơi vào vùng không gian nhãn bị chồng lấn
│       └── borderline_cases.csv      # Các mẫu nằm sát ranh giới quyết định (p ~ 0.5)
│
├── src/                              # Mã nguồn chuẩn hóa của hệ thống Machine Learning
│   ├── __init__.py
│   ├── config.py                     # Quản lý tập trung đường dẫn, tên cột, seed và siêu tham số
│   ├── cleaning.py                   # Lớp xử lý missing values, làm sạch chuỗi và cắt tỉa ngoại lai
│   ├── encoding_scaling.py           # Phân chia dữ liệu phân tầng, One-Hot Encoding & RobustScaler
│   ├── feature_engineering.py        # Kỹ thuật sinh đặc trưng phi tuyến và chọn lọc Mutual Information
│   ├── pipeline.py                   # Đóng gói Pipeline tiền xử lý và lưu trữ Parquet splits
│   ├── predict.py                    # Script dòng lệnh (CLI) thực thi suy luận từ tệp CSV thô
│   ├── tuning.py                     # Framework tối ưu hóa siêu tham số dùng chung (GridSearchCV)
│   ├── evaluation/                   # Module đánh giá, chẩn đoán và phân tích lỗi
│   │   ├── __init__.py
│   │   ├── benchmark.py              # Đo thời gian fit, RAM peak, latency suy luận & chọn Champion
│   │   ├── error_analysis.py         # Mổ xẻ ma trận nhầm lẫn, phân tích kNN overlap và điểm mù
│   │   ├── metrics.py                # Tính toán Precision, Recall, F1-Macro, ROC-AUC, Brier score
│   │   └── plots.py                  # Xuất bản các hình vẽ chẩn đoán chất lượng cao
│   └── models/                       # Module xây dựng, huấn luyện và tuning 3 họ mô hình
│       ├── __init__.py
│       ├── linear_model.py           # Huấn luyện mô hình Logistic Regression (L1/L2, C-tuning)
│       ├── tree_model.py             # Huấn luyện mô hình Random Forest (n_estimators, max_depth)
│       └── nonlinear_model.py        # Huấn luyện mô hình Support Vector Machine RBF (C, gamma)
│
└── tests/                            # Bộ kiểm thử tự động (Unit Tests)
    ├── test_encoding_scaling.py      # Kiểm tra logic phân chia dữ liệu và cấu hình biến đổi
    ├── test_predict.py               # Kiểm tra tính toàn vẹn của luồng suy luận CLI
    └── test_tree_model.py            # Kiểm tra quy trình đánh giá và xuất báo cáo mô hình cây
```

---

## 3. Hướng Dẫn Cài Đặt Môi Trường

### Yêu cầu tiên quyết
* **Python:** Phiên bản `Python 3.12` (Khuyến nghị `3.12.3`, hỗ trợ tốt nhất cho các thư viện khoa học dữ liệu hiện đại).
* **Hệ điều hành:** Hỗ trợ đầy đủ Windows, macOS và Linux.

---

### Cách 1: Thiết lập với Virtualenv (`venv` - Khuyên Dùng)

Mở Terminal / PowerShell tại thư mục gốc của dự án (`BTL_ML`):

```powershell
# 1. Khởi tạo môi trường ảo Python 3.12 cô lập
python -m venv venv

# 2. Kích hoạt môi trường ảo
# Trên Windows PowerShell:
.\venv\Scripts\activate
# Trên macOS / Linux:
source venv/bin/activate

# 3. Cài đặt toàn bộ dependencies runtime cốt lõi
pip install -r requirements.txt

# 4. (Tùy chọn) Cài đặt bổ sung công cụ kiểm thử hoặc phát triển
pip install -r requirements-dev.txt
```

---

### Cách 2: Thiết lập với Conda

```bash
# 1. Tạo môi trường Conda mới với Python 3.12
conda create -n btl_ml python=3.12 -y

# 2. Kích hoạt môi trường
conda activate btl_ml

# 3. Cài đặt dependencies qua pip
pip install -r requirements.txt
```

---

### Cách 3: Đóng gói và Chạy bằng Docker (Tùy Chọn)

Dự án đã trang bị sẵn `Dockerfile` chuẩn trên nền `python:3.12-slim`:

```bash
# 1. Xây dựng Docker Image
docker build -t btl_ml_app .

# 2. Khởi chạy kiểm thử suy luận tự động trong Container
docker run --rm btl_ml_app
```

---

## 4. Hướng Dẫn Tái Hiện Thực Nghiệm (Reproducibility Commands)

Sau khi đã kích hoạt môi trường ảo, bạn có thể thực thi chính xác các cú pháp dòng lệnh dưới đây để tái hiện toàn bộ các kết quả trong báo cáo:

### 1. Tái hiện Toàn Bộ Pipeline Tiền Xử Lý (Preprocessing Stage)
Đọc dữ liệu thô `data/raw/bank.csv`, loại bỏ rò rỉ thông tin `duration`, tạo 32 đặc trưng chuẩn hóa và lưu trữ các phân tách Parquet vào `data/processed/`:
```bash
python run_pipeline.py --mode preprocess
```

---

### 2. Huấn Luyện & Tối Ưu Siêu Tham Số 3 Họ Mô Hình (Training Stage)
Thực thi kiểm định chéo 5-Fold Stratified CV, tìm kiếm tham số tối ưu và lưu mô hình:
```bash
python run_pipeline.py --mode train
```
> *Mẹo:* Bạn cũng có thể huấn luyện riêng từng mô hình theo ý muốn:
> - Mô hình Tuyến tính: `python src/models/linear_model.py`
> - Mô hình Cây Random Forest: `python src/models/tree_model.py`
> - Mô hình Phi tuyến SVM: `python src/models/nonlinear_model.py`

---

### 3. Đánh Giá Toàn Diện, Xuất Biểu Đồ & Phân Tích Lỗi (Evaluation Stage)
Tính toán chỉ số kiểm thử độc lập, sinh ma trận đối chiếu CV vs Test, xuất 3 đồ thị chẩn đoán vào `figures/`, thực thi kNN Error Analysis và đo benchmark hệ thống:
```bash
python run_pipeline.py --mode evaluate
```

---

### 4. Chạy Suy Luận Dòng Lệnh với `src/predict.py` (CLI Inference)
Dự đoán nhãn và xác suất cho một tệp CSV khách hàng mới thông qua mô hình Champion:
```bash
python src/predict.py --input data/sample_test.csv --output data/predictions_test.csv
```
> *Tùy chọn nâng cao:*
> ```bash
> python src/predict.py --input <đường_dẫn_csv_thô> --output <đường_dẫn_lưu_kết_quả> --model models/champion_model.pkl
> ```

---

### 5. Chạy Toàn Bộ Quy Trình Từ Đầu Đến Cuối (End-to-End Pipeline)
Thực thi tự động lần lượt 4 giai đoạn: Tiền xử lý ➔ Huấn luyện ➔ Đánh giá ➔ Suy luận:
```bash
python run_pipeline.py --mode all
```

---

### 6. Chạy Bộ Kiểm Thử Tự Động Hóa (Unit Tests)
Đảm bảo 100% tính đúng đắn logic của các hàm tiền xử lý, cấu trúc cây và luồng suy luận CLI:
```bash
pytest tests/
```

---

### 7. Khởi Chạy Ứng Dụng Web Giao Diện Tương Tác (Streamlit Web App)
Khởi chạy bảng điều khiển trực quan hỗ trợ nhập thông tin khách hàng đơn lẻ hoặc tải lên tệp CSV hàng loạt:
```bash
streamlit run app.py
```

---

## 5. Dữ Liệu & Artifacts Mô Hình

### Dữ liệu dự án
* **Dữ liệu gốc (Raw Data):** Dữ liệu chuẩn của bài toán là tập `bank.csv` (11,162 dòng) trích xuất từ chiến dịch tiếp thị ngân hàng Bồ Đào Nha.
  * Tệp đã được lưu sẵn trong thư mục `data/raw/bank.csv` của mã nguồn Git.
  * Liên kết nguồn dữ liệu công khai:
    - [UCI Machine Learning Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
    - [Kaggle Bank Marketing Dataset (janiobachmann)](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)
* **Dữ liệu mẫu kiểm thử (Sample Test):** `data/sample_test.csv` (10 bản ghi dữ liệu thô đại diện) sẵn sàng cho việc kiểm thử suy luận nhanh.

### Artifacts Mô hình
* **Pipeline tiền xử lý:** `models/preprocessing_pipeline.pkl` (~92 KB) được tích hợp trực tiếp trong repo.
* **Mô hình Champion:** `models/champion_model.pkl` (~41.5 MB - Random Forest đã tối ưu 200 cây quyết định, max_depth=10, min_samples_split=5) đã được lưu sẵn trong thư mục `models/` của Git.
* Do dung lượng toàn bộ repo nhỏ hơn giới hạn 100 MB của GitHub, toàn bộ artifacts đã được track đầy đủ và có thể tái hiện ngay lập tức mà không cần tải thêm từ bên ngoài.

---

## 6. Kết Quả Kiểm Thử Môi Trường Sạch (Clean Environment Testing)

Dự án đã trải qua kiểm thử độc lập trên môi trường ảo sạch hoàn toàn (`.clean_test_env`), xác nhận:
* **Không thiếu thư viện:** Tất cả các module hoạt động mượt mà chỉ với các gói được định nghĩa trong `requirements.txt`.
* **Không sai lệch đường dẫn:** Toàn bộ đường dẫn được tự động phân giải qua `pathlib.Path` tại `src/config.py`.
* **Bảo toàn kết quả:** 14/14 bài kiểm tra tự động trong `tests/` đều vượt qua (`passed`).
* Chi tiết kiểm thử và log chi tiết từng bước xem tại: [Biên bản kiểm thử môi trường sạch](file:///reports/clean_environment_test_report.md).
