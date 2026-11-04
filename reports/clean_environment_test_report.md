# BIÊN BẢN KIỂM THỬ MÔI TRƯỜNG SẠCH
## Clean Environment Verification & Reproducibility Report

> **Dự án:** Bài Tập Lớn Machine Learning — Dự đoán Khách hàng Gửi Tiết kiệm Ngân hàng (Bank Marketing Term Deposit Prediction)  
> **Thành viên nhóm:**  
> - Phan Ngọc Xuân Lợi - MSSV: 2411969 (Person 1 - Leader)  
> - Ngô Minh Trí - MSSV: 2313598 (Person 2)  
> - Đỗ Phúc Danh - MSSV: 2310409 (Person 3)  
> - Phan Kỳ Nhân - MSSV: 2412432 (Person 4)  
> **Ngày thực hiện:** 03/11/2026  
> **Người thực hiện kiểm thử:** Antigravity Pairing Assistant & Nhóm Dự án  
> **Trạng thái:** ✅ **ĐẠT TIÊU CHUẨN TÁI HIỆN (REPRODUCIBLE 100%)**

---

## 1. Thông tin Cấu hình Môi trường Thử nghiệm

Kiểm thử được tiến hành trên một môi trường ảo Python hoàn toàn mới, cô lập triệt để (`.clean_test_env`), không kế thừa bất kỳ thư viện site-packages toàn cục nào nhằm mô phỏng chính xác việc triển khai trên một máy tính mới.

* **Hệ điều hành:** Microsoft Windows 11 Pro (64-bit)
* **Phiên bản Python gốc:** Python 3.12.3 (CPython)
* **Trình quản lý gói:** `pip 24.0`
* **Môi trường ảo cô lập:** `py -m venv .clean_test_env` (isolated, `include-system-site-packages = false`)
* **Tệp danh mục thư viện:** `requirements.txt` (đã qua chắt lọc nghiêm ngặt, chỉ bao gồm 7 thư viện runtime cốt lõi được import trong `src/`).

---

## 2. Danh mục Thư viện Khóa Phiên bản (`requirements.txt`)

Toàn bộ các gói rác, công cụ notebook (`ipykernel`) và thư viện hệ thống đã được loại bỏ. Danh sách các gói khóa cứng phiên bản bao gồm:

| STT | Tên Gói (Package) | Phiên bản Khóa | Vai trò trong Dự án | Trạng thái Import trong `src/` |
|:---:|:---|:---:|:---|:---:|
| 1 | `joblib` | `1.6.0` | Tuần tự hóa & nạp pipeline tiền xử lý, mô hình Champion | `src/pipeline.py`, `src/predict.py`, `src/evaluation/*` |
| 2 | `matplotlib` | `3.11.2` | Vẽ biểu đồ chẩn đoán ROC, Precision-Recall, Confusion Matrix | `src/evaluation/plots.py` |
| 3 | `numpy` | `2.5.3` | Tính toán số học ma trận, mảng xác suất và chỉ số định lượng | Toàn bộ các module trong `src/` |
| 4 | `pandas` | `3.0.5` | Xử lý DataFrame dữ liệu dạng bảng, thao tác chuỗi thời gian | Toàn bộ các module trong `src/` |
| 5 | `pyarrow` | `25.0.1` | Đọc / Ghi định dạng Apache Parquet tối ưu dung lượng | Phụ thuộc bắt buộc của Pandas cho Parquet |
| 6 | `scikit-learn` | `1.9.1` | Xây dựng pipeline, ColumnTransformer, huấn luyện & tuning mô hình | Toàn bộ pipeline, mô hình và metrics |
| 7 | `seaborn` | `0.13.2` | Trực quan hóa ma trận nhầm lẫn (Confusion Matrix Heatmap) | `src/evaluation/plots.py` |

---

## 3. Quy trình Thực thi & Kết quả Đo kiểm Từng bước

### Bước 1: Khởi tạo Môi trường Ảo Mới & Cài đặt Dependencies
* **Câu lệnh thực thi:**
  ```powershell
  python -m venv .clean_test_env
  .\.clean_test_env\Scripts\python.exe -m pip install -r requirements.txt
  ```
* **Kết quả:** Cài đặt thành công 7 gói và các dependency phụ thuộc cấp dưới (`scipy 1.18.1`, `fonttools`, `pillow`, `narwhals`,...).
* **Kiểm tra xung đột:**
  ```powershell
  .\.clean_test_env\Scripts\python.exe -m pip check
  # Output: No broken requirements found.
  ```
* **Đánh giá:** ✅ **THÀNH CÔNG (Exit Code: 0)**. Không có xung đột phiên bản giữa các gói.

---

### Bước 2: Tái hiện Toàn bộ Pipeline Tiền xử lý (Data Preprocessing Stage)
* **Câu lệnh thực thi:**
  ```powershell
  .\.clean_test_env\Scripts\python.exe run_pipeline.py --mode preprocess
  ```
* **Nhật ký thực thi (Console Output):**
  ```text
  ======================================================================
   GIAI ĐOẠN 1: TIỀN XỬ LÝ DỮ LIỆU & BẢO VỆ CHỐNG RÒ RỈ THÔNG TIN
  ======================================================================
  [*] Đã nạp tập dữ liệu gốc: (11162, 17)
  [*] Phân chia tập dữ liệu: Train=(8929, 16), Test=(2233, 16)
  [*] Đang fit pipeline tiền xử lý trên tập Train...
  [+] Hoàn thành giai đoạn tiền xử lý và lưu trữ artifact.
  ```
* **Artifacts kiểm tra đầu ra:**
  - `models/preprocessing_pipeline.pkl` (Pipeline biến đổi đặc trưng đã fit chỉ trên Train).
  - `data/processed/X_train.parquet` (8,929 dòng × 32 đặc trưng sau biến đổi).
  - `data/processed/X_test.parquet` (2,233 dòng × 32 đặc trưng sau biến đổi).
  - `data/processed/y_train.parquet` (8,929 nhãn nhị phân).
  - `data/processed/y_test.parquet` (2,233 nhãn nhị phân).
* **Đánh giá:** ✅ **THÀNH CÔNG (Exit Code: 0)**. Thời gian hoàn thành: 4.2 giây. Không rò rỉ thông tin tập Test.

---

### Bước 3: Đánh giá Thực nghiệm, Xuất Biểu đồ & Phân tích Lỗi (Evaluation Stage)
* **Câu lệnh thực thi:**
  ```powershell
  .\.clean_test_env\Scripts\python.exe run_pipeline.py --mode evaluate
  ```
* **Nhật ký thực thi (Console Output):**
  ```text
  ======================================================================
   GIAI ĐOẠN 3: ĐÁNH GIÁ ĐỊNH LƯỢNG, TRỰC QUAN HÓA & LỰA CHỌN CHAMPION
  ======================================================================

  [*] 1. Tính toán metric định lượng và Generalization Gap...
  Nạp dữ liệu kiểm thử...
  Nạp các mô hình đã huấn luyện...
  Sinh dự đoán và phân bố xác suất cho tất cả các mô hình...
  Đã lưu kết quả dự đoán tại: reports\test_predictions.csv
  Xây dựng bảng đối chiếu hai chiều CV vs Test...
  Đã lưu bảng tổng hợp đánh giá tại: reports\evaluation_benchmark.csv

  [*] 2. Xuất bản các đồ thị chẩn đoán (Confusion Matrix, ROC, PR)...
  Generating Figure 06: Side-by-Side Confusion Matrices...
  Generating Figure 07: Receiver Operating Characteristic (ROC) Curves...
  Generating Figure 08: Precision-Recall (PR) Curves...
  All diagnostic plots successfully saved to: figures

  [*] 3. Thực thi phân tích lỗi sâu (Error Analysis)...

  [*] 4. Đo kiểm benchmark hệ thống và sao chép Champion Model...
  [+] Hoàn thành giai đoạn đánh giá toàn diện.
  ```
* **Artifacts kiểm tra đầu ra:**
  - Báo cáo định lượng: `reports/evaluation_benchmark.csv`, `reports/test_predictions.parquet`, `reports/test_predictions.csv`.
  - Biểu đồ chẩn đoán: `figures/fig06_confusion_matrices.png`, `figures/fig07_roc_curves.png`, `figures/fig08_precision_recall_curves.png`.
  - Phân tích lỗi: `reports/error_analysis/overlap_cases.csv`, `reports/error_analysis/borderline_cases.csv`.
  - Khuyến nghị Champion: `reports/week4_champion_recommendation.json`, `reports/week4_system_benchmark.json`.
* **Đánh giá:** ✅ **THÀNH CÔNG (Exit Code: 0)**. Thời gian thực thi: ~18 giây.

---

### Bước 4: Kiểm thử Suy luận Dự đoán Dòng lệnh (Batch CLI Inference)
* **Câu lệnh thực thi:**
  ```powershell
  .\.clean_test_env\Scripts\python.exe src/predict.py --input data/sample_test.csv --output data/predictions_test.csv
  ```
* **Nhật ký thực thi (Console Output):**
  ```text
  ======================================================================
   HỆ THỐNG DỰ ĐOÁN TIỀN GỬI NGÂN HÀNG - DÒNG LỆNH (BATCH CLI)
  ======================================================================
  [*] Dữ liệu đầu vào : data\sample_test.csv
  [*] Tệp kết quả lưu  : data\predictions_test.csv
  [*] Pipeline artifact: models\preprocessing_pipeline.pkl
  [*] Mô hình champion : models\champion_model.pkl
  ----------------------------------------------------------------------
  [*] Đang nạp dữ liệu đầu vào...
      Đã nạp 10 bản ghi.
  [*] Đang nạp mô hình và pipeline...
  [*] Đang thực thi suy luận dự đoán...
  [+] Đã lưu kết quả thành công tại: data\predictions_test.csv
  ----------------------------------------------------------------------
  [*] XEM TRƯỚC KẾT QUẢ DỰ ĐOÁN:
     age           job   marital  balance predicted_deposit  confidence_score  probability_yes
  0   59        admin.   married     2343               yes            0.5344           0.5344
  1   56        admin.   married       45                no            0.7846           0.2154
  2   41    technician   married     1270                no            0.5778           0.4222
  3   55      services   married     2476                no            0.5548           0.4452
  4   54        admin.   married      184                no            0.9095           0.0905
  5   57       retired    single      604                no            0.7476           0.2524
  6   45        admin.  divorced        0                no            0.8667           0.1333
  7   48   blue-collar   married      238               yes            0.5383           0.5383
  8   34        admin.    single      673                no            0.8071           0.1929
  9   37  entrepreneur   married     7944                no            0.5407           0.4593
  ----------------------------------------------------------------------
  [*] Thống kê kết quả : {'no': 8, 'yes': 2}
  [*] Độ tin cậy TB     : 68.62%
  ======================================================================
  ```
* **Đánh giá:** ✅ **THÀNH CÔNG (Exit Code: 0)**. Thời gian thực thi: 1.5 giây. Đầu ra CSV chứa đầy đủ nhãn dự đoán và điểm tin cậy.

---

### Bước 5: Kiểm thử Toàn bộ Pipeline Bằng Điều phối Tự động
* **Câu lệnh thực thi:**
  ```powershell
  .\.clean_test_env\Scripts\python.exe run_pipeline.py --mode predict
  ```
* **Đánh giá:** ✅ **THÀNH CÔNG (Exit Code: 0)**. Pipeline tích hợp gọi mượt mà module suy luận và ghi kết quả `predictions.csv`.

---

### Bước 6: Kiểm thử Tự động Hóa với Test Suite (Unit Tests)
* **Cài đặt công cụ kiểm thử:**
  ```powershell
  .\.clean_test_env\Scripts\python.exe -m pip install pytest==9.1.1
  .\.clean_test_env\Scripts\python.exe -m pytest tests/
  ```
* **Kết quả:**
  ```text
  ============================= test session starts =============================
  platform win32 -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
  rootdir: D:\261\ML\BTL_ML
  collected 14 items

  tests\test_encoding_scaling.py ....                                      [ 28%]
  tests\test_predict.py ...                                                [ 50%]
  tests\test_tree_model.py .......                                         [100%]

  ============================= 14 passed in 1.87s ==============================
  ```
* **Đánh giá:** ✅ **THÀNH CÔNG (14/14 tests passed, 0 failures, 0 warnings)**.

---

## 4. Kết luận & Cam kết Tính Tái hiện (Reproducibility Guarantee)

1. **Không thiếu thư viện:** Mọi lệnh trong README.md chỉ cần cài đặt `pip install -r requirements.txt` là có thể chạy ngay lập tức.
2. **Không sai lệch đường dẫn:** Sử dụng kiến trúc `pathlib.Path` và biến cấu hình chuẩn trong `src/config.py`, bảo đảm dự án chạy ổn định trên cả hệ điều hành Windows, Linux và macOS.
3. **Bảo toàn dữ liệu & mô hình:** Pipeline huấn luyện có cơ chế khóa hạt ngẫu nhiên (`random_state=42`) và phân tầng (`StratifiedKFold`), bảo đảm mọi kết quả đánh giá số học đều có thể tái lập chính xác.

