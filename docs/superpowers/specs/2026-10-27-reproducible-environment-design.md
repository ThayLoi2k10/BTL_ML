# Thiết kế: Chuẩn hóa môi trường và đảm bảo tính tái hiện

Ngày: 2026-10-27

## Mục tiêu

Chuẩn hóa dependencies của project theo những package thực sự cần thiết, tài liệu hóa chính xác cấu trúc và lệnh chạy, cung cấp CLI có thể chạy được cho tiền xử lý và suy luận dữ liệu thô, và xác minh quy trình trong một virtual environment mới.

## Bối cảnh đã xác minh

- Python hiện dùng trong project là Python 3.12.3.
- Source trong `src/` nhập trực tiếp `joblib`, `matplotlib`, `numpy`, `pandas`, `seaborn` và `scikit-learn`.
- Pipeline đọc/ghi Parquet cần `pyarrow` làm engine.
- `src/pipeline.py` hiện có các hàm build/fit/save nhưng chưa có lệnh CLI end-to-end.
- `src/predict.py` hiện chưa tồn tại.
- Project đã có preprocessing pipeline, ba mô hình tuần 3, champion artifact, dữ liệu thô/processed và nhiều artifact đánh giá được Git theo dõi.
- Thành viên được ghi nhận trong README/báo cáo: Person 1 Phan Ngọc Xuân Lợi (2411969), Person 2 Trí, Person 3 Danh, Person 4 Nhân. Họ tên đầy đủ và MSSV của Person 2–4 chưa có trong repo.

## Phạm vi

### Bao gồm

1. Thay `requirements.txt` bằng các dependency runtime trực tiếp được pin chính xác; thêm `pyarrow` vì Parquet là định dạng dữ liệu project.
2. Thêm `requirements-dev.txt` để cài runtime requirements cùng `pytest`, giữ pytest ngoài môi trường runtime cơ bản.
3. Thêm CLI preprocessing vào `src/pipeline.py`, giữ API hiện có.
4. Thêm `src/predict.py` để suy luận từ CSV dữ liệu thô qua preprocessing artifact và champion model.
5. Viết lại `README.md` với chủ đề/dataset chính xác, thành viên/phân công, cây thư mục, thiết lập venv, các lệnh tái lập, cách gọi inference, yêu cầu dữ liệu và artifact.
6. Tạo biên bản kiểm thử trong virtual environment mới trên máy hiện tại, nêu rõ đây không phải máy tính khác.

### Không bao gồm

- Không tạo Dockerfile trong lần này; Docker là mục tùy chọn và base image Python 3.10 không phù hợp với phiên bản thư viện/Python hiện đang dùng.
- Không xóa hoặc thay thế venv hiện có của người dùng.
- Không tạo hoặc bịa liên kết chia sẻ public cho dữ liệu/mô hình. Ghi liên kết nguồn dữ liệu công khai phù hợp và mô tả trạng thái artifacts trong Git.
- Không thay đổi thuật toán, kết quả huấn luyện, schema processed, hoặc các artifact model được track ngoài việc tạo artifact mới khi xác minh pipeline.

## Thiết kế kỹ thuật

### Dependency manifests

`requirements.txt` chỉ liệt kê runtime dependencies với cú pháp `package==x.y.z`, bao gồm các thư viện được import trong `src/` và engine Parquet bắt buộc. Phiên bản được lấy từ môi trường project đã cài và phải tương thích với Python 3.12. `requirements-dev.txt` tham chiếu `requirements.txt` và pin `pytest` cho kiểm thử.

### CLI tiền xử lý

`python -m src.pipeline` sẽ gọi một runner end-to-end mới trong `src/pipeline.py`:

1. Đọc `data/raw/bank.csv` và kiểm tra schema qua `load_correct_dataset`.
2. Tách target `deposit`; chia train/test 80/20 phân tầng với seed 42 trên dữ liệu thô.
3. Fit preprocessing pipeline chỉ trên phần train và transform cả train/test.
4. Xác nhận schema/giá trị đầu ra bằng `validate_transformed_splits`.
5. Lưu `models/preprocessing_pipeline.pkl` và bốn Parquet splits trong `data/processed/`.

Runner sẽ trả về metadata/paths để có thể kiểm thử với thư mục tạm; gọi trực tiếp module sẽ dùng project defaults.

### CLI suy luận

`python -m src.predict --input <raw.csv> --output <predictions.csv>`:

- Đọc raw CSV, yêu cầu đầy đủ predictor columns hợp lệ nhưng cho phép bỏ target `deposit`.
- Tải `models/preprocessing_pipeline.pkl` và `models/champion_model.pkl`.
- Gọi preprocessing `.transform()` (không fit lại), sau đó `predict()` và `predict_proba()` nếu model cung cấp xác suất.
- Ghi kết quả với nhãn dự đoán và xác suất lớp `yes` ra CSV, giữ nguyên thứ tự dòng.
- Báo lỗi rõ nếu input/schema/artifact thiếu hoặc model không cung cấp `predict_proba` cần thiết.

### README và các lệnh thực nghiệm

README sẽ mô tả các lệnh theo thứ tự phụ thuộc, gồm:

1. Tạo/kích hoạt Python 3.12 venv và cài `requirements.txt` (dev extras để chạy tests).
2. Cài đặt dữ liệu raw nếu không có sẵn, rồi chạy preprocessing CLI.
3. Chạy các module huấn luyện Logistic Regression, Random Forest và SVC.
4. Chạy `src.evaluation.metrics`, system benchmark, error analysis và plots theo đúng thứ tự đầu vào/đầu ra.
5. Gọi `src.predict` trên file CSV raw mẫu.
6. Chạy test suite.

README phân biệt lệnh yêu cầu raw data, model artifacts và outputs; giải thích các file hiện được track trong Git, nguồn tải dữ liệu Bank Marketing công khai, và không khẳng định có public link riêng cho model/data khi chưa tìm thấy URL đó.

### Clean-environment verification

Tạo venv kiểm thử mới, riêng biệt với `venv/` hiện tại; cài dependency bằng manifests, chạy tất cả các câu lệnh thực nghiệm được tài liệu hóa và test suite, đồng thời dùng một CSV nhỏ lấy từ schema raw làm inference smoke test. Ghi lại Python/OS, lệnh, trạng thái và các giới hạn vào `reports/clean_environment_test.md`. Không xóa venv hiện có. Việc thực hiện là trên cùng máy hiện tại, không được ghi là đã thử trên một máy khác.

## Kiểm thử

- Unit test cho runner preprocessing với dữ liệu/schema nhỏ hoặc monkeypatch đường dẫn, xác nhận pipeline fit train-only, artifact/splits được ghi đúng nơi và lỗi schema được nêu rõ.
- Unit test cho inference CLI bằng artifact sklearn nhỏ, xác nhận nhãn/xác suất và số dòng đầu ra; test các trường hợp thiếu artifact và schema sai.
- Cài lại trong clean venv mới; kiểm tra `pip check`, chạy test suite, preprocessing, ba lệnh huấn luyện, đánh giá/bảng benchmark, error analysis/plots và inference đúng thứ tự.
- So sánh shapes/schema cùng các kết quả metric được lưu với baseline hiện có; ghi nhận chênh lệch nếu có thay vì yêu cầu byte-identical giữa môi trường khác nhau.

## Đầu ra dự kiến

- Cập nhật: `requirements.txt`, `README.md`, `src/pipeline.py`.
- Tạo: `requirements-dev.txt`, `src/predict.py`, tests mới và `reports/clean_environment_test.md`.
- Không tạo Dockerfile trong scope đã duyệt.
