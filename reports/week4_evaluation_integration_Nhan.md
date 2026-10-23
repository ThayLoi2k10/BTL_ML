# Đóng góp Tuần 4 - Đánh giá và lựa chọn mô hình của Nhân

Tài liệu này tích hợp các kết quả Tuần 4 hiện có và bổ sung phần benchmark hệ thống, ma trận đánh đổi, giới hạn thực tế và khuyến nghị champion của Nhân. Các mô hình và pipeline từ Tuần 1-3 được giữ nguyên. Toàn bộ phép đo tốc độ dùng các artifact đã huấn luyện và `data/processed/X_test.parquet`; không có bước `fit` hoặc tiền xử lý lại.

## 4. Đánh giá mô hình

### 4.1. Chỉ số đánh giá và nguyên tắc thực nghiệm

Phần chỉ số định lượng kế thừa từ đóng góp của Lợi trong `src/evaluation/metrics.py` và `reports/week4_metrics_and_evaluation_Loi.md`. Bài toán là phân loại nhị phân với lớp mục tiêu `yes`. Macro-F1 tiếp tục là chỉ số chính vì đánh trọng số bằng nhau cho hai lớp và nhất quán với Cross-Validation của Tuần 3. Precision và Recall của lớp `yes`, Accuracy, ROC-AUC và PR-AUC được dùng để giải thích thêm sự đánh đổi nghiệp vụ.

Ba mô hình được đánh giá trên cùng tập test độc lập gồm 2.233 mẫu và 32 đặc trưng đã tiền xử lý. Tập test không tham gia huấn luyện, tuning hoặc lựa chọn siêu tham số. Generalization gap được tính theo quy ước:

`Generalization gap = Macro-F1 CV (Tuần 3) - Macro-F1 Test (Tuần 4)`.

### 4.2. Kết quả định lượng kế thừa từ Lợi

| Mô hình | CV Macro-F1 | Test Macro-F1 | Gap CV-Test | Accuracy | Precision yes | Recall yes | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression tuned | 0,6960 | 0,7098 | -0,0137 | 0,7147 | 0,7384 | 0,6163 | 0,7605 | 0,7654 |
| Random Forest tuned | 0,7233 | **0,7263** | -0,0030 | **0,7313** | 0,7626 | **0,6285** | **0,7747** | **0,7774** |
| SVC RBF tuned | 0,7178 | 0,7185 | -0,0007 | 0,7268 | **0,7835** | 0,5851 | 0,7729 | 0,7727 |

Cả ba gap đều nhỏ và mang dấu âm nhẹ, tức điểm test cao hơn điểm CV trong lần chia dữ liệu hiện tại. Kết quả này không cho thấy suy giảm ngoài mẫu, nhưng không phải bằng chứng rằng mô hình sẽ giữ nguyên hiệu năng trên dữ liệu tương lai hoặc dữ liệu từ một chiến dịch khác.

### 4.3. Đầu ra trực quan của Trí

Tại baseline `d7e2baa`, không có `src/evaluation/plots.py` hoặc artifact biểu đồ Tuần 4 thuộc phần Trí trên `main` hay các remote branch đã fetch. Vì vậy phần trực quan chưa thể tích hợp. Nhân không tạo biểu đồ thay thế để tránh nhận phần việc của Trí và không chèn kết quả giả vào báo cáo.

### 4.4. Benchmark hệ thống của Nhân

Benchmark được triển khai tại `src/evaluation/benchmark.py`. Cả ba artifact đều là estimator độc lập, nhận trực tiếp 32 đặc trưng đã tiền xử lý; tên và thứ tự cột trong `X_test` khớp hoàn toàn với `feature_names_in_`. Do đó benchmark không gọi `preprocessing_pipeline.pkl` lần thứ hai.

Quy trình đo sử dụng `time.perf_counter_ns`, 25 lượt warm-up và 200 lượt đo cho một mẫu; 3 lượt warm-up và 15 lượt đo cho toàn bộ batch 2.233 mẫu. Đầu vào được cố định, garbage collection tạm dừng trong vòng lặp đo và cấu hình thực thi đã lưu của từng estimator được giữ nguyên. Kích thước dùng MB thập phân, với 1 MB bằng 1.000.000 byte.

Môi trường đo: Windows 11, CPU AMD64 Family 25 Model 116 với 16 logical CPU, Python 3.13.0, scikit-learn 1.9.1, NumPy 2.1.2, pandas 2.2.3 và joblib 1.6.0. Thời gian dưới đây chỉ có ý nghĩa trong môi trường này và có thể thay đổi theo tải hệ thống hoặc phần cứng.

| Mô hình | Latency 1 mẫu median (ms) | Latency p95 (ms) | Batch 2.233 mẫu median (ms) | Throughput median (mẫu/s) | Kích thước (MB) | Train time đã ghi nhận | Tuning time đã ghi nhận |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0,3490** | **0,3818** | **0,4244** | **5.261.546** | **0,002575** | N/A | N/A |
| Random Forest | 41,1086 | 43,6485 | 61,5356 | 36.288 | 41,501817 | N/A | N/A |
| SVC RBF | 0,7476 | 1,0237 | 631,8272 | 3.534 | 1,522779 | 5,684 giây | khoảng 320 giây |

Thời gian train cuối và Grid Search của Logistic Regression, Random Forest không được lưu trong artifact Tuần 3 nên được ghi là N/A. Không huấn luyện lại chỉ để điền hai ô này. Đối với SVC, 5,684 giây là thời gian fit toàn bộ tập train và khoảng 320 giây là thời gian Grid Search do Danh ghi nhận; hai giá trị này phụ thuộc phần cứng.

## 5. So sánh, phân tích lỗi và giới hạn

### 5.1. Trade-off Comparison

| Tiêu chí | Logistic Regression | Random Forest | SVC RBF |
|---|---|---|---|
| Hiệu năng dự đoán | Macro-F1 0,7098; thấp nhất trong ba mô hình nhưng cao hơn rõ rệt so với Dummy | **Macro-F1 0,7263 và Recall yes 0,6285 cao nhất** | Macro-F1 0,7185; **Precision yes 0,7835 cao nhất** |
| Tổng quát hóa | Gap -0,0137; nhỏ | Gap -0,0030; rất nhỏ | Gap -0,0007; nhỏ nhất |
| Độ trễ đơn mẫu | **Nhanh nhất**, median 0,3490 ms | Chậm nhất, median 41,1086 ms do overhead của ensemble 300 cây | 0,7476 ms; nhanh hơn Random Forest trong phép đo đơn mẫu |
| Throughput batch | **Cao nhất**, khoảng 5,26 triệu mẫu/s | Khoảng 36,3 nghìn mẫu/s; phù hợp chấm điểm chiến dịch theo batch trên máy đo | Thấp nhất, khoảng 3,53 nghìn mẫu/s |
| Kích thước artifact | **0,002575 MB** | **41,501817 MB**, lớn nhất | 1,522779 MB |
| Khả năng giải thích | Cao: có hệ số có dấu và độ lớn | Trung bình: có global feature importance nhưng không có luật quyết định gọn | Thấp: phụ thuộc support vectors và kernel RBF |
| Điểm mạnh thực tế | Nhẹ, nhanh, dễ giải thích | Cân bằng hiệu năng/recall tốt nhất, batch throughput cao | Precision cao, footprint nhỏ hơn Random Forest |
| Điểm yếu thực tế | Bỏ lỡ tương tác phi tuyến; Macro-F1 thấp hơn | Artifact lớn, latency đơn mẫu cao, giải thích hạn chế hơn tuyến tính | Batch chậm, Recall yes thấp nhất, khó giải thích |

Không có một mô hình thống trị trên mọi tiêu chí. Logistic Regression phù hợp khi footprint, latency và khả năng giải thích là ràng buộc chính. SVC phù hợp hơn khi ưu tiên Precision lớp `yes` và chấp nhận throughput batch thấp. Random Forest phù hợp nhất với mục tiêu hiện tại là chấm điểm chiến dịch theo batch và tối ưu Macro-F1/Recall, miễn là dung lượng 41,50 MB và mức giải thích trung bình được chấp nhận.

### 5.2. Phân tích lỗi kế thừa từ Danh

Phần này sử dụng trực tiếp `reports/week4_error_analysis_Danh.md` và các bảng trong `reports/error_analysis/`; không chạy lại hoặc sửa `src/evaluation/error_analysis.py`.

- Logistic Regression có 637 lỗi, Random Forest 600 lỗi và SVC 610 lỗi trên 2.233 mẫu test.
- False negative chiếm 63,7% lỗi của Logistic Regression, 65,5% của Random Forest và 72,0% của SVC. Vì vậy Precision cao của SVC đi kèm số khách hàng `yes` bị bỏ sót nhiều hơn.
- Random Forest tốt hơn Logistic Regression trong so sánh McNemar (`p = 0,041`), nhưng Random Forest và SVC không khác biệt có ý nghĩa thống kê (`p = 0,540`). Chênh lệch Macro-F1 nhỏ giữa hai mô hình phi tuyến cần được diễn giải thận trọng.
- Khoảng 72% lỗi Random Forest là lỗi chung của cả ba mô hình. Điều này cho thấy giới hạn thông tin của dữ liệu quan trọng hơn việc tiếp tục đổi họ thuật toán.
- Hai nhóm khó nổi bật là khách hàng chưa từng được liên hệ và khách hàng có `contact=unknown`. Ngoài ra, `poutcome=success` là tín hiệu rất mạnh nhưng vẫn tạo một nhóm false positive mà cả ba mô hình cùng mắc.
- `duration` có quan hệ hậu nghiệm mạnh với kết quả nhưng chỉ biết sau cuộc gọi. Biến này đã không được dùng trong 32 đặc trưng dự đoán, phù hợp với yêu cầu tránh leakage cho chấm điểm trước khi liên hệ.

### 5.3. Practical Limitations

1. **Chưa có SLA vận hành.** Dự án chưa ghi nhận yêu cầu chính thức về latency, throughput, RAM hoặc kích thước model. Vì vậy benchmark hỗ trợ so sánh tương đối trên máy đo, không chứng minh một artifact đáp ứng hạ tầng triển khai cụ thể.
2. **Timing phụ thuộc môi trường.** Kết quả chịu ảnh hưởng của CPU, phiên bản thư viện, tải nền, cơ chế song song `n_jobs=-1` của Random Forest và cache hệ thống. Cần benchmark lại trên hạ tầng triển khai trước khi chốt ngân sách tài nguyên.
3. **Thiếu thời gian train của hai mô hình.** Tuần 3 không lưu thời gian Grid Search hoặc fit cuối của Logistic Regression và Random Forest. Các giá trị này được giữ là N/A; việc tái huấn luyện nằm ngoài phạm vi Tuần 4 của Nhân.
4. **Điểm SVC không phải xác suất đã hiệu chỉnh.** `metrics.py` ánh xạ `decision_function` qua sigmoid để có giá trị từ 0 đến 1. Phép biến đổi đơn điệu vẫn giữ thứ hạng cho ROC-AUC/PR-AUC, nhưng không chứng minh calibration xác suất. Các diễn giải theo ngưỡng xác suất hoặc độ tin cậy của SVC cần thận trọng.
5. **Giới hạn dữ liệu trước cuộc gọi.** Nhiều lỗi tập trung ở vùng chồng lấn lớp, thiếu lịch sử liên hệ và thiếu kênh liên lạc. `duration` chỉ có sau cuộc gọi nên không thể dùng cho hệ thống chọn khách hàng trước chiến dịch. Đây là mức trần thông tin của bài toán hiện tại.
6. **Khả năng chuyển miền chưa được kiểm tra.** Test set đến từ cùng nguồn và quy trình chia dữ liệu hiện tại. Chưa có đánh giá theo thời gian, chi nhánh, chiến dịch mới hoặc drift sau triển khai.
7. **Đầu ra biểu đồ của Trí còn thiếu.** Không có `plots.py` hoặc figure Tuần 4 để tích hợp tại baseline đã inspect. Báo cáo chưa thể xác nhận các đường ROC/PR, confusion matrix hoặc calibration plot do Person 2 phụ trách.

### 5.4. Khuyến nghị champion tạm thời

**Khuyến nghị tạm thời: Random Forest tuned**, phục vụ bối cảnh chấm điểm chiến dịch theo batch.

Cơ sở lựa chọn gồm bốn nhóm bằng chứng: Test Macro-F1 cao nhất (0,7263), Recall lớp `yes` cao nhất (0,6285), generalization gap nhỏ (-0,0030), và throughput batch đo được khoảng 36.288 mẫu/giây. So với SVC, Random Forest tăng Macro-F1 khoảng 0,0078 và Recall khoảng 0,0435, trong khi batch nhanh hơn khoảng 10 lần trên máy thử nghiệm. Đổi lại, artifact lớn hơn nhiều và latency đơn mẫu cao hơn.

Khuyến nghị này **chưa phải quyết định được cả nhóm xác nhận**. McNemar không cho thấy Random Forest khác SVC có ý nghĩa thống kê, và dự án chưa có SLA vận hành. Nếu yêu cầu cuối cùng ưu tiên model cực nhẹ, latency đơn mẫu và khả năng giải thích, Logistic Regression có thể là lựa chọn thực dụng hơn. Nếu ưu tiên Precision lớp `yes`, SVC là phương án thay thế cần xem xét.

`models/champion_model.pkl` là bản sao byte-for-byte của `models/best_tree_model.pkl`; SHA-256 của hai file giống nhau. Ba artifact Tuần 3 ban đầu được giữ nguyên.

## Artifact và khả năng tái lập

- Benchmark đầy đủ: `reports/week4_system_benchmark.json`.
- Ma trận đánh đổi dạng máy đọc: `reports/week4_tradeoff_matrix.json`.
- Quyết định và trạng thái xác nhận: `reports/week4_champion_recommendation.json`.
- Notebook tích hợp: `notebooks/06_evaluation.ipynb`.
- Artifact đề xuất tạm thời: `models/champion_model.pkl`.

Tái chạy benchmark từ thư mục gốc bằng `py -3.13 -m src.evaluation.benchmark`. Lệnh chỉ gọi `predict`, ghi lại benchmark của Nhân và sao chép artifact đã chọn; không gọi `fit` hoặc ghi đè các model Tuần 3.
