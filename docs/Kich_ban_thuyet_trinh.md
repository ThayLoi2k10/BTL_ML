# Kịch bản Thuyết trình & Bộ câu hỏi Phản biện

> **Phụ trách:** Danh (Person 3, Tuần 5)
> **File trình chiếu:** `docs/Slides.pptx` (14 slide) và bản xuất `docs/Slides.pdf`.
> **Tổng thời lượng mục tiêu:** **11 phút 15 giây** (khung quy định 10–12 phút).
> Lời nói chi tiết cho từng slide nằm trong **Speaker Notes** của file PowerPoint (View → Notes). Tài liệu này là bản tóm tắt để cả nhóm tập dượt.

---

## 1. Phân công người nói và thời lượng

| Slide | Nội dung | Người nói | Thời lượng | Câu chuyển sang người tiếp theo |
|:-:|:---|:-:|:-:|:---|
| 1 | Tiêu đề, giới thiệu nhóm | **Danh** | 0:20 | "Xin mời bạn Lợi." |
| 2 | Bài toán & lý do chọn ML | **Lợi** | 0:50 | — |
| 3 | Dữ liệu & 4 thách thức | **Lợi** | 0:50 | "Bạn Nhân trình bày pipeline xử lý các vấn đề này." |
| 4 | Pipeline chống rò rỉ dữ liệu | **Nhân** | 0:50 | "Bạn Trí trình bày đặc trưng và mã hóa." |
| 5 | Biến đổi đặc trưng 16 → 74 → 32 | **Trí** | 0:50 | — |
| 6 | Ba họ mô hình & Cross-Validation | **Trí** | 0:50 | "Bạn Nhân phân tích learning curves." |
| 7 | Learning curves (bias/variance) | **Nhân** | 0:40 | "Bạn Lợi trình bày kết quả trên tập Test." |
| 8 | Kết quả Test + lý do chọn Macro-F1 | **Lợi** | 1:00 | "Bạn Trí phân tích đường ROC và ngưỡng." |
| 9 | ROC & chọn ngưỡng theo nghiệp vụ | **Trí** | 1:00 | "Bạn Nhân trình bày ma trận đánh đổi." |
| 10 | Ma trận đánh đổi (Trade-off) | **Nhân** | 0:50 | "Bạn Danh phân tích các ca dự đoán sai." |
| 11 | Mổ xẻ ca sai: nguồn gốc lỗi | **Danh** | 0:55 | — |
| 12 | Điểm mù & giới hạn cấu trúc mô hình | **Danh** | 0:55 | "Bạn Nhân trình bày quyết định Champion." |
| 13 | Champion: Random Forest | **Nhân** | 0:40 | "Bạn Danh tổng kết." |
| 14 | Kết luận & hướng phát triển, mời Q&A | **Danh** | 0:45 | "Nhóm em xin cảm ơn, mời thầy/cô đặt câu hỏi." |

**Tổng theo người:** Lợi 2:40 · Trí 2:40 · Nhân 3:00 · Danh 2:55 → **11:15**.

**Nguyên tắc phân công:** mỗi người trình bày phần mình đã làm ở Tuần 1–4. Cụ thể: Lợi phụ trách mô hình tuyến tính và metrics; Trí phụ trách encoding, Random Forest và biểu đồ chẩn đoán; Danh phụ trách SVM và phân tích lỗi; Nhân phụ trách tích hợp pipeline, learning curves, benchmark và champion. Riêng phần EDA Tuần 1 (slide 3) đang giao cho Lợi; nếu người khác làm EDA thì đổi người nói cho slide này.

**Mẹo canh giờ:** khi tập, đặt đồng hồ 11:15. Nếu vượt quá 12:00, cắt câu ở slide 5 (bỏ phần ví dụ đặc trưng) và slide 10 (chỉ đọc 3 dòng đầu của bảng).

---

## 2. Thuật ngữ thống nhất khi nói

| Nói là | Không nói | Ý nghĩa |
|:---|:---|:---|
| **Rò rỉ dữ liệu** (leakage) | "lộ dữ liệu", "gian lận" | Dùng thông tin không có tại thời điểm dự đoán; ở đây là `duration` và thống kê của tập Test |
| **Trước cuộc gọi** | "trước khi chạy" | Bối cảnh dự đoán: chấm điểm khách trước khi telesale gọi |
| **Macro-F1** | "F1" chung chung | Trung bình F1 của hai lớp, trọng số bằng nhau |
| **Lớp dương = `yes`** | "lớp 1" | Khách đăng ký tiền gửi |
| **FN — bỏ sót khách** / **FP — gọi nhầm** | "sai loại 1/2" | Để hội đồng nghe là hiểu ngay |
| **Generalization gap** = CV − Test | "chênh lệch" | Âm nghĩa là điểm Test ≥ điểm CV |
| **Chồng lấn lớp** | "dữ liệu xấu" | Các mẫu giống nhau về đặc trưng nhưng khác nhãn |
| **Champion** | "mô hình tốt nhất" | Mô hình được chọn để triển khai (Random Forest Tuned) |
| **Ngưỡng τ** | "threshold" lẫn lộn | Mặc định 0.5; có thể đổi theo nghiệp vụ |

**Quy ước số liệu:** dấu chấm thập phân, dấu phẩy phân cách hàng nghìn (0.7263; 11,162), giống biểu đồ của nhóm. Khi nói thì đọc làm tròn: "không phẩy bảy hai sáu", "sáu mươi ba phần trăm".

---

## 3. Các con số "phải thuộc"

| Chỉ số | Giá trị | Nguồn |
|:---|:---|:---|
| Kích thước dữ liệu | 11,162 dòng · 16 thuộc tính · 47.4% yes | `bank.csv` |
| Train / Test | 8,929 / 2,233 (80/20, stratify, seed 42) | `split_data` |
| Số chiều | 16 → 15 → 39 → 74 → **32** | Pipeline Tuần 2 |
| CV Macro-F1 (LR / RF / SVC) | 0.6960 / **0.7233** / 0.7178 | `best_hyperparameters.json` |
| Test Macro-F1 (LR / RF / SVC) | 0.7098 / **0.7263** / 0.7185 | `evaluation_benchmark.csv` |
| RF trên Test | Recall 62.9% · Precision 76.3% · ROC-AUC 0.7747 · PR-AUC 0.7774 | như trên |
| SVC Precision | **78.4%** (cao nhất) | như trên |
| RF ở τ ≈ 0.41 | Recall 71.9% · Precision 68.4% | `test_predictions.parquet` |
| McNemar | RF vs LR p = 0.041 · RF vs SVC p = 0.54 | `error_analysis/head_to_head.csv` |
| Lỗi RF | 600 lỗi; 65% là FN; 72% chung cả 3 mô hình; 84% trong vùng chồng lấn | `error_analysis/` |
| Benchmark RF | 53.7 ms/mẫu · 24.1 nghìn mẫu/s · 41.5 MB | `week4_tradeoff_matrix.json` |

---

## 4. Bộ 5 câu hỏi phản biện trọng tâm

### Q1. Vì sao chọn Macro-F1 làm chỉ số chính, không dùng Accuracy? *(Lợi trả lời)*

**Trả lời ngắn (30 giây):**
- Mục tiêu nghiệp vụ là tìm khách `yes`. Accuracy có thể cao mà vẫn bỏ sót hết khách `yes`: Dummy luôn đoán `no` vẫn đạt Accuracy 52.6% với Recall bằng 0.
- Macro-F1 lấy trung bình F1 của hai lớp với **trọng số bằng nhau**, nên mô hình buộc phải làm tốt cả hai. F1 lại là trung bình điều hòa của Precision và Recall, nên phạt cả việc gọi nhầm (FP) lẫn bỏ sót (FN).
- Nhóm dùng **cùng một chỉ số** khi tuning (Tuần 3) và khi đánh giá (Tuần 4), nên so sánh CV với Test là nhất quán.
- Bổ sung thêm ROC-AUC và PR-AUC vì hai chỉ số này **không phụ thuộc ngưỡng**. Doanh nghiệp có thể đổi ngưỡng theo ngân sách gọi (slide 9).

**Nếu bị hỏi tiếp — "Dữ liệu gần cân bằng (47/53) thì Accuracy cũng được chứ?"**
Đúng là chênh lệch không lớn, nên Accuracy và Macro-F1 xếp hạng ba mô hình giống nhau. Tuy vậy, Macro-F1 vẫn cho thấy rõ sự đánh đổi Precision/Recall mà Accuracy che đi. Ví dụ SVC có Accuracy 72.7%, gần bằng RF (73.1%), nhưng Recall thấp hơn 4.3 điểm %.

---

### Q2. Nhóm chống rò rỉ dữ liệu trong pipeline như thế nào? *(Nhân trả lời)*

**Trả lời ngắn:**
1. **Chia Train/Test trước tiên**, phân tầng theo nhãn, seed 42. Chưa có bất kỳ phép `fit` nào trước bước này.
2. **Loại `duration`** ngay ở bước đầu (`PreCallSchemaGuard`): thời lượng cuộc gọi chỉ biết sau khi gọi, mà mô hình phải dự đoán trước cuộc gọi.
3. **Mọi bước học tham số nằm trong một `sklearn.Pipeline`** và chỉ `fit` trên X_train. Các tham số này gồm: median điền khuyết, ngưỡng IQR, encoder, scaler, bộ chọn Mutual Information. X_test chỉ đi qua `transform`.
4. **Tập Test bị khóa** suốt Tuần 3 (tuning, learning curves), chỉ dùng một lần ở Tuần 4.
5. **Có kiểm chứng:** biến đổi lại X_test thô bằng pipeline đã lưu cho kết quả trùng khớp `X_test.parquet`. Phần phân tích lỗi cũng kiểm tra từng dòng nhãn khớp với dữ liệu thô.

**Câu hỏi bẫy có thể gặp — "Khi chạy CV ở Tuần 3, pipeline có được fit lại trong từng fold không?"**
Trả lời trung thực: **không**. Pipeline được fit một lần trên toàn bộ X_train, sau đó CV chạy trên dữ liệu đã biến đổi. Vì vậy trong CV có một lượng **rò rỉ nhỏ giữa các fold**: scaler và bộ chọn Mutual Information đã "nhìn" cả fold validation. Hệ quả là điểm CV có thể hơi lạc quan. Tuy nhiên:
- Tập **Test hoàn toàn sạch**, vì pipeline không bao giờ thấy nó.
- Điểm Test **không thấp hơn** điểm CV (gap từ −0.014 đến −0.003), nên trên thực tế lượng rò rỉ này không làm sai kết luận.
- Hướng sửa: đưa cả pipeline vào bên trong `GridSearchCV` (Pipeline gồm preprocessing và model) để mỗi fold tự fit lại. Đây là cải tiến đã ghi nhận.

---

### Q3. Căn cứ toán học nào để chọn Random Forest làm Champion? *(Nhân trả lời, Danh bổ sung phần thống kê)*

**Trả lời ngắn:**
- **Chỉ số chính:** Test Macro-F1 cao nhất 0.7263. RF cũng cao nhất về Recall (62.9%), ROC-AUC (0.7747), PR-AUC (0.7774) và Youden J = TPR − FPR (0.4554).
- **Ổn định:** gap CV − Test = −0.003, nên không có dấu hiệu overfit lên Test.
- **Kiểm định McNemar.** Kiểm định này chỉ xét những mẫu mà hai mô hình bất đồng, và tính xác suất (theo phân phối nhị thức, giả thuyết H₀ là hai mô hình tốt như nhau) để thấy chênh lệch ít nhất như vậy:
  - RF vs Logistic: 174 mẫu chỉ RF đúng, 137 mẫu chỉ LR đúng → **p = 0.041 < 0.05**. RF tốt hơn có ý nghĩa thống kê.
  - RF vs SVC: 113 so với 103 → **p = 0.54**. Không khác biệt có ý nghĩa.
- Vì RF và SVC hòa về mặt thống kê, nhóm dùng **tiêu chí phụ theo nghiệp vụ**: Recall cao hơn 4.3 điểm % (bắt thêm khoảng 46 khách `yes` trên tập Test) và chấm điểm theo lô nhanh hơn khoảng 10 lần.
- **Đánh đổi chấp nhận:** file 41.5 MB, 54 ms cho một mẫu, khả năng giải thích ở mức trung bình. Các điểm này chấp nhận được vì mô hình được dùng để chấm điểm danh sách gọi theo lô.

**Nếu bị hỏi — "Chênh lệch 0.008 Macro-F1 giữa RF và SVC có đáng kể không?"** Không đáng kể về thống kê (p = 0.54). Nhóm nói rõ điều này và không khẳng định RF "vượt trội" SVC. Lựa chọn dựa trên Recall và chi phí vận hành.

---

### Q4. Bản chất của các ca dự đoán sai là gì? *(Danh trả lời)*

**Trả lời ngắn:**
- **Lỗi chủ yếu là FN (65%)**, tức là bỏ sót khách sẽ đăng ký.
- **Phần lớn là lỗi không thể khử** với dữ liệu trước cuộc gọi. Ba bằng chứng:
  1. 72% lỗi của RF cũng là lỗi của cả Logistic và SVC. Ba họ thuật toán khác nhau mà sai cùng chỗ.
  2. 84% lỗi nằm ở vùng chồng lấn lớp: đa số 15 khách giống nhất trong tập Train có nhãn ngược với nhãn thật.
  3. Hậu kiểm bằng `duration` (không dùng để huấn luyện): khách FN có cuộc gọi dài 732 giây, gấp đôi nhóm TP (314 giây). Những khách này có hồ sơ "lạnh" nhưng được thuyết phục **trong** cuộc gọi, điều mà mô hình trước cuộc gọi không thể biết.
- **Hai điểm mù có thể cải thiện:**
  - Khách chưa từng được liên hệ: Recall chỉ 50%, so với 86% ở khách đã từng liên hệ.
  - Khách có `contact = unknown`: RF chỉ bắt được 15/107 khách `yes`, Logistic chỉ bắt được 2/107. Khi điền khuyết, giá trị này thành mức `missing` và mô hình học nó như tín hiệu "không".
- **Giới hạn cấu trúc từng họ mô hình:**
  - Logistic giả thiết cộng tính, nên không biểu diễn được tương tác (hệ số `contact_missing` −0.40 luôn kéo xác suất xuống).
  - Random Forest có variance cao (gap train–validation 0.164 trên learning curve), dễ học thuộc vùng ít dữ liệu.

**Nếu bị hỏi — "Phương pháp kNN có phải lập luận vòng không?"** Một phần là có, vì kNN cũng là bộ phân loại cục bộ, và nhóm đã ghi rõ giới hạn này trong báo cáo. Nhưng kết luận được củng cố bởi hai bằng chứng độc lập: lỗi chung của 3 mô hình và kiểm tra hậu kiểm bằng `duration`.

---

### Q5. Rủi ro khi đưa mô hình vào môi trường thực tế là gì? *(Trí trả lời, Lợi bổ sung phần `predict.py`)*

**Trả lời ngắn — 6 rủi ro và cách giảm thiểu:**

| Rủi ro | Biểu hiện | Giảm thiểu |
|:---|:---|:---|
| **Data drift** | Hành vi khách, lãi suất, mùa vụ thay đổi; dữ liệu Test cùng nguồn với Train nên chưa phản ánh được tương lai | Theo dõi phân bố đặc trưng và tỉ lệ `yes` thực tế; huấn luyện lại định kỳ |
| **Xác suất chưa hiệu chỉnh** | P(yes) của RF/SVC chưa được calibration. Riêng SVC dùng sigmoid trên `decision_function` | Calibration (Platt/Isotonic) nếu dùng xác suất để ra quyết định theo chi phí |
| **Điểm mù dữ liệu** | Khách chưa từng liên hệ và khách `contact=unknown` bị bỏ sót nhiều | Cảnh báo riêng cho hai nhóm này; bổ sung đặc trưng |
| **Phiên bản thư viện** | `preprocessing_pipeline.pkl` được lưu bằng scikit-learn 1.6.1, nạp bằng 1.9.1 sinh cảnh báo `InconsistentVersionWarning` | Khóa phiên bản trong `requirements.txt` (phần của Trí), kiểm thử môi trường sạch |
| **Vòng phản hồi (feedback loop)** | Mô hình chỉ học được từ khách đã được gọi. Nếu chỉ gọi khách điểm cao thì nhãn tương lai bị lệch | Giữ một phần nhỏ danh sách gọi ngẫu nhiên để thu nhãn không thiên lệch |
| **Công bằng & giải thích** | Có đặc trưng tuổi (`age_group_60_plus`); RF khó giải thích từng quyết định | Rà soát theo nhóm tuổi; dùng feature importance hoặc SHAP khi cần giải trình |

Thêm về vận hành: RF mất khoảng 54 ms cho một mẫu và nặng 41.5 MB. Mức này ổn khi chấm điểm theo lô, nhưng nếu cần API thời gian thực tải cao thì nên benchmark lại trên hạ tầng thật.

---

## 5. Câu hỏi dự phòng (trả lời 1–2 câu)

| Câu hỏi | Người trả lời | Ý chính |
|:---|:-:|:---|
| Vì sao loại `duration` dù nó dự đoán rất tốt? | Nhân | Chỉ biết sau cuộc gọi. Dùng nó thì mô hình "biết trước đáp án" và vô dụng khi triển khai. Riêng `duration` đạt AUC 0.80 trên Test, cao hơn mọi mô hình, chính vì nó là thông tin tương lai |
| Vì sao chọn K = 32 đặc trưng? | Trí | Ngân sách kỹ thuật bằng 2 × số thuộc tính thô, chọn trước, không tinh chỉnh theo Test. Đây là một giới hạn có ghi nhận |
| Random Forest overfit (gap 0.164) mà vẫn được chọn? | Nhân | Gap đó là giữa train và validation. Điều quan trọng là điểm validation và Test cao nhất và ổn định (gap CV − Test −0.003). Bagging cùng `max_depth` và `min_samples_split` kiểm soát được variance |
| Sao không thử XGBoost/LightGBM? | Trí | Nằm ngoài phạm vi 3 họ mô hình của đề bài. Đây là hướng phát triển (slide 14). Phân tích lỗi cho thấy giới hạn chính nằm ở dữ liệu nên kỳ vọng mức tăng nhỏ |
| Điểm Test cao hơn CV có lạ không? | Lợi | Chênh lệch nhỏ (≤ 0.014), nằm trong dao động của một lần chia dữ liệu (CV std ≈ 0.004–0.010). Không phải dấu hiệu bất thường |
| Chạy dự đoán cho khách mới thế nào? | Lợi | `python src/predict.py --input data/sample_test.csv --output predictions.csv`. Script nạp pipeline và champion, in ra nhãn và xác suất (phần của Lợi ở Tuần 5) |

---

## 6. Checklist trước buổi bảo vệ

- [ ] Điền họ tên đầy đủ và MSSV của 4 thành viên lên slide 1 (hiện chỉ ghi tên gọi).
- [ ] Cả nhóm xác nhận Champion là Random Forest (file của Nhân đang ghi trạng thái *provisional*).
- [ ] Đối chiếu số liệu slide với `docs/Report.pdf` của Nhân, phải khớp 100%.
- [ ] Tập dượt bấm giờ ít nhất 2 lần; tổng thời gian trong khoảng 10:30–11:45.
- [ ] Mở sẵn `Slides.pptx` ở chế độ Presenter View để thấy speaker notes; mang theo `Slides.pdf` dự phòng.
- [ ] (Tùy chọn) Nếu Lợi làm xong `app.py`, chuẩn bị chạy demo trong phần Q&A.
