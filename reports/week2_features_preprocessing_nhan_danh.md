# Bản thảo báo cáo Week 2 — Features / Preprocessing

## 3.1.  Feature Engineering và Feature Selection

### 3.1.1. Cơ sở lựa chọn đặc trưng

Feature engineering sử dụng trực tiếp kết luận Week 1: `balance`, `campaign` và `previous` lệch phải; `pdays=-1` là sentinel “chưa từng liên hệ”; `month`, `poutcome` và `contact` có association đáng chú ý với target; dữ liệu không có ID khách hàng rõ ràng. `duration` không được dùng vì chỉ biết trong hoặc sau cuộc gọi hiện tại và sẽ gây future-information leakage cho bài toán dự đoán trước cuộc gọi.

Transformer không sử dụng `deposit`. Tất cả 24 feature đều được tính từ thông tin có thể tồn tại trước cuộc gọi hiện tại.

### 3.1.2. Danh sách 24 engineered features

| STT | Feature | Công thức / logic | Nguồn | Ý nghĩa và lý do |
|---:|---|---|---|---|
| 1 | `age_group` | Chia `age` thành `<30`, `30–39`, `40–59`, `>=60` | `age` | Cho phép quan hệ theo giai đoạn tuổi thay vì chỉ tuyến tính. |
| 2 | `age_squared` | `(age/10)^2` | `age` | Bổ sung quan hệ cong theo tuổi thay vì chỉ một hệ số tuyến tính. |
| 3 | `balance_positive_log` | `log1p(max(balance, 0))` | `balance` | Giảm đuôi phải mạnh của phần số dư dương. |
| 4 | `balance_debt_log` | `log1p(max(-balance, 0))` | `balance` | Giữ riêng quy mô thấu chi thay vì xem số âm là lỗi. |
| 5 | `has_negative_balance` | `1(balance < 0)` | `balance` | Biểu diễn trạng thái thấu chi. |
| 6 | `balance_per_age` | `balance / max(age, 1)` | `balance`, `age` | Tương tác quy mô tài chính với giai đoạn vòng đời. |
| 7 | `any_loan` | `1(housing=yes OR loan=yes)` | `housing`, `loan` | Tóm tắt trạng thái có ít nhất một khoản vay. |
| 8 | `both_loans` | `1(housing=yes AND loan=yes)` | `housing`, `loan` | Nắm bắt tương tác khi đồng thời có hai khoản vay. |
| 9 | `credit_obligation_count` | Tổng ba cờ `default`, `housing`, `loan` | Ba biến tín dụng | Số trạng thái nghĩa vụ/rủi ro tín dụng được ghi nhận. |
| 10 | `financial_stress_flag` | `1(default=yes OR balance<0)` | `default`, `balance` | Kết hợp hai tín hiệu khó khăn tài chính có ý nghĩa nghiệp vụ. |
| 11 | `loan_with_negative_balance` | `1(any loan AND balance<0)` | `housing`, `loan`, `balance` | Tương tác giữa nghĩa vụ vay và trạng thái thấu chi. |
| 12 | `prior_contacted` | `1(pdays>=0 OR previous>0)` | `pdays`, `previous` | Tách trạng thái đã liên hệ khỏi sentinel `pdays=-1`. |
| 13 | `prior_contact_recency_log` | `log1p(pdays)` nếu `pdays>=0`, ngược lại 0 | `pdays` | Độ trễ liên hệ trước trên thang log. |
| 14 | `prior_contact_recent_30d` | `1(0<=pdays<=30)` | `pdays` | Cờ liên hệ gần đây theo mốc 30 ngày dễ diễn giải. |
| 15 | `previous_contacts_log` | `log1p(max(previous,0))` | `previous` | Giảm lệch phải của số lần liên hệ trước. |
| 16 | `previous_contact_intensity` | `previous/(pdays+1)` nếu đã liên hệ | `previous`, `pdays` | Kết hợp tần suất và độ gần đây của lịch sử liên hệ. |
| 17 | `campaign_contacts_log` | `log1p(max(campaign,0))` | `campaign` | Giảm đuôi phải mạnh của số lần liên hệ hiện tại. |
| 18 | `campaign_repeat_contact` | `1(campaign>1)` | `campaign` | Phân biệt lần đầu và liên hệ lặp lại. |
| 19 | `campaign_vs_previous_ratio` | `campaign/(previous+1)` | `campaign`, `previous` | Cường độ chiến dịch hiện tại so với lịch sử. |
| 20 | `successful_prior_contact_recency` | `1/(pdays+1)` khi `poutcome=success` | `poutcome`, `pdays` | Kết hợp thành công lịch sử với độ gần đây thay vì sao chép one-hot. |
| 21 | `failed_prior_campaign_recontact` | `1(poutcome=failure AND campaign>1)` | `poutcome`, `campaign` | Tương tác giữa thất bại quá khứ và nỗ lực liên hệ hiện tại. |
| 22 | `contact_known` | `1(contact không phải unknown/missing)` | `contact` | Missingness indicator cho 21,02% phương thức liên hệ không rõ. |
| 23 | `contact_month_sin` | `sin(2π(month-1)/12)` | `month` | Thành phần chu kỳ tháng, giữ quan hệ Dec–Jan. |
| 24 | `contact_month_cos` | `cos(2π(month-1)/12)` | `month` | Kết hợp với sin để biểu diễn đầy đủ vị trí trong năm. |

### 3.1.3. Một chiến lược chọn đặc trưng

Week 2 chỉ sử dụng **một** chiến lược: chọn top-K bằng Mutual Information sau encoding. Phương pháp này phù hợp vì `deposit` là target nhị phân, đầu vào sau `ColumnTransformer` hoàn toàn là số và MI có thể phát hiện association phi tuyến hoặc không đơn điệu mà Pearson không mô tả được.

Selector được fit bằng `X_train` và `y_train`; không đọc `y_test`. `random_state=42` cố định giúp kết quả xác định. `K=32` là ngân sách kỹ thuật bằng hai lần số predictor thô, không phải ngưỡng khoa học và không được tinh chỉnh dựa trên test.

| Giai đoạn | Số chiều |
|---|---:|
| Predictor thô | 16 |
| Sau khi loại `duration` leakage | 15 |
| Sau khi thêm đúng 24 feature | 39 |
| Sau encoding/scaling | 74 |
| Sau Mutual Information top-K | 32 |

Các cột được giữ lại gồm:

`pdays`, `poutcome_success`, `successful_prior_contact_recency`, `prior_contact_recency_log`, `previous_contact_intensity`, `balance`, `balance_per_age`, `contact_known`, `contact_missing`, `campaign_vs_previous_ratio`, `balance_positive_log`, `previous_contacts_log`, `age_squared`, `previous`, `credit_obligation_count`, `any_loan`, `prior_contacted`, `poutcome_missing`, `contact_cellular`, `contact_month_cos`, `housing_no`, `housing_yes`, `age`, `day`, `contact_month_sin`, `month_may`, `age_group_60_plus`, `balance_debt_log`, `month_mar`, `month_oct`, `month_sep`, `campaign`.

Danh sách trên là kết quả trên train split hiện tại; tên đầy đủ có prefix transformer và toàn bộ MI score được lưu trong output của `notebooks/04_features_pipeline.ipynb`.

## 3.2. Tích hợp pipeline và kiểm soát leakage

### 3.2.1. Thứ tự tiền xử lý

Pipeline cuối cùng có thứ tự:

1. `PreCallSchemaGuard`: kiểm tra schema, ghi nhận index dùng trong fit và loại `duration`;
2. `StringCleaner`;
3. `MissingHandler`;
4. `OutlierClipper`;
5. `BankMarketingFeatureEngineer`: thêm đúng 24 feature;
6. `ColumnTransformer`: dùng cấu hình encoding/scaling và cấu hình tương ứng cho feature mới;
7. `MutualInformationSelector`: chọn 32 feature bằng train labels.

`split_data` của Trí được dùng nguyên với stratification, `test_size=0.2`, `random_state=42`. Kết quả gồm 8.929 dòng train và 2.233 dòng test. Tỷ lệ `yes` là 47,3849% ở train và 47,3802% ở test, gần như tỷ lệ 47,3839% của toàn bộ dữ liệu.

### 3.2.2. Kiểm soát leakage

- Toàn bộ dữ liệu được split trước khi pipeline được fit.
- Lệnh học duy nhất là `pipeline.fit(X_train, y_train)`.
- X_test chỉ đi qua `pipeline.transform(X_test)`; `y_test` không được đưa vào pipeline.
- Mean, median, IQR, encoder categories, scaler parameters và MI scores đều được học từ train.
- Target `deposit` bị tách trước pipeline và schema guard từ chối target hoặc cột lạ/ID-like.
- `duration` bị loại trước bước cleaning; output không đổi khi cột này bị bỏ hoặc thay bằng giá trị cực đoan.
- `pdays=-1` tiếp tục được giữ làm sentinel; các feature ngày thực chỉ dùng `pdays>=0`.
- `poutcome` được giữ vì là kết quả chiến dịch trước, nhưng phải tiếp tục bảo đảm timestamp này có trước thời điểm dự đoán.

Hai sửa đổi nhỏ trên module kế thừa được phân loại rõ ràng:

- `src/cleaning.py`: **interface compatibility** — bỏ qua numeric column không còn trong input để integration có thể loại `duration` trước cleaning; thuật toán cleaning của Lợi không thay đổi.
- `src/encoding_scaling.py`: **execution bug** — smoke loader trỏ nhầm `dataset.csv`; đổi thành đúng `bank.csv`. Split và cấu hình encoding/scaling của Trí không thay đổi.

### 3.2.3. Kết quả và artifact

Train và test sau transform có kích thước lần lượt `(8929, 32)` và `(2233, 32)`, cùng tên/cùng thứ tự cột, không có NaN hoặc vô cực. Pipeline joblib reload và transform nhất quán trên cùng mẫu.

Các output được sinh cục bộ:

- `models/preprocessing_pipeline.pkl`;
- `data/processed/X_train.parquet`;
- `data/processed/X_test.parquet`;
- `data/processed/y_train.parquet`;
- `data/processed/y_test.parquet`.

Theo `.gitignore` hiện tại, `models/*.pkl` và `data/processed/*` là generated artifact không được stage. Chúng được kiểm chứng bằng cách reload cục bộ. Giai đoạn này dừng tại output preprocessing, chưa huấn luyện hoặc đánh giá mô hình.
