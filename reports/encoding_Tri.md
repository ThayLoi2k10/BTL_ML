# 2.3. Mã hóa biến phân loại, chuẩn hóa thang đo và phân chia dữ liệu

## Chiến lược phân chia dữ liệu

Tập dữ liệu được chia thành hai phần độc lập: tập huấn luyện và tập kiểm thử theo tỷ lệ 80/20. Cụ thể, 80% số dòng được dùng để huấn luyện mô hình và 20% còn lại được giữ riêng để đánh giá khả năng tổng quát hóa của mô hình trên dữ liệu chưa từng thấy. Quá trình chia dữ liệu sử dụng `random_state=42` để đảm bảo kết quả có thể tái lập khi chạy lại trên cùng một tập dữ liệu.

Vì bài toán dự đoán `deposit` là bài toán phân loại nhị phân với hai nhãn `yes` và `no`, quá trình chia dữ liệu sử dụng cơ chế phân tầng `stratify=y`. Cơ chế này giúp giữ tỷ lệ hai lớp nhãn trong tập train và test gần giống với tỷ lệ ban đầu của toàn bộ dữ liệu. Nhờ đó, tập kiểm thử phản ánh tốt hơn phân bố thực tế và tránh trường hợp một lớp bị xuất hiện quá ít sau khi chia dữ liệu.

## Chiến lược mã hóa biến phân loại

Các biến phân loại được xử lý theo hai nhóm chính: biến có thứ bậc và biến danh nghĩa.

Với biến có thứ bậc, cột `education` được mã hóa bằng `OrdinalEncoder` vì các mức học vấn có quan hệ thứ tự tự nhiên: `primary < secondary < tertiary`. Cách mã hóa này giúp giữ lại thông tin thứ bậc giữa các nhóm học vấn. Bộ mã hóa được cấu hình thêm `handle_unknown='use_encoded_value'` và `unknown_value=-1` để xử lý an toàn nếu tập kiểm thử xuất hiện giá trị chưa từng có trong tập huấn luyện.

Với các biến phân loại danh nghĩa gồm `job`, `marital`, `default`, `housing`, `loan`, `contact`, `month` và `poutcome`, nhóm sử dụng `OneHotEncoder`. Các biến này không có thứ tự tự nhiên, do đó không nên ánh xạ trực tiếp thành số nguyên vì có thể khiến mô hình hiểu sai rằng tồn tại quan hệ lớn hơn/nhỏ hơn giữa các nhóm. `OneHotEncoder` tạo ra các cột nhị phân riêng cho từng nhãn, giúp biểu diễn thông tin phân loại rõ ràng hơn. Tham số `handle_unknown='ignore'` được bật để tránh lỗi khi tập test hoặc dữ liệu mới xuất hiện nhãn lạ chưa có trong tập train.

## Chiến lược chuẩn hóa thang đo biến số

Các biến số được chia thành hai nhóm chuẩn hóa. Nhóm `balance`, `duration`, `campaign`, `pdays` và `previous` được chuẩn hóa bằng `RobustScaler`. Đây là các đặc trưng có thể có dải giá trị biến thiên lớn, phân bố lệch hoặc chứa ngoại lai. `RobustScaler` sử dụng trung vị và khoảng tứ phân vị IQR nên ít bị ảnh hưởng bởi các giá trị cực đoan hơn so với chuẩn hóa dựa trên trung bình và độ lệch chuẩn.

Hai biến `age` và `day` được chuẩn hóa bằng `StandardScaler`. Các biến này có miền giá trị rõ ràng hơn và ít chịu ảnh hưởng bởi ngoại lai cực đoan hơn nhóm trên. `StandardScaler` đưa dữ liệu về dạng có trung bình bằng 0 và độ lệch chuẩn bằng 1, giúp các thuật toán nhạy cảm với thang đo hoạt động ổn định hơn.

## Cấu hình ColumnTransformer

Toàn bộ cấu hình mã hóa và chuẩn hóa được đóng gói dưới dạng danh sách tuple `(name, transformer, columns)` tương thích với `ColumnTransformer` của Scikit-Learn. Cách tổ chức này giúp đảm bảo các bộ biến đổi chỉ được `fit` trên tập huấn luyện rồi sau đó mới `transform` tập kiểm thử, qua đó hạn chế rò rỉ dữ liệu từ tập test sang quá trình huấn luyện.

| Nhóm cột | Cột áp dụng | Bộ chuyển đổi | Lý do lựa chọn |
|---|---|---|---|
| Biến phân loại có thứ bậc | `education` | `OrdinalEncoder` | Giữ quan hệ thứ tự `primary < secondary < tertiary` |
| Biến phân loại danh nghĩa | `job`, `marital`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome` | `OneHotEncoder(handle_unknown='ignore')` | Không áp đặt thứ tự giả và tránh lỗi khi có nhãn lạ |
| Biến số lệch/dao động lớn | `balance`, `duration`, `campaign`, `pdays`, `previous` | `RobustScaler` | Bền vững hơn trước ngoại lai nhờ median và IQR |
| Biến số ổn định hơn | `age`, `day` | `StandardScaler` | Đưa về trung bình 0 và độ lệch chuẩn 1 |
