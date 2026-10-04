# Mục 3.4 — Mô hình phi tuyến (SVM)

## 3.4. Mô hình phi tuyến: Support Vector Machine

### 3.4.1. Lý do lựa chọn

Nhóm chọn **Support Vector Machine cho phân loại (`SVC`)** làm mô hình phi tuyến. Sau tiền xử lý ở Tuần 2, tập Train có **8.929 mẫu** và **32 đặc trưng** đã được mã hóa và chuẩn hóa. Hai lớp của `deposit` gần cân bằng (4.698 *no* / 4.231 *yes*). Đây là kích thước dữ liệu vừa phải: chi phí huấn luyện của SVM vào khoảng \(O(n^2)\)–\(O(n^3)\) theo số mẫu nên vẫn chạy được, và các đặc trưng đã được scale, vốn là điều kiện quan trọng để kernel RBF hoạt động đúng. Với số mẫu như vậy, MLP không có lợi thế rõ rệt mà lại cần nhiều siêu tham số hơn và khó tái lập hơn.

### 3.4.2. Cơ sở lý thuyết

**Siêu phẳng lề cực đại.** Với dữ liệu huấn luyện \(\{(x_i, y_i)\}_{i=1}^{n}\), \(y_i \in \{-1, +1\}\), SVM tìm siêu phẳng \(w^T x + b = 0\) phân tách hai lớp sao cho **lề** (khoảng cách từ siêu phẳng tới điểm gần nhất của mỗi lớp) là lớn nhất. Độ rộng lề bằng \(2/\lVert w \rVert\), nên cực đại lề tương đương cực tiểu \(\lVert w \rVert^2\).

**Lề mềm (soft margin).** Dữ liệu thực tế hiếm khi tách được hoàn toàn, nên SVM đưa thêm biến bù \(\xi_i \ge 0\) cho phép một số điểm vi phạm lề:

\[
\min_{w,b,\xi}\ \frac{1}{2}\lVert w\rVert^2 + C\sum_{i=1}^{n}\xi_i
\quad\text{s.t.}\quad y_i(w^T\phi(x_i)+b) \ge 1-\xi_i,\ \ \xi_i \ge 0
\]

Hệ số phạt \(C\) cân bằng giữa lề rộng và lỗi trên tập huấn luyện. \(C\) nhỏ ưu tiên lề rộng, mô hình đơn giản hơn (regularization mạnh) và có nguy cơ underfit. \(C\) lớn phạt nặng các điểm phân loại sai, làm biên quyết định uốn theo dữ liệu nhiều hơn và có nguy cơ overfit.

**Bài toán đối ngẫu và support vectors.** Dùng nhân tử Lagrange \(\alpha_i\), bài toán trên chuyển thành dạng đối ngẫu:

\[
\max_{\alpha}\ \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j K(x_i, x_j)
\quad\text{s.t.}\quad 0 \le \alpha_i \le C,\ \ \sum_i \alpha_i y_i = 0
\]

Hàm quyết định là \(f(x) = \operatorname{sign}\big(\sum_i \alpha_i y_i K(x_i, x) + b\big)\). Chỉ những điểm có \(\alpha_i > 0\), gọi là **support vectors** (các điểm nằm trên hoặc vi phạm lề), mới đóng góp vào biên quyết định.

**Kernel trick.** Trong dạng đối ngẫu, dữ liệu chỉ xuất hiện qua tích vô hướng \(\phi(x_i)^T\phi(x_j)\). Vì vậy có thể thay bằng một hàm kernel \(K(x_i, x_j)\) để làm việc ngầm trong không gian đặc trưng nhiều chiều (thậm chí vô hạn chiều) mà không cần tính \(\phi(x)\) tường minh. Nhờ đó, một siêu phẳng tuyến tính trong không gian \(\phi\) tương ứng với một biên phi tuyến trong không gian gốc. Nhóm thử hai kernel:

- **Linear:** \(K(x_i, x_j) = x_i^T x_j\), tương đương một bộ phân loại tuyến tính.
- **RBF (Gaussian):** \(K(x_i, x_j) = \exp\!\big(-\gamma \lVert x_i - x_j\rVert^2\big)\). Hệ số \(\gamma\) quyết định bán kính ảnh hưởng của mỗi support vector. \(\gamma\) nhỏ cho biên trơn, gần tuyến tính. \(\gamma\) lớn cho biên rất "gồ ghề", bám sát từng điểm và dễ overfit.

### 3.4.3. Thiết lập thực nghiệm

- **Mô hình:** `sklearn.svm.SVC(random_state=42)`.
- **Mốc so sánh trước tối ưu:** SVC với cấu hình mặc định (`kernel='rbf'`, `C=1`, `gamma='scale'`).
- **Tối ưu:** dùng framework chung `src/tuning.py` (`tune_model`, `GridSearchCV`) với không gian tìm kiếm gồm **33 tổ hợp**:
  - `kernel='linear'`: \(C \in \{0.1, 1, 10\}\). Kernel linear không dùng \(\gamma\) nên không quét \(\gamma\).
  - `kernel='rbf'`: \(C \in \{0.1, 1, 10, 100, 300, 1000\}\) × \(\gamma \in \{\text{scale}, 0.001, 0.01, 0.1, 1\}\). Hai giá trị \(C = 300, 1000\) được thêm sau lần quét đầu (lưới \(C \le 100\)), vì cấu hình tốt nhất khi đó nằm ở biên trên \(C=100\); mục đích là kiểm tra cực đại có nằm ngoài biên hay không.
- **Đánh giá:** **5-fold Stratified Cross-Validation** (`random_state=42`) và thước đo **F1-Macro**, giống hệt cách chia fold của mô hình tuyến tính ở Mục 3.2 để các kết quả so sánh được với nhau. Chỉ dùng tập Train; tập Test không được đụng tới.
- **Huấn luyện cuối:** mô hình với bộ siêu tham số tốt nhất được huấn luyện lại trên **toàn bộ tập Train** và đo thời gian bằng `time.perf_counter`.

### 3.4.4. Kết quả

**Bảng 3.4a — Điểm Cross-Validation trước và sau tối ưu**

| Mô hình | CV F1-Macro (mean ± std) | Siêu tham số |
|---|---:|---|
| SVC (Default) | 0.6960 ± 0.0051 | `kernel='rbf'`, \(C=1\), `gamma='scale'` |
| **SVC (Tuned)** | **0.7178 ± 0.0104** | `kernel='rbf'`, \(C=100\), \(\gamma=0.01\) |

**Bảng 3.4b — Một số tổ hợp tiêu biểu trong Grid Search** (đầy đủ 33 dòng trong `reports/nonlinear_cv_results.csv`)

| Hạng | Kernel | \(C\) | \(\gamma\) | CV F1-Macro (mean ± std) |
|---:|---|---:|---|---:|
| 1 | rbf | 100 | 0.01 | 0.7178 ± 0.0104 |
| 2 | rbf | 300 | 0.01 | 0.7172 ± 0.0116 |
| 3 | rbf | 1000 | scale | 0.7170 ± 0.0084 |
| 4 | rbf | 1 | 0.1 | 0.7164 ± 0.0082 |
| 5 | rbf | 10 | 0.01 | 0.7129 ± 0.0104 |
| 6 | rbf | 1000 | 0.01 | 0.7124 ± 0.0132 |
| 16 | linear | 10 | – | 0.6965 ± 0.0076 |
| 17 | rbf | 1 | scale | 0.6960 ± 0.0051 |
| 26 | rbf | 1000 | 0.1 | 0.6491 ± 0.0062 |
| 29 | rbf | 100 | 1 | 0.6358 ± 0.0052 |
| 33 | rbf | 0.1 | 1 | 0.3817 ± 0.0048 |

**Chi phí tính toán:** toàn bộ Grid Search (33 tổ hợp × 5 fold = 165 lần fit) mất khoảng **320 giây**. Huấn luyện mô hình tối ưu trên toàn bộ 8.929 mẫu Train mất **≈ 5,7 giây** và tạo ra **5.506 support vectors**. Các con số thời gian phụ thuộc vào cấu hình máy chạy.

### 3.4.5. Nhận xét

1. **Tối ưu siêu tham số có hiệu quả:** F1-Macro tăng từ 0.6960 lên 0.7178 (+0.0218), lớn hơn đáng kể so với độ lệch chuẩn giữa các fold (~0.005–0.010).
2. **Kernel RBF tốt hơn kernel linear:** SVC kernel linear chỉ đạt tối đa 0.6965, gần như bằng Logistic Regression ở Mục 3.2 (0.6960–0.6990). Trong khi đó, RBF đạt 0.7178. Điều này cho thấy dữ liệu có quan hệ phi tuyến mà mô hình tuyến tính không nắm bắt hết.
3. **Tương tác giữa \(C\) và \(\gamma\):** các cấu hình tốt nhất nằm trên một "dải" mà \(C\) tăng thì \(\gamma\) giảm (\(C=1,\gamma=0.1\); \(C=10/100,\gamma=0.01\)). Khi cả \(C\) và \(\gamma\) đều lớn (\(C=100,\gamma=1\) hoặc \(C=1000,\gamma=0.1\)), mô hình overfit và điểm CV giảm xuống còn 0.636–0.649. Khi \(C\) quá nhỏ kết hợp \(\gamma\) lớn (\(C=0.1,\gamma=1\)), mô hình suy biến gần như dự đoán một lớp (F1-Macro 0.38, ngang Dummy Baseline 0.3448).
4. **Kiểm tra biên của lưới:** ở lần quét đầu (\(C \le 100\)), cấu hình tốt nhất nằm đúng biên trên \(C=100\), nên chưa chắc đó là cực đại thật. Sau khi mở rộng lên \(C = 300\) và \(1000\), điểm không tăng thêm: với \(\gamma=0.01\), F1-Macro lần lượt là 0.7178 (\(C=100\)), 0.7172 (\(C=300\)) và 0.7124 (\(C=1000\)). Như vậy \(C=100\) là cực đại bên trong lưới chứ không phải hệ quả của việc lưới bị cắt. Đồng thời, \(C\) lớn làm thời gian fit tăng mạnh (khoảng 13 s lên 36 s mỗi fold với \(\gamma=0.01\)) mà không cải thiện điểm.
5. **Hạn chế:** khoảng 6 cấu hình đứng đầu cách nhau chưa tới 0.006 F1-Macro, nhỏ hơn độ lệch chuẩn giữa các fold (~0.01). Vì vậy, về mặt thống kê, các cấu hình này coi như tương đương; \(C=100,\gamma=0.01\) được chọn vì có điểm trung bình cao nhất. Số support vectors lớn (~62% số mẫu) cho thấy hai lớp chồng lấn nhiều. Điều này cũng làm SVM dự đoán chậm hơn và khó diễn giải hơn Logistic Regression, vì SVM kernel RBF không có hệ số đặc trưng trực tiếp.

Mô hình tối ưu được lưu tại `models/best_nonlinear_model.pkl`. Mã nguồn nằm ở `src/models/nonlinear_model.py`. Bảng tổng hợp lưu tại `reports/nonlinear_model_benchmark.csv`, và kết quả đầy đủ của Grid Search lưu tại `reports/nonlinear_cv_results.csv`.
