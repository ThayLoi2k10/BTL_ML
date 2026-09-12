# Bài Tập Lớn Machine Learning

## 0. Link dataset: https://www.kaggle.com/datasets/goyaladi/twitter-bot-detection-dataset?resource=download&select=bot_detection_data.csv

## 1. Thành viên nhóm
* **Person 1:** Phan Ngọc Xuân Lợi - 2411969
* **Person 2:** [Họ tên] - [MSSV]
* **Person 3:** [Họ tên] - [MSSV]
* **Person 4:** [Họ tên] - [MSSV]

---

## 2. Cài đặt môi trường

Mở PowerShell tại thư mục muốn lưu dự án và chạy lần lượt:

```powershell
# 1. Kéo code về máy (chỉ chạy lần đầu tiên)
git clone https://github.com/ThayLoi2k10/BTL_ML.git

# 2. Tạo và kích hoạt môi trường ảo
py -m venv venv
.\venv\Scripts\activate

# 3. Cài đặt thư viện theo chuẩn của nhóm
pip install -r requirements.txt

# 4. Đăng ký kernel cho Jupyter Notebook / VS Code
python -m ipykernel install --user --name=btl_ml_env --display-name "Python (BTL_ML)"
```

---

## 3. Quy trình làm việc với Git

### Kéo code mới nhất trước khi làm
Mỗi khi bắt đầu làm việc, luôn lấy bản cập nhật mới nhất từ GitHub về máy:
```bash
git pull origin main
```

### Đẩy code trực tiếp lên nhánh main
Sau khi viết code hoặc chỉnh sửa notebook xong:
```bash
# 1. Xem danh sách các file đã chỉnh sửa
git status

# 2. Thêm toàn bộ thay đổi vào hàng chờ
git add .

# 3. Lưu commit kèm mô tả việc vừa làm
git commit -m "mo ta ngan gon cong viec"

# 4. Đẩy lên nhánh main
git push origin main
```

### Làm việc theo nhánh riêng (Khuyên dùng khi làm song song)
```bash
# 1. Tạo và chuyển sang nhánh riêng của bạn
git checkout -b feature-ten_ban

# 2. Thêm file và commit
git add .
git commit -m "hoan thanh phan viec"

# 3. Đẩy nhánh riêng lên GitHub
git push origin feature-ten_ban
```

---

## 4. Lưu ý quan trọng
* **Dữ liệu thô:** Tải file dữ liệu gốc từ Drive chung về, đổi tên thành `dataset.csv` và bỏ vào thư mục `data/raw/`. File này đã được chặn trong `.gitignore`, tuyệt đối không cố ép đẩy lên GitHub.
* **Kernel Notebook:** Khi mở file `.ipynb` trên VS Code, bấm vào nút chọn **Kernel** ở góc trên cùng bên phải và chọn đúng **`Python (BTL_ML)`**.