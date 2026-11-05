"""Targeted text corrections to the inherited PPTX; preserve all slide objects.

No layouts, charts, images or design elements are recreated. The corrected PDF
must be exported and inspected separately. Re-running is idempotent.
"""
from pathlib import Path
import re
import zipfile
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "docs/Slides.pptx"
REPLACEMENTS = {
    "Gap CV − Test ≤ 0.014 → không overfit · Dummy chỉ đạt 0.345":
        "Gap CV − Test nhỏ không loại trừ overfit · Dummy 0.345",
    "Precision / Recall tính cho lớp 'yes' tại ngưỡng 0.5. Ô tô xanh = tốt nhất.":
        "Lớp yes; model.predict (SVC: margin 0). PR-AUC = AP. Ô xanh = lớn nhất.",
    "Đường ROC và lựa chọn ngưỡng theo nghiệp vụ": "Đường ROC và minh họa đánh đổi ngưỡng",
    "Ngưỡng là quyết định kinh doanh": "Ngưỡng phải chọn trên validation",
    "SVC RBF tại τ = 0.50: Precision 78.4% — gần 4/5 cuộc gọi thành công":
        "SVC margin 0: Precision 78.4% trên Test; điểm không phải xác suất",
    "B · Cao điểm huy động → ưu tiên Recall": "B · Minh họa hậu nghiệm trên Test",
    "Random Forest hạ τ ≈ 0.41: ": "τ ≈ 0.41 chỉ minh họa, chưa triển khai: ",
    "0.64 ms": "0.55 ms", "53.7 ms": "28.38 ms", "1.19 ms": "0.91 ms",
    "2.71 triệu mẫu/s": "3.39 triệu mẫu/s", "24.1 nghìn mẫu/s": "44.36 nghìn mẫu/s",
    "2.3 nghìn mẫu/s": "3.09 nghìn mẫu/s",
    "Bối cảnh: chấm điểm danh sách gọi theo lô → ưu tiên Macro-F1 và Recall hơn độ trễ 1 mẫu":
        "Model-only predict; bản ghi c0ae203. Batch ưu tiên Macro-F1/Recall; chưa có SLA.",
    "Mổ xẻ ca sai: lỗi đến từ dữ liệu, không từ thuật toán": "Mổ xẻ ca sai và giới hạn dữ liệu quan sát",
    "FN = khách được thuyết phục trong cuộc gọi dài.  ": "FN có duration trung vị cao hơn TP.  ",
    "Mô hình trước cuộc gọi không thể biết → lỗi không thể khử.": "Chỉ là association; không chứng minh nguyên nhân hay lỗi không thể khử.",
    "RF bắt 15/107 khách 'yes', Logistic chỉ bắt 2/107 — điền khuyết tạo thiên lệch hệ thống":
        "RF bắt 15/107 khách 'yes', Logistic 2/107; association, chưa chứng minh nhân quả",
    "Champion: Random Forest (Tuned)": "Đề xuất Champion: Random Forest (Tuned)",
    "Hòa SVC về thống kê (p = 0.54)": "Chưa bác bỏ khác biệt với SVC (p = 0.54)",
    "Phân định bằng Recall +4.3 điểm % và tốc độ batch nhanh ~10×":
        "Recall +4.3 điểm %; batch ~14×. McNemar kiểm tra lỗi, không kiểm tra F1.",
    "41.5 MB · 54 ms/mẫu · giải thích mức trung bình":
        "41.5 MB · 28.38 ms/mẫu · chờ xác nhận nhóm",
    "Pipeline không rò rỉ, tái lập được": "Test tách khỏi fit; CV chưa fold-local",
    "So sánh công bằng 3 họ mô hình": "Cùng split và metric; ghi nhận giới hạn CV",
    "Trần thông tin của bài toán trước cuộc gọi": "Giới hạn thông tin trước cuộc gọi",
    "StratifiedKFold(5) — mỗi fold lần lượt làm tập validation":
        "StratifiedKFold(5); preprocessing/MI đã fit toàn Train trước CV",
    "GridSearch + F1-macro": "GridSearch + F1-macro (CV có giới hạn)",
    "Biến đổi đặc trưng: 16 → 74 → 32 cột": "Biến đổi đặc trưng: 16 → 15 → 39 → 74 → 32",
    "PR-AUC": "AP",
    "Phân loại nhị phân trên dữ liệu Bank Marketing — chấm điểm trước cuộc gọi, không rò rỉ dữ liệu":
        "Phân loại Bank Marketing trước cuộc gọi; Test tách khỏi fit, CV có giới hạn",
    "FN = khách được thuyết phục trong cuộc gọi dài.": "FN có duration trung vị cao hơn TP.",
    "chỉ dùng một lần ở Tuần 4": "dùng đánh giá, so sánh và phân tích lỗi ở Tuần 4",
    "không có dấu hiệu overfit lên Test": "không chứng minh không overfit; RF vẫn có gap learning curve lớn",
    "SVC tại 0.5 đạt 78.4%": "SVC tại margin 0 đạt Precision Test 78.4%",
    "mô hình cho xác suất; ngưỡng là quyết định kinh doanh": "RF cho điểm predict_proba chưa calibration; ngưỡng phải chọn trên validation",
    "(8 luồng CPU, scikit-learn 1.9.1)": "(20 CPU logic, Python 3.12.3, scikit-learn 1.9.1; bản ghi 03/11/2026)",
    "~24 nghìn mẫu/giây": "~44.36 nghìn mẫu/giây",
    "54 ms cho 1 mẫu": "28.38 ms cho 1 mẫu",
    "~2.3 nghìn mẫu/giây": "~3.09 nghìn mẫu/giây",
    "nguyên nhân nằm ở dữ liệu": "gợi ý giới hạn đặc trưng hiện có, không xác định nguyên nhân",
    "Họ có hồ sơ 'lạnh' nhưng được thuyết phục trong cuộc gọi. FP từ chối nhanh như nhóm TN.": "Chỉ là quan sát hậu nghiệm; không xác định tác động của độ dài cuộc gọi.",
    "Đây là lỗi không thể khử của bài toán dự đoán trước cuộc gọi, là cái giá có chủ đích của việc loại duration.": "Không đủ bằng chứng kết luận lỗi không thể khử. duration vẫn phải loại để đúng thời điểm dự đoán.",
    "→ KHÔNG khác biệt.": "→ chưa bác bỏ khác biệt, không chứng minh tương đương.",
    "nhanh hơn ~10 lần": "nhanh hơn ~14 lần",
    "pipeline chống rò rỉ, tái lập được; so sánh công bằng 3 họ mô hình": "Test tách khỏi fit; CV có preprocessing/MI trước fold; cùng quy ước so sánh ba họ mô hình",
}


def main():
    with zipfile.ZipFile(PATH) as z:
        entries = [(info, z.read(info.filename)) for info in z.infolist()]
    changed = []
    with zipfile.ZipFile(PATH, "w") as z:
        for info, data in entries:
            if info.filename.startswith(("ppt/slides/slide", "ppt/notesSlides/notesSlide")) and info.filename.endswith(".xml"):
                tree = etree.fromstring(data)
                dirty = False
                for elem in tree.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}t'):
                    original = elem.text or ""
                    if info.filename.startswith('ppt/notesSlides/'):
                        original = original.split("\nNghiệm thu 05/11/2026:")[0]
                    new = original
                    new = re.sub(r"(?:Đề xuất ){2,}", "Đề xuất ", new)
                    new = re.sub(r"(?: \(CV có giới hạn\)){2,}", " (CV có giới hạn)", new)
                    for before, after in REPLACEMENTS.items():
                        if after not in new:
                            new = new.replace(before, after)
                    if new != original:
                        elem.text = new
                        dirty = True
                # Keep one short, relevant qualification on the affected slides.
                if info.filename.startswith('ppt/notesSlides/'):
                    number = int(re.search(r'notesSlide(\d+)', info.filename).group(1))
                    notes = {
                        6: "CV dùng preprocessing/MI fit toàn Train trước fold; độ lạc quan chưa biết. Cần pipeline CV fold-local.",
                        7: "Learning curves cũng dùng feature selection cố định trước fold; không phải ước lượng end-to-end độc lập.",
                        9: "Mốc 0.41 là quan sát hậu nghiệm trên Test, không phải ngưỡng được phê duyệt; chọn trên validation mới.",
                        12: "McNemar kiểm tra lỗi ghép cặp, không kiểm tra Macro-F1; p-value chưa hiệu chỉnh nhiều phép so sánh.",
                        13: "Champion vẫn chờ xác nhận nhóm; không có SLA vận hành đã được duyệt.",
                    }
                    for elem in tree.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}t'):
                        if elem.text and len(elem.text) > 120:
                            elem.text = elem.text.split("\nNghiệm thu 05/11/2026:")[0]
                            if number in notes:
                                elem.text += "\nNghiệm thu 05/11/2026: " + notes[number]
                            dirty = True
                            break
                if dirty:
                    data = etree.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True)
                    changed.append(info.filename)
            z.writestr(info, data)
    print("Text parts corrected:", changed)


if __name__ == "__main__":
    main()
