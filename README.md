# FIT4112 – Lab 5: Học bán giám sát (Self-Training) khi thiếu nhãn

**Học phần:** Khai phá dữ liệu · **Đối tượng:** sinh viên HTTT (Foundation) và PTPM (Applied).  
**Nguồn thiết kế:** [air_guard](https://github.com/TrangLe1912/air_guard), rút gọn cho sinh viên từ [air_quality_classification](https://github.com/TrangLe1912/air_quality_classification).  
**Độ khó:** cơ bản → vận dụng; **một buổi dự kiến 200 phút**.

Lab tiếp nối [Lab 4 – dự báo PM2.5](https://github.com/TrangLe1912/fit4112_lab_4), nhưng **khác bài toán**: Lab 5 phân lớp nhị phân rủi ro ô nhiễm *mang tính minh họa* khi chỉ một phần nhãn sẵn có. Trọng tâm là **thiếu nhãn, pseudo-label, self-training, Macro-F1 và nguy cơ đánh giá sai**, không phải triển khai cảnh báo thật.

> **Quan trọng:** `risk_label` được tạo từ trung bình PM2.5 trong 24 giờ gần nhất với ngưỡng **75 µg/m³ chỉ để minh họa**. Đây **không phải AQI chính thức**, không dùng cho hướng dẫn sức khỏe hoặc cảnh báo môi trường thực tế. Dữ liệu mặc định là **giả lập**.

## Sau lab, sinh viên có thể
1. Giải thích phân biệt supervised và semi-supervised; phân biệt nhãn thật và pseudo-label.
2. Chia dữ liệu theo thời gian, **ẩn nhãn chỉ trong FIT**, giữ nguyên validation/test.
3. Huấn luyện supervised baseline từ tập còn nhãn.
4. Chạy Self-Training với ngưỡng xác suất tin cậy `tau` và quan sát số nhãn giả được thêm.
5. So sánh Accuracy, Macro-F1 và Confusion Matrix, giải thích khi Self-Training không cải thiện.

## Cấu trúc
```text
fit4112_lab_5/
├── README.md
├── requirements.txt
├── src/lab5.py
├── scripts/generate_demo_data.py
├── data/raw/README.md
├── notebooks/
│   ├── 01_prepare_labels.ipynb
│   ├── 02_missing_labels_and_baseline.ipynb
│   ├── 03_self_training_evaluation.ipynb
│   └── 04_co_training_optional.ipynb
├── docs/
│   ├── LAB05_STUDENT.md
│   ├── TRACKS.md
│   ├── REPORT_TEMPLATE.md
│   └── INSTRUCTOR_GUIDE.md
├── tests/
│   ├── smoke_test.py
│   └── run_notebooks.py
└── .github/workflows/validate.yml
```

## Cài đặt và chạy
Python 3.11 hoặc 3.12, dùng Jupyter hoặc VS Code:
```bash
git clone https://github.com/TrangLe1912/fit4112_lab_5.git
cd fit4112_lab_5
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```
Chạy **01 → 02 → 03** theo thứ tự. Notebook 01 tạo dữ liệu giả lập tại `data/raw/demo_air_quality.csv` nếu chưa có. Không cần tải dữ liệu bên ngoài, không cần Papermill. Các notebook ghi kết quả vào `data/processed/` (gitignore).

## Hai lộ trình cùng rubric 100 điểm

| | Foundation (gợi ý HTTT) | Applied (gợi ý PTPM) |
|---|---|---|
| Huấn luyện | Code mẫu supervised / Self-Training | Cùng quy trình |
| Ngưỡng tin cậy | `tau=0.90` cố định | `tau ∈ {0.80,0.90}`, chọn bằng validation |
| Trọng tâm | Hiểu pseudo-label, so sánh Macro-F1 | Lập luận chọn tau, trade-off số nhãn giả |
| Điểm tối đa | **100/100** | **100/100** |

Người học tự chọn lộ trình; không khóa theo lớp. Xem [docs/TRACKS.md](docs/TRACKS.md).

## Nộp bài
- Ba notebook đã chạy, có câu trả lời/biểu đồ.
- Báo cáo theo [mẫu](docs/REPORT_TEMPLATE.md), tối đa **2 trang**.
- Ghi Macro-F1 của baseline và Self-Training trên **cùng test**, số pseudo-label được thêm, nêu giới hạn.
- [Phiếu thực hành và rubric](docs/LAB05_STUDENT.md).
- **Không bắt buộc:** Co-Training, dashboard, Papermill, thử nhiều mô hình hay viết Self-Training từ đầu.

## Chạy kiểm thử dành cho giảng viên
```bash
python tests/smoke_test.py
python tests/run_notebooks.py
```

## Dữ liệu thực (tự chọn)
Có thể thử [Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data): đặt ZIP theo hướng dẫn trong [data/raw/README.md](data/raw/README.md), đổi `DATA_SOURCE="uci_zip"` trong notebook 01 và chạy lại 01–03. **Không** hiểu kết quả rủi ro minh họa là AQI chính thức.

**Lưu ý về leakage:** `PM2.5`, `pm25_24h`, `datetime` và `risk_label` bị loại khỏi `FEATURES`; model chỉ dùng PM10, NO2, TEMP, PRES, WSPM, sin/cos của giờ. Tất cả phép điền thiếu dùng `SimpleImputer` được fit trên phần train có nhãn/pseudo-label, không fit trên validation/test.
