# FIT4112 Lab 05 – Hướng dẫn giảng viên

## Mục tiêu
Hướng sinh viên từ supervised classification sang semi-supervised learning khi thiếu nhãn. Bài này **không yêu cầu lập trình thuật toán từ đầu**, không bắt buộc Papermill/Co-Training/dashboard.

## Kế hoạch thực hành dự kiến (200 phút)

| Hoạt động | Thời gian |
|---|---:|
| Khởi động: nhãn thật/pseudo-label, mục tiêu phân lớp | 20 phút |
| Notebook 01: data, nhãn proxy, leakage | 45 phút |
| Notebook 02: FIT/val/test, ẩn nhãn, baseline | 45 phút |
| Notebook 03: Self-Training, tau, metrics | 65 phút |
| Tổng kết: phản biện, viết báo cáo | 25 phút |

## Chuẩn bị máy học
```bash
python -m pip install -r requirements.txt
python tests/smoke_test.py
python tests/run_notebooks.py
```
Notebook 01 tự sinh dữ liệu giả lập, không tải dữ liệu thật. Dùng dữ liệu UCI là bài nâng cao; khi đổi nguồn phải chạy lại ba notebook.

## Nguyên tắc tránh leakage
1. **Nhãn proxy** được xây dựng từ rolling PM2.5 24 giờ, không có ý nghĩa AQI chính thức.
2. `FEATURES` chỉ gồm PM10, NO2, TEMP, PRES, WSPM, hour sin/cos; không có PM2.5, rolling PM2.5, datetime hay risk_label.
3. Chia theo thời gian FIT/validation/test; **chỉ FIT bị ẩn nhãn**. Validation không dùng làm pseudo-label; test giữ độc lập.
4. Pipeline có `SimpleImputer` fit trên phần huấn luyện, không dùng thống kê validation/test.
5. Với Applied, chọn tau trên validation một lần; tuyệt đối không điều chỉnh sau khi xem test.
6. Đây là bài học về **semi-supervised**, không cam kết cải thiện Macro-F1 và không dùng để phát cảnh báo sức khỏe.

## Hai lộ trình
Foundation (HTTT) chạy tau cố định 0.9, tập trung giải thích. Applied (PTPM) thử tau 0.8/0.9, giải thích lựa chọn bằng validation. **Cùng rubric 100 điểm, cùng cơ hội điểm tối đa.**

## Kiểm tra nhanh
- Chạy Notebook 01–03 mặc định Foundation và xem kết quả.
- Chạy lại Notebook 03 Applied, xác nhận có bảng validation, lịch sử pseudo-label và metric test.
- Xem confusion matrix cho hai phương pháp.
- Smoke test xác nhận tập validation/test không chứa nhãn ẩn, features không leakage, các phép dự đoán chạy được.
- Notebook 04 Co-Training là đề xuất nghiên cứu, **không có triển khai Co-Training bắt buộc**.
