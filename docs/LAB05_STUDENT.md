# FIT4112 – Lab 05: Semi-Supervised Learning với Self-Training

**Thời lượng dự kiến:** 200 phút · **Hình thức:** cá nhân · **Bài nộp:** 3 notebook + báo cáo tối đa 2 trang.

## Bối cảnh
Dữ liệu cảm biến chất lượng không khí có rất nhiều dòng nhưng chỉ một phần nhãn được cung cấp trong quá trình học máy. Làm thế nào tận dụng các mẫu chưa nhãn để huấn luyện mô hình?

**Nhắc lại:** nhãn trong bài là `risk_label` nhị phân minh họa, tạo bởi PM2.5 trung bình trượt 24 giờ với ngưỡng giảng dạy 75 µg/m³. **Không phải AQI chính thức**. Dữ liệu demo là giả lập.

## Hoạt động A – Nhãn và feature (20 điểm)
Chạy Notebook 01:
1. Xác định đặc trưng được phép: PM10, NO2, TEMP, PRES, WSPM, sin/cos giờ.
2. Đếm số mẫu, tỷ lệ hai lớp, giá trị thiếu.
3. Vì sao `PM2.5`, `pm25_24h`, `risk_label` không được xuất hiện trong X?
4. Nhãn proxy có giới hạn gì so với AQI chính thức?

## Hoạt động B – Chia dữ liệu và thiếu nhãn (20 điểm)
Chạy Notebook 02:
1. Kiểm tra FIT – validation – test theo thời gian.
2. Chỉ ẩn nhãn ở FIT, với `MISSING_FRACTION=0.75`.
3. Nêu số mẫu còn nhãn và chưa nhãn. Giải thích vì sao nhãn thật trong validation/test không thể được thêm vào pool pseudo-label.

## Hoạt động C – Baseline supervised và Self-Training (25 điểm)
Chạy Notebook 03:
- Foundation: `TRACK="FOUNDATION"`, `tau=0.90`.
- Applied: `TRACK="APPLIED"`, thử `tau=0.80` và `0.90`, chọn bằng **Macro-F1 trên validation**, test chỉ đánh giá sau lựa chọn.
- Cả hai cần báo cáo số pseudo-label được chọn qua các vòng lặp và giải thích tác động của ngưỡng tin cậy.
- **Không bắt buộc Self-Training phải tốt hơn baseline.** Khuyến khích phản biện pseudo-label sai.

## Hoạt động D – Đánh giá và trực quan hóa (20 điểm)
1. So sánh Accuracy và Macro-F1 trên **cùng tập test**.
2. Đọc hai confusion matrix, nêu lớp có nguy cơ bị nhầm.
3. Giải thích tại sao accuracy cao chưa đủ khi các lớp mất cân bằng.

## Hoạt động E – Báo cáo và giới hạn (15 điểm)
1. Tóm tắt kết quả bằng số liệu, không bịa số.
2. Đưa ra một tình huống Self-Training có thể gây hại.
3. Nêu hai giới hạn: dữ liệu giả lập; nhãn không phải AQI chính thức; calibration; distribution shift; pseudo-label bias.

## Rubric 100 điểm

| Tiêu chí | Điểm |
|---|---:|
| A. Hiểu nhãn, đặc trưng, chống leakage | 20 |
| B. Chia theo thời gian và ẩn nhãn đúng | 20 |
| C. Supervised baseline, Self-Training, ngưỡng tau | 25 |
| D. Macro-F1, Accuracy, confusion matrix, diễn giải | 20 |
| E. Báo cáo, kết luận có căn cứ, giới hạn | 15 |
| **Tổng** | **100** |

**Công bằng:** Foundation và Applied đều có thể đạt 100/100. Chọn Applied không tự động cộng điểm. Notebook 04 Co-Training chỉ tự chọn, không là điều kiện đạt tối đa.

## Nộp bài
- `01_prepare_labels.ipynb`, `02_missing_labels_and_baseline.ipynb`, `03_self_training_evaluation.ipynb` có kết quả và câu trả lời.
- `LAB05_REPORT.md` hoặc PDF, tối đa 2 trang theo `REPORT_TEMPLATE.md`.
- Không đưa `data/processed`, `.venv`, dữ liệu khách hàng/cảm biến riêng tư hoặc các file ZIP lớn vào repo bài nộp.
