# Hai lộ trình cùng một Lab 5

## Foundation (gợi ý HTTT)
- Mở Notebook 01: đọc dữ liệu, mô tả cách tạo nhãn và loại leakage.
- Notebook 02: chạy baseline, hiểu tại sao ẩn nhãn trong FIT mà giữ nhãn thật validation/test.
- Notebook 03: `TRACK="FOUNDATION"`; `tau=0.90` cố định, xem lịch sử số lượng pseudo-label.
- Kết luận từ Macro-F1 và confusion matrix; lý giải nếu Self-Training tệ hơn supervised.

## Applied (gợi ý PTPM)
- Chạy cùng Notebook 01–02, làm rõ chất lượng nhãn và đặc trưng.
- Notebook 03: `TRACK="APPLIED"`; thử `tau=.80/.90` trên validation **thuộc dữ liệu trước test**.
- Giải thích trade-off: ngưỡng thấp thêm nhiều pseudo-label hơn nhưng cũng có thể đưa nhiều lỗi hơn.
- Chốt ngưỡng dựa trên validation, rồi mới xem test một lần.

## Nguyên tắc
- Hai lộ trình đều có thể đạt **100/100**; sinh viên tự chọn.
- Không ép cải thiện điểm số: pseudo-label có thể làm xấu mô hình.
- Bất kể lộ trình nào: tuyệt đối không đưa nhãn mục tiêu, PM2.5 hay PM2.5 rolling vào features.
- Co-Training không bắt buộc, không cộng điểm chỉ vì chạy thêm thuật toán.
