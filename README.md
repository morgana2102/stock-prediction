# 📈 Stock Price Prediction with SMA Strategy & LSTM (PyTorch)

Dự án dự đoán giá cổ phiếu kết hợp chiến lược đường trung bình động (SMA) và mô hình LSTM (PyTorch). Project hỗ trợ lưu trữ mô hình theo từng ticker và hiển thị trực quan kết quả.

## 🚀 Tính năng chính:

✅ Tải dữ liệu thời gian thực từ Stooq  
✅ Chiến lược giao dịch SMA linh hoạt với tham số tùy chỉnh  
✅ Huấn luyện mô hình LSTM dự đoán giá đóng cửa cổ phiếu  
✅ Lưu mô hình riêng biệt cho từng ticker  
✅ Dự đoán giá tương lai từ mô hình đã train  
✅ Biểu đồ trực quan so sánh:
- Hiệu suất tích lũy giữa thị trường và chiến lược SMA
- Giá thực tế và giá dự báo từ LSTM
- Tỷ suất sinh lời hàng ngày thực tế và dự báo  

---

## 📂 Cấu trúc thư mục:

├── data/ # Thư mục dữ liệu thô nếu cần lưu cục bộ
├── model/ # Lưu trữ các mô hình LSTM theo từng ticker (ví dụ: lstm_model_AAPL.pth)
├── pic/ # Lưu trữ các biểu đồ được sinh ra
├── src/ # Toàn bộ mã nguồn chính
│ ├── config.py # Tham số cấu hình (ticker, SMA, đường dẫn, v.v.)
│ ├── data_loader.py # Load dữ liệu từ Stooq
│ ├── scaling_data.py # Chuẩn hóa dữ liệu
│ ├── strategy.py # Chiến lược SMA
│ ├── model.py # Mô hình LSTM với PyTorch
│ ├── train.py # Huấn luyện mô hình
│ ├── predict.py # Dự đoán giá tương lai
│ ├── visualization.py# Vẽ biểu đồ
│ ├── main.py # Tích hợp toàn bộ workflow

## Công nghệ sử dụng
- Python, PyTorch, Pandas, Matplotlib, Numpy

## Cách chạy dự án

- Cài đặt thư viện cần thiết:
```bash
pip install -r requirements.txt
```

- Cấu hình tham số tại ```src/config.py```

- Huấn luyện mô hình:
```bash
python -m src.train
```

- Dự đoán giá tương lai:
```bash
python -m src.predict
```

- Tích hợp toàn bộ workflow (SMA + LSTM):
```
python -m src.main
```
