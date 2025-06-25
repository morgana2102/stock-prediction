import torch
import os
from src.data_loader import load_realtime_data
from src.model import LSTMModel
from src.config import default_ticker, sequence_length, model_path
from src.scaling_data import scale_data
from src.create_sequences import create_sequences

def predict_future(ticker=default_ticker):
    """
    Dự báo giá đóng cửa tương lai dựa trên mô hình đã train.
    Args:
        ticker (str): Mã chứng khoán cần dự báo, mặc định là mã chứng khoán gợi ý.
    Returns:
        list: Dự báo giá đóng cửa tương lai.
    """
    global model_path, sequence_length
    model_file = f"{model_path}/lstm_model_{ticker}.pth"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Không tìm thấy mô hình tại {model_file}. Vui lòng train trước.")

    # Tải dữ liệu thời gian thực từ Stooq
    data = load_realtime_data(ticker)
    features = data[['Close']].values
    scaled_data, scaler = scale_data(features)

    X, _ = create_sequences(scaled_data, sequence_length)
    X = torch.tensor(X).float()

    model = LSTMModel()
    model.load_state_dict(torch.load(model_file))
    model.eval()

    test_input = X[-1].unsqueeze(0)
    pred_scaled = model(test_input).item()
    predicted_price = scaler.inverse_transform([[pred_scaled]])[0][0]
    latest_price = features[-1][0]
    print(f"Giá hiện tại: {latest_price:.2f}, Dự báo giá tiếp theo: {predicted_price:.2f}")

if __name__ == "__main__":
    predict_future()