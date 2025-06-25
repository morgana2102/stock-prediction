import torch
import torch.optim as optim
import torch.nn as nn
import os
from src.config import model_path

def train_model(model, X, y, num_epochs=100, lr=0.001):
    """    
    Huấn luyện mô hình LSTM với dữ liệu đầu vào.
    Args:
        model (torch.nn.Module): Mô hình LSTM đã được định nghĩa.
        X (torch.Tensor): Dữ liệu đầu vào đã được tạo chuỗi.
        y (torch.Tensor): Giá trị mục tiêu tương ứng với X.
        num_epochs (int): Số lượng epoch để huấn luyện mô hình.
        lr (float): Tốc độ học của bộ tối ưu hóa.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


def train_and_save(model, X, y, save_path=model_path, num_epochs=100, lr=0.001):
    """    
    Huấn luyện mô hình LSTM và lưu mô hình đã huấn luyện vào file.
    Args:
        model (torch.nn.Module): Mô hình LSTM đã được định nghĩa.
        X (torch.Tensor): Dữ liệu đầu vào đã được tạo chuỗi.
        y (torch.Tensor): Giá trị mục tiêu tương ứng với X.
        save_path (str): Đường dẫn để lưu mô hình đã huấn luyện.
        num_epochs (int): Số lượng epoch để huấn luyện mô hình.
        lr (float): Tốc độ học của bộ tối ưu hóa.
    Returns:
        model (torch.nn.Module): Mô hình LSTM đã được huấn luyện.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    train_model(model, X, y, num_epochs=num_epochs, lr=lr)
    torch.save(model.state_dict(), save_path)
    print(f"Đã lưu mô hình vào {save_path}")

    return model

if __name__ == "__main__":
    from src.data_loader import load_realtime_data
    from src.model import LSTMModel
    from src.create_sequences import create_sequences
    from src.scaling_data import scale_data
    from src.config import default_ticker, suggested_tickers, sequence_length, model_path

    # Tải dữ liệu và chuẩn bị chuỗi
    print("Danh sách ticker gợi ý:")
    print(", ".join(suggested_tickers))
    user_ticker = input(f"Nhập ticker (mặc định {default_ticker}): ").strip()
    ticker = user_ticker if user_ticker else default_ticker
    print(f"Sử dụng ticker: {ticker}")
    
    data = load_realtime_data(ticker)
    features = data[['Close']].values
    scaled_features, scaler = scale_data(features)
    X, y = create_sequences(scaled_features, sequence_length)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    # Khởi tạo mô hình và huấn luyện
    model = LSTMModel()
    save_path = f'{model_path}/lstm_model_{ticker}.pth'
    train_and_save(model, X, y, save_path)

    # Thông báo hoàn thành
    print(f"Huấn luyện mô hình hoàn tất và đã lưu tại {save_path}")
    print("Bạn có thể sử dụng mô hình này để dự báo giá trong tương lai.")