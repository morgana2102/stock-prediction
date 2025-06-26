import torch
import torch.optim as optim
import torch.nn as nn
import os
from tqdm import trange, tqdm
from src.config import model_path

def train_model(model, X, y, num_epochs=100, lr=0.001):
    """    
    Huấn luyện mô hình RNN + LSTM với dữ liệu đầu vào.
    Args:
        model (torch.nn.Module): Mô hình RNN + LSTM đã được định nghĩa.
        X (torch.Tensor): Dữ liệu đầu vào đã được tạo chuỗi.
        y (torch.Tensor): Giá trị mục tiêu tương ứng với X.
        num_epochs (int): Số lượng epoch để huấn luyện mô hình.
        lr (float): Tốc độ học của bộ tối ưu hóa.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in trange(num_epochs, desc="Training", unit="epoch"):
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 20 == 0  or epoch == 0 or epoch == num_epochs-1:
            tqdm.write(f"Epoch [{epoch+1 if epoch > 0 else 0}/{num_epochs}], Loss: {loss.item():.4f}")


def train_and_save(model, X, y, save_path=model_path, num_epochs=100, lr=0.001):
    """    
    Huấn luyện mô hình RNN + LSTM và lưu mô hình đã huấn luyện vào file.
    Args:
        model (torch.nn.Module): Mô hình RNN + LSTM đã được định nghĩa.
        X (torch.Tensor): Dữ liệu đầu vào đã được tạo chuỗi.
        y (torch.Tensor): Giá trị mục tiêu tương ứng với X.
        save_path (str): Đường dẫn để lưu mô hình đã huấn luyện.
        num_epochs (int): Số lượng epoch để huấn luyện mô hình.
        lr (float): Tốc độ học của bộ tối ưu hóa.
    Returns:
        model (torch.nn.Module): Mô hình RNN + LSTM đã được huấn luyện.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    train_model(model, X, y, num_epochs=num_epochs, lr=lr)
    torch.save(model.state_dict(), save_path)

    return model

if __name__ == "__main__":
    import argparse
    import src.config as config
    from src.data_loader import load_data
    from src.model import RNN_LSTMModel
    from src.create_sequences import create_sequences
    from src.scaling_data import scale_data
    from src.config import default_ticker, suggested_tickers, sequence_length, model_path, pic_path
    from src.visualization import plot_stock_price_lstm

    # Thiết lập parser cho các tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Stock Price Prediction Config")

    parser.add_argument('--ticker', type=str, default=config.ticker, help='Mã cổ phiếu')
    parser.add_argument('--data_source', type=str, choices=['Y', 'N'], default='N',
                        help='Y: tải dữ liệu thời gian thực, N: tải dữ liệu từ file đã lưu')
    parser.add_argument('--sequence_length', type=int, default=config.sequence_length, help='Chiều dài chuỗi LSTM')

    args = parser.parse_args()

    # Cập nhật config bằng giá trị truyền vào
    config.ticker = args.ticker
    config.data_source = 'https://stooq.com' if args.data_source == 'Y' else 'local'
    config.sequence_length = args.sequence_length

    print(f"Đang chạy với ticker: {config.ticker}, Nguồn dữ liệu: {config.data_source}, Chiều dài chuỗi: {config.sequence_length}")

    # # Tải dữ liệu và chuẩn bị chuỗi
    # print("Danh sách ticker gợi ý:")
    # print(", ".join(suggested_tickers))
    # user_ticker = input(f"Nhập ticker (mặc định {default_ticker}): ").strip()
    # ticker = user_ticker if user_ticker else default_ticker
    # print(f"Sử dụng ticker: {ticker}")

    # print('Chọn kiểu dữ liệu: Y nếu muốn train bằng dữ liệu thời gian thực,' \
    # ' N nếu muốn train bằng dữ liệu từ file đã lưu.')
    # data_source = input("Nhập Y hoặc N: ").strip().upper()
    # data_source = 'https://stooq.com' if data_source == 'Y' else 'local'
    data = load_data(config.ticker, config.data_source)
    features = data[['Close']].values
    scaled_features, scaler = scale_data(features)
    X, y = create_sequences(scaled_features, config.sequence_length)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    # Khởi tạo mô hình và huấn luyện
    model = RNN_LSTMModel()
    save_path = f'{config.model_path}/model_{config.ticker.lower()}.pth'
    train_and_save(model, X, y, save_path)

    # Thông báo hoàn thành
    print(f"Huấn luyện mô hình hoàn tất và đã lưu tại {save_path}")
    print("Bạn có thể sử dụng mô hình này để dự báo giá trong tương lai.")

    # Vẽ biểu đồ giá thực tế và giá dự báo từ mô hình RNN+LSTM
    plot_stock_price_lstm(data, config.ticker, save_path=f"{pic_path}/{config.ticker}/rnn_lstm_prediction_{config.ticker}.png")