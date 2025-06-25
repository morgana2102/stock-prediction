import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import os

import torch
from src.config import pic_path, model_path
from src.create_sequences import create_sequences
from src.model import LSTMModel
from src.scaling_data import scale_data
from src.strategy import apply_sma_strategy
from src.config import sequence_length

def plot_cumulative_return(data, short_window, long_window, save_path=f"{pic_path}/sma_strategy_performance.png"):
    """    
    Vẽ biểu đồ hiệu suất tích luỹ của chiến lược SMA.
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu với cột 'Close'.
        short_window (int): Kỳ ngắn hạn của SMA.
        long_window (int): Kỳ dài hạn của SMA.
        save_path (str): Đường dẫn lưu biểu đồ.
    """
    cumulative_market, cumulative_strategy = apply_sma_strategy(data, short_window, long_window)

    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_market, label='Thị trường')
    plt.plot(cumulative_strategy, label='Chiến lược SMA')
    plt.title(f'Hiệu suất tích luỹ (SMA {short_window}/{long_window})')
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_daily_return_sma(data, short_window, long_window, save_path=f"{pic_path}/sma_daily_return.png"):
    """    
    Vẽ biểu đồ tỷ suất sinh lời từng ngày của chiến lược SMA so với thị trường.
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu với cột 'Close'.
        short_window (int): Kỳ ngắn hạn của SMA.
        long_window (int): Kỳ dài hạn của SMA.
        save_path (str): Đường dẫn lưu biểu đồ.
    """
    df = data.copy()
    df['Market_Return'] = df['Close'].pct_change()
    df['SMA_short'] = df['Close'].rolling(short_window).mean()
    df['SMA_long'] = df['Close'].rolling(long_window).mean()
    df.dropna(inplace=True)

    df['Signal'] = np.where(df['SMA_short'] > df['SMA_long'], 1, 0)
    df['Position'] = df['Signal'].shift(1)
    df['Strategy_Return'] = df['Position'] * df['Market_Return']

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Market_Return'], label="Tỷ suất sinh lời thị trường", alpha=0.5)
    plt.plot(df.index, df['Strategy_Return'], label="Tỷ suất sinh lời chiến lược SMA", alpha=0.7)
    plt.title("So sánh tỷ suất sinh lời từng ngày (Market vs SMA)")
    plt.legend()
    plt.grid()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_stock_price_lstm(data, ticker, save_path=f"{pic_path}/lstm_prediction.png"):
    """
    Vẽ biểu đồ giá thực tế và giá dự báo từ mô hình LSTM.
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu với cột 'Date' và 'Close'.
        predicted_prices (list or np.array): Giá dự báo từ mô hình LSTM.
        save_path (str): Đường dẫn lưu biểu đồ.
    """
    global sequence_length, model_path
    
    df = data.copy().reset_index(drop=True)

    # Chỉ lấy cột 'Close' để dự báo
    features = df[['Close']].values
    scaled_data, scaler = scale_data(features)

    X, _ = create_sequences(scaled_data, sequence_length)
    X = torch.tensor(X).float()

    # Tải mô hình LSTM đã huấn luyện
    if not os.path.exists(f"{model_path}/lstm_model_{ticker}.pth"):
        raise FileNotFoundError(f"Không tìm thấy mô hình tại {model_path}/lstm_model_{ticker}.pth. Vui lòng train trước.")
    
    # Khởi tạo mô hình LSTM
    model = LSTMModel()

    # Tải trọng số mô hình đã huấn luyện
    model.load_state_dict(torch.load(f"{model_path}/lstm_model_{ticker}.pth"))
    model.eval()
    predicted_all = model(X).detach().numpy()
    predicted_all = scaler.inverse_transform(predicted_all)

    dates = pd.to_datetime(df['Date']).values[-len(predicted_all):]
    real_price = df['Close'].values[-len(predicted_all):]
    predicted_all = np.array(predicted_all).flatten()

    plt.figure(figsize=(12, 8))
    plt.plot(dates, real_price, label="Giá thực tế")
    plt.plot(dates, predicted_all, label="Giá dự báo LSTM")
    plt.title("So sánh giá thực tế và giá dự báo từ LSTM")
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=5))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.xticks(rotation=45)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    from src.data_loader import load_realtime_data
    from src.config import default_ticker, pic_path

    # Tải dữ liệu và vẽ biểu đồ
    ticker = default_ticker
    data = load_realtime_data(ticker)
    
    # Vẽ biểu đồ hiệu suất tích luỹ & tỷ suất thông thường của chiến lược SMA
    plot_cumulative_return(data=data, short_window=20, long_window=100, save_path=f"{pic_path}/{ticker}/sma_cumulative_return.png")
    plot_daily_return_sma(data=data, short_window=20, long_window=100, save_path=f"{pic_path}/{ticker}/sma_daily_return.png")
    
    # Vẽ biểu đồ giá thực tế và giá dự báo từ mô hình LSTM
    plot_stock_price_lstm(data=data, ticker=ticker, save_path=f"{pic_path}/{ticker}/lstm_prediction.png")