import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm

def apply_sma_strategy(data, short_window=20, long_window=100):
    """    
    Áp dụng chiến lược SMA (Simple Moving Average) trên dữ liệu giá cổ phiếu
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu với cột 'Close'.
        short_window (int): Kỳ hạn SMA ngắn hạn.
        long_window (int): Kỳ hạn SMA dài hạn.
    Returns:
        pd.Series, pd.Series: Hai chuỗi dữ liệu chứa hiệu suất tích luỹ của thị trường và chiến lược SMA.
    """
    data = data.copy()
    data['SMA_short'] = data['Close'].rolling(short_window).mean()
    data['SMA_long'] = data['Close'].rolling(long_window).mean()
    data.dropna(inplace=True)

    data['Signal'] = np.where(data['SMA_short'] > data['SMA_long'], 1, 0)
    data['Position'] = data['Signal'].shift(1)
    data['Market_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Position'] * data['Market_Return']

    cumulative_market = (1 + data['Market_Return']).cumprod()
    cumulative_strategy = (1 + data['Strategy_Return']).cumprod()

    return cumulative_market, cumulative_strategy

def grid_search_sma(data, short_range, long_range):
    """    
    Thực hiện tìm kiếm lưới (grid search) để tìm bộ tham số SMA tối ưu
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu với cột 'Close'.
        short_range (range): Khoảng giá trị cho kỳ hạn SMA ngắn hạn.
        long_range (range): Khoảng giá trị cho kỳ hạn SMA dài hạn.
    Returns:
        pd.DataFrame: DataFrame chứa kết quả tìm kiếm lưới với các cột 'short_window', 'long_window', và 'final_return'.
    """
    results = []
    for short_w in tqdm(short_range):
        for long_w in long_range:
            if short_w >= long_w:
                continue
            df = data.copy()
            df['SMA_short'] = df['Close'].rolling(short_w).mean()
            df['SMA_long'] = df['Close'].rolling(long_w).mean()
            df.dropna(inplace=True)

            df['Signal'] = np.where(df['SMA_short'] > df['SMA_long'], 1, 0)
            df['Position'] = df['Signal'].shift(1)
            df['Market_Return'] = df['Close'].pct_change()
            df['Strategy_Return'] = df['Position'] * df['Market_Return']
            cumulative_return = (1 + df['Strategy_Return']).cumprod()

            final_return = cumulative_return.iloc[-1]
            results.append({"short_window": short_w, "long_window": long_w, "final_return": final_return})
    return pd.DataFrame(results).sort_values(by="final_return", ascending=False)

if __name__ == "__main__":
    import argparse
    import src.config as config
    from src.data_loader import load_data

    # print("Danh sách ticker gợi ý:")
    # print(", ".join(suggested_tickers))
    # user_ticker = input(f"Nhập ticker (mặc định {default_ticker}): ").strip()
    # ticker = user_ticker if user_ticker else default_ticker
    # print(f"Sử dụng ticker: {ticker}")

    # print('Chọn kiểu dữ liệu: Y nếu muốn tải dữ liệu thời gian thực, N nếu muốn tải dữ liệu từ file đã lưu.')
    # data_source = input("Nhập Y hoặc N: ").strip().upper()
    # data_source = 'https://stooq.com' if data_source == 'Y' else 'local'

    # Thiết lập parser cho các tham số dòng lệnh
    parser = argparse.ArgumentParser(description="SMA Strategy Config")
    parser.add_argument('--ticker', type=str, default=config.ticker, help='Mã cổ phiếu')
    parser.add_argument('--data_source', type=str, choices=['Y', 'N'], default='N',
                        help='Y: tải dữ liệu thời gian thực, N: tải dữ liệu từ file đã lưu')
    parser.add_argument('--short_range', type=int, nargs='+', default=config.short_range, help='Khoảng giá trị cho kỳ hạn SMA ngắn hạn')
    parser.add_argument('--long_range', type=int, nargs='+', default=config.long_range, help='Khoảng giá trị cho kỳ hạn SMA dài hạn')
    args = parser.parse_args()

    # Cập nhật config bằng giá trị truyền vào
    config.ticker = args.ticker
    config.data_source = 'https://stooq.com' if args.data_source == 'Y' else 'local'
    config.short_range = args.short_range
    config.long_range = args.long_range

    print(f"Đang chạy với ticker: {config.ticker}, Nguồn dữ liệu: {config.data_source}, Kỳ hạn SMA ngắn hạn: {config.short_range}, Kỳ hạn SMA dài hạn: {config.long_range}")

    data = load_data(config.ticker, config.data_source)

    # Grid Search tìm bộ tham số SMA tối ưu
    sma_param_results = grid_search_sma(data, config.short_range, config.long_range)
    print(sma_param_results.head())

    # In ra bộ tham số tốt nhất
    best = sma_param_results.iloc[0]
    print(f"Bộ tham số tốt nhất: SMA ngắn hạn = {best['short_window']}, SMA dài hạn = {best['long_window']}, Lợi nhuận cuối cùng = {best['final_return']:.4f}")