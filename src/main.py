import argparse
from src.data_loader import load_data
from src.strategy import apply_sma_strategy, grid_search_sma
from src.predict import predict_future
from src.visualization import *
from src.config import default_ticker, suggested_tickers, short_range, long_range, pic_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Prediction Main Pipeline")
    parser.add_argument('--ticker', type=str, default=default_ticker, help='Mã cổ phiếu')
    parser.add_argument('--data_source', type=str, choices=['Y', 'N'], default='N',
                        help='Y: tải dữ liệu thời gian thực, N: tải dữ liệu từ file đã lưu')
    args = parser.parse_args()

    ticker = args.ticker
    data_source = 'https://stooq.com' if args.data_source.upper() == 'Y' else 'local'

    print("Danh sách ticker gợi ý:")
    print(", ".join(suggested_tickers))
    print(f"Sử dụng ticker: {ticker}")
    print(f"Chọn kiểu dữ liệu: {'Thời gian thực' if data_source == 'https://stooq.com' else 'Local file'}")

    data = load_data(ticker, data_source)

    # Grid Search tìm bộ tham số SMA tối ưu
    sma_param_results = grid_search_sma(data, short_range, long_range)
    print(sma_param_results.head())

    # Áp dụng chiến lược với bộ tham số tốt nhất
    best = sma_param_results.iloc[0]
    apply_sma_strategy(data, short_window=int(best['short_window']), long_window=int(best['long_window']))

    # Vẽ biểu đồ hiệu suất tích luỹ & tỷ suất thông thường của chiến lược SMA
    plot_cumulative_return(data=data,
                           short_window=int(best['short_window']),
                           long_window=int(best['long_window']),
                           save_path=f"{pic_path}/{ticker}/sma_cumulative_return_{ticker}.png")
    plot_daily_return_sma(data=data,
                           short_window=int(best['short_window']),
                           long_window=int(best['long_window']),
                           save_path=f"{pic_path}/{ticker}/sma_daily_return_{ticker}.png")

    # Dự báo giá tiếp theo sử dụng mô hình đã lưu
    predict_future(ticker=ticker)

    # Vẽ biểu đồ giá thực tế và giá dự báo từ mô hình LSTM
    plot_stock_price_lstm(data=data, 
                          ticker=ticker,
                          save_path=f"{pic_path}/{ticker}/rnn_lstm_prediction_{ticker}.png")

    # Thông báo hoàn thành
    print(f"Đã hoàn thành các bước phân tích và dự báo cho ticker: {ticker}")
    print(f"Các biểu đồ đã được lưu tại: {pic_path}/{ticker}/")