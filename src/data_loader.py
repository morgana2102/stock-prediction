import pandas as pd
import requests
from io import StringIO
from src.config import data_source

def format_ticker(ticker, data_source="https://stooq.com"):
    """Định dạng mã chứng khoán theo chuẩn của data_source."""
    if data_source == "https://stooq.com":
        if ticker.lower().endswith(".us") or ticker.lower().endswith("-usd"):
            return ticker.lower()
        else:
            return f'{ticker.lower()}.us'
    elif data_source == "local":
        if ticker.lower().endswith(".us") or ticker.lower().endswith("-usd"):
            return ticker.lower()
        else:
            return f'{ticker.lower()}_us'  # Thêm đuôi _us cho dữ liệu cục bộ
    else:
        raise ValueError("Nguồn dữ liệu không hợp lệ. Chỉ hỗ trợ 'https://stooq.com' hoặc 'local'.")

def load_realtime_data(ticker, data_source="https://stooq.com", interval='d'):
    """Tải dữ liệu từ data_source (Stooq) theo thời gian thực."""
    ticker = format_ticker(ticker)
    url = f"{data_source}/q/d/l/?s={ticker}&i={interval}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Lỗi tải dữ liệu từ Stooq")

    data = pd.read_csv(StringIO(response.text))
    data.rename(columns={"Date": "Date", "Close": "Close", "Open": "Open", "High": "High", "Low": "Low", "Volume": "Volume"}, inplace=True)
    data.sort_values(by="Date", inplace=True)
    return data

def load_data(ticker, data_source):
    ticker = format_ticker(ticker, data_source)
    if data_source == "https://stooq.com":
        return load_realtime_data(ticker, data_source)
    elif data_source == "local":
        file_path = f"data/{ticker}.csv"
        try:
            data = pd.read_csv(file_path)
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            return data
        except FileNotFoundError:
            raise Exception(f"Không tìm thấy dữ liệu cục bộ cho ticker {ticker}")
    else:
        raise ValueError("Nguồn dữ liệu không hợp lệ. Chỉ hỗ trợ 'https://stooq.com' hoặc 'local'.")

if __name__ == "__main__":
    # Ví dụ sử dụng
    ticker = "googl"  # Mã chứng khoán Apple trên sàn Mỹ
    data = load_data(ticker, data_source='https://stooq.com')
    print(data.head())  # In 5 dòng đầu tiên của dữ liệu
