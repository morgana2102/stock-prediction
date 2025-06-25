import pandas as pd
import requests
from io import StringIO
from src.config import data_source

def format_ticker(ticker, data_source="https://stooq.com"):
    """Định dạng mã chứng khoán theo chuẩn của data_source."""
    if data_source == "https://stooq.com":
        if ticker.lower().endswith(".us") or ticker.lower().endswith("-usd"):
            return ticker.lower()
        return f"{ticker.lower()}.us"
    return ticker

def load_realtime_data(ticker, data_source="https://stooq.com", interval='d'):
    """Tải dữ liệu từ data_source (Stooq) theo thời gian thực."""
    ticker = format_ticker(ticker)
    url = f"{data_source}/q/d/l/?s={ticker}&i={interval}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Lỗi tải dữ liệu từ data_source")

    data = pd.read_csv(StringIO(response.text))
    data.rename(columns={"Date": "Date", "Close": "Close", "Open": "Open", "High": "High", "Low": "Low", "Volume": "Volume"}, inplace=True)
    data.sort_values(by="Date", inplace=True)
    return data

def fetch_ticker_list_stooq(url="https://stooq.com/t/us.txt"):
    """Lấy danh sách ticker từ Stooq."""
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Không thể tải danh sách ticker từ Stooq.")

    tickers = [line.strip() for line in response.text.splitlines() if line.strip()]
    return tickers

if __name__ == "__main__":
    # Ví dụ sử dụng
    ticker = "AAPL"  # Mã chứng khoán Apple trên sàn Mỹ
    data = load_realtime_data(ticker)
    print(data.head())  # In 5 dòng đầu tiên của dữ liệu
