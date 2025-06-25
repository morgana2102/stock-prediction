import numpy as np

def create_sequences(data, seq_length):
    """
    Tạo các chuỗi dữ liệu từ dữ liệu đầu vào.
    Args:
        data (np.ndarray): Dữ liệu đầu vào.
        seq_length (int): Chiều dài của chuỗi.
    Returns:
        np.ndarray, np.ndarray: Mảng các chuỗi dữ liệu và nhãn tương ứng.
    """
    xs, ys = [], []

    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

# Example usage:
if __name__ == "__main__":
    from src.data_loader import load_realtime_data
    ticker = "AAPL.US"  # Ví dụ: Apple trên sàn Mỹ 
    data = load_realtime_data(ticker)
    features = data[['Close']].values  # Chỉ lấy cột 'Close' để tạo chuỗi
    sequence_length = 60  # Độ dài chuỗi
    X, y = create_sequences(features, sequence_length)
    print(X.shape)  # Kích thước của X
    print(X[0], y[0])  # In chuỗi đầu tiên