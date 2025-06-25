from sklearn.preprocessing import MinMaxScaler

def scale_data(data):
    """
    Chuẩn hóa dữ liệu bằng MinMaxScaler.

    Args:
        data (np.ndarray): Dữ liệu đầu vào cần chuẩn hóa.

    Returns:
        np.ndarray: Dữ liệu đã được chuẩn hóa.
        MinMaxScaler: Bộ chuẩn hóa đã được huấn luyện.
    """
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
    return data_scaled, scaler