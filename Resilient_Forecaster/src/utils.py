import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def load_data(path):
    df = pd.read_csv(path)

    # Force Close column to numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    # Drop any NaN rows
    df = df.dropna()

    return df[['Close']]


def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data.values.astype(float))
    return scaled, scaler


def create_sequences(data, window_size=60):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y)


def plot_predictions(actual, predicted, title="Stock Prediction"):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Predicted")
    plt.title(title)
    plt.legend()
    plt.show()


def calculate_mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)


def remove_outliers_iqr(data):
    """
    Removes outliers using IQR method.
    Works well for poisoned spikes.
    """

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

    return filtered_data