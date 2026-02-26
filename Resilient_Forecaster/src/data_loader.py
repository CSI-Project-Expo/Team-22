import yfinance as yf
import pandas as pd
import os


def download_stock_data(ticker="AAPL", period="5y"):
    """
    Downloads stock data and saves it to data/raw_stocks.csv
    """
    print(f"Downloading {ticker} data...")

    data = yf.download(ticker, period=period)

    if data.empty:
        raise ValueError("No data downloaded.")

    data = data[['Close']]
    data.reset_index(inplace=True)

    os.makedirs("data", exist_ok=True)
    data.to_csv("data/raw_stocks.csv", index=False)

    print("Saved to data/raw_stocks.csv")
    return data


if __name__ == "__main__":
    download_stock_data()