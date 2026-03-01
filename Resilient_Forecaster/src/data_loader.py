import yfinance as yf      # Imports yfinance library to download stock market data
import pandas as pd        # Imports pandas library for data manipulation and analysis
import os                  # Imports os module to work with folders and file paths


def download_stock_data(ticker="AAPL", period="5y"):
    """
    Downloads stock data and saves it to data/raw_stocks.csv
    """
    # Function definition:
    # ticker="AAPL" → Default stock symbol is Apple
    # period="5y" → Default period is last 5 years

    print(f"Downloading {ticker} data...")  
    # Prints which stock data is being downloaded

    data = yf.download(ticker, period=period)  
    # Downloads stock data using yfinance
    # Stores the result in a pandas DataFrame called 'data'

    if data.empty:
        # Checks if no data was downloaded
        raise ValueError("No data downloaded.")
        # Raises an error if the dataset is empty

    data = data[['Close']]
    # Keeps only the 'Close' column from the stock data
    # 'Close' = Closing price of stock each day

    data.reset_index(inplace=True)
    # Resets the index so Date becomes a normal column
    # inplace=True means modify the same DataFrame

    os.makedirs("data", exist_ok=True)
    # Creates a folder named "data" if it does not already exist
    # exist_ok=True prevents error if folder already exists

    data.to_csv("data/raw_stocks.csv", index=False)
    # Saves the DataFrame to a CSV file
    # index=False means do not save row numbers

    print("Saved to data/raw_stocks.csv")
    # Prints confirmation message

    return data
    # Returns the DataFrame so it can be used elsewhere


if __name__ == "__main__":
    # This block runs only if the file is executed directly
    # (Not when imported as a module)

    download_stock_data()
    # Calls the function with default values (AAPL, 5y)