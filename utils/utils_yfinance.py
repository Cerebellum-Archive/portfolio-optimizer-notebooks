# utils/utils_yfinance.py
"""
Utility functions for fetching and processing data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import numpy as np

def log_returns(df):
    """Calculate log returns from price data."""
    return np.log(df) - df.shift(1)

def fetch_yfinance_returns(tickers, start_date, end_date=None):
    """
    Downloads closing prices for a list of tickers from Yahoo Finance
    and returns a DataFrame of daily log returns.

    Parameters
    ----------
    tickers : list of str
        A list of ticker symbols to download.
    start_date : str
        The start date for the data in 'YYYY-MM-DD' format.
    end_date : str, optional
        The end date for the data in 'YYYY-MM-DD' format. 
        If None, downloads up to the most recent date.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the daily log returns for each ticker.
    """
    print(f"Downloading data for {len(tickers)} tickers from {start_date}...")
    
    # Download the closing prices
    try:
        close_prices = yf.download(tickers, start=start_date, end=end_date)['Close']
    except Exception as e:
        print(f"An error occurred during download: {e}")
        return pd.DataFrame() # Return empty dataframe on error

    # Handle the case where only one ticker is downloaded (returns a Series)
    if isinstance(close_prices, pd.Series):
        close_prices = close_prices.to_frame(name=tickers[0])

    # Calculate log returns
    returns_df = log_returns(close_prices)
    
    # Drop the first row which will be NaN after shift
    return returns_df.dropna()

if __name__ == '__main__':
    # Example usage: This block will run if the script is executed directly
    
    # Define a list of tickers and a date range
    sample_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    start = '2022-01-01'
    end = '2022-12-31'
    
    # Fetch the returns data
    returns_data = fetch_yfinance_returns(sample_tickers, start_date=start, end_date=end)
    
    # Print the first few rows of the result
    if not returns_data.empty:
        print("\nSuccessfully fetched and processed data.")
        print("Head of returns DataFrame:")
        print(returns_data.head())
