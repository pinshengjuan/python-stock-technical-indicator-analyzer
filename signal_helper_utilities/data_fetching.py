import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def get_multiple_days_price(ticker: str, days: int) -> pd.DataFrame:
    """
    Fetch historical stock data for a given ticker and number of days.
    Args:
        ticker (str): Stock ticker symbol.
        days (int): Number of days of historical data to fetch. 
    Returns:
        pd.DataFrame: DataFrame containing historical stock data.
    """
    end_date: datetime = datetime.now()
    start_date: datetime = end_date - timedelta(days=days)
    stock = yf.Ticker(ticker)
    try:
        data_set = stock.history(start=start_date, end=end_date)
    except ValueError:
        data_set = stock.history(period='max')
    return data_set