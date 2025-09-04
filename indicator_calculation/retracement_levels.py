import pandas as pd

def fib_retracement(data: pd.DataFrame, periods: int = 90, fib_ratios: list = [0, 0.382, 0.5, 0.618, 1]) -> tuple:
    """
    Calculate Fibonacci retracement levels for the stock.
    
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        periods (int): Number of periods for calculation (default 90).
        fib_ratios (list[float]): Fibonacci ratios (default [0, 0.382, 0.5, 0.618, 1]).
    
    Returns:
        tuple: (fib_levels, fib_labels, fib_start_date, fib_end_date)
    """
    recent_df = data.tail(periods)
    high_price = recent_df['Close'].max()
    low_price = recent_df['Close'].min()
    fib_levels = [low_price + (high_price - low_price) * ratio for ratio in fib_ratios]
    fib_labels = [f'{ratio} (${level:.2f})' for ratio, level in zip(fib_ratios, fib_levels)]
    fib_start_date = recent_df.index[0].strftime('%Y-%m-%d')
    fib_end_date = recent_df.index[-1].strftime('%Y-%m-%d')
    return fib_levels, fib_labels, fib_start_date, fib_end_date