import os
import threading
import sys
import time
import pandas as pd
import numpy as np
from typing import Any
from dotenv import load_dotenv
import json
from beautifultable import BeautifulTable

# Load environment variables from .env file
load_dotenv()

def spinner(msg="Generating analysis results"):
    """
    Display a spinner while processing.
    This function runs in a separate thread to avoid blocking the main thread.
    """
    spinner_chars = ['|', '/', '-', '\\']
    idx = 0
    while not getattr(threading.current_thread(), "stop", False):
        sys.stdout.write(f'\r{msg} {spinner_chars[idx % len(spinner_chars)]}')
        sys.stdout.flush()
        time.sleep(0.1)
        idx += 1
    sys.stdout.write(f'\r{msg} Done!\n')
    sys.stdout.flush()

def load_json(file):
    try:
        with open(file, 'r', encoding="utf-8") as file:
            config = json.load(file)
            return config
    except FileNotFoundError:
        print("Error: File 'market_sentiment.json' not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")

def load_env() -> dict[str, Any]:
    """
    Load configuration settings from environment variables or defaults.
    Returns:
        dict: Configuration settings.
    """
    return {
        'TICKER_LIST': os.getenv('TICKER_LIST', 'GOOGL'),
        'DAYS': int(os.getenv('DAYS', 365)),
    }

def wilder_smoothing(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Apply Wilder's smoothing to a series.
    Args:
        series (pd.Series): Series to smooth.
        period (int): Period for smoothing.
    Returns:
        pd.Series: Smoothed series.
    """
    smoothed = np.zeros(len(series))
    smoothed[period-1] = series[:period].mean()
    for i in range(period, len(series)):
        smoothed[i] = (smoothed[i-1] * (period-1) + series.iloc[i]) / period
    smoothed[:period-1] = np.nan
    return pd.Series(smoothed, index=series.index)

def write_to_file(data_set: Any, filename: str, mode: str = 'w') -> None:
    """
    Write data to a file.
    Args:
        data_set (Any): Data to write to the file.
        filename (str): Name of the file to write to.
        mode (str): Mode in which to open the file (default is 'w').
    """
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/data/'
    if not os.path.isdir(parent_dir):
        os.mkdir(parent_dir)
    filename: str = parent_dir + filename
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    with open(filename, mode, encoding='utf8') as f:
        f.write(str(data_set))

def print_condition(vix):
    sentiment_data = load_json('market_sentiment.json')
    for idx, ind in enumerate(sentiment_data['market_sentiment']):
        vix_condition = ind.get("vix")
        if vix_condition and check_vix_condition(vix, vix_condition):
            market_type = ind.get("name")
            vix_show = np.round(vix, 2)
            print(f"{market_type} Market, vix: {vix_show}")
            conditions = sentiment_data["market_sentiment"][idx]["condition_description"]

            # Prepare data for the table
            table_data = BeautifulTable()
            for condition in conditions:
                indicator = list(condition.keys())[0]
                requirement = condition[indicator]
                table_data.rows.append([indicator, requirement])
            table_data.columns.header = ["Indicators", "Requirements"]
            table_data.set_style(BeautifulTable.STYLE_SEPARATED)
            print(table_data)

def check_vix_condition(vix_value, vix_condition):
    try:
        # Split the condition into parts (e.g., ">=20", "<30")
        conditions = [cond.strip() for cond in vix_condition.split(",")]
        result = True

        for cond in conditions:
            # Parse each condition
            if cond.startswith(">="):
                threshold = float(cond[2:].strip())
                result = result and vix_value >= threshold
            elif cond.startswith(">"):
                threshold = float(cond[1:].strip())
                result = result and vix_value > threshold
            elif cond.startswith("<="):
                threshold = float(cond[2:].strip())
                result = result and vix_value <= threshold
            elif cond.startswith("<"):
                threshold = float(cond[1:].strip())
                result = result and vix_value < threshold
            elif cond.startswith("="):
                threshold = float(cond[1:].strip())
                result = result and vix_value == threshold
            else:
                return False  # Invalid condition format

        return result
    except ValueError:
        print("Error: Invalid VIX value or condition format")
        return False