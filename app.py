import logging
from multiprocessing import Pool
import pandas as pd
import numpy as np
import gc
from pydantic import BaseModel
from typing import List
from signal_helper_utilities.utils import load_json, load_env, write_to_file, print_condition, check_vix_condition
from signal_helper_utilities.data_fetching import get_multiple_days_price
from signal_helper_utilities.plot import draw
from indicators import INDICATOR_REGISTRY

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IndicatorCalculationError(Exception):
    """Raised when an indicator calculation fails."""
    pass

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass

class SeriesConfig(BaseModel):
    type: str
    label: str
    y_key: str
    color: str
    legend: bool

class PlotConfig(BaseModel):
    active: bool
    add_price_chart: bool
    subplot_position: int
    h_ratio: int
    series: List[SeriesConfig]

class IndicatorConfig(BaseModel):
    name: str
    periods: int | None = None
    macd_periods: List[int] | None = None
    stochastic_periods: List[int] | None = None
    deviations: float | None = None
    ratios: List[float] | None = None
    overbought: float | None = None
    oversold: float | None = None
    plot: PlotConfig

class Config(BaseModel):
    figure: dict
    tech_indicators: List[IndicatorConfig]

def get_data_for_ticker(ticker: str, config: dict) -> pd.DataFrame:
    """
    Fetches and processes technical indicators for a single ticker with optimized memory usage.
    
    Args:
        ticker (str): The stock ticker symbol.
        config (dict): Configuration dictionary.
    
    Returns:
        pd.DataFrame: DataFrame with processed technical indicators and a 'Ticker' column.
    """
    env = load_env()
    day_count = env["DAYS"]

    # Load price and volume data
    data_set = get_multiple_days_price(ticker, day_count)
    
    # Remove duplicate indices, keeping the last occurrence
    data_set = data_set.loc[~data_set.index.duplicated(keep='last')]
    
    # Check for duplicate columns
    if data_set.columns.duplicated().any():
        logger.warning(f"Duplicate columns found in data for ticker '{ticker}': {data_set.columns[data_set.columns.duplicated()].tolist()}")
        data_set = data_set.loc[:, ~data_set.columns.duplicated(keep='last')]
    
    # Convert numeric columns to float32 for memory efficiency
    for col in ['Close', 'Volume']:
        if col in data_set.columns:
            data_set[col] = data_set[col].astype(np.float32)
    
    tech_data_set = pd.DataFrame(index=data_set.index)
    tech_data_set['Ticker'] = ticker  # Use category type for Ticker
    tech_data_set['Ticker'] = tech_data_set['Ticker'].astype('category')
    tech_data_set['Close'] = data_set['Close']
    # Only include Price_Change if needed for plotting
    if any(ind['name'].lower() == 'volume' and ind['plot']['active'] for ind in config['tech_indicators']):
        tech_data_set['Price_Change'] = data_set['Close'].diff().astype(np.float32)

    tech_data_set['Volume'] = data_set['Volume']

    # Collect active indicators
    active_indicators = [ind for ind in config['tech_indicators'] if ind['plot']['active']]
    
    # Collect SMA periods for active indicators
    sma_periods = [ind['periods'] for ind in active_indicators if ind['name'].lower() == 'sma' and 'periods' in ind]
    
    indicator_dfs = []
    if sma_periods:
        indicator = INDICATOR_REGISTRY['sma']
        sma_df = indicator.calculate(data_set, sma_periods)
        sma_df = sma_df.loc[:, ~sma_df.columns.duplicated(keep='last')]
        # Convert to float32
        sma_df = sma_df.astype(np.float32)
        indicator_dfs.append(sma_df)

    for ind_config in active_indicators:
        name = ind_config['name'].lower()
        if name == 'sma':
            continue
        if name not in INDICATOR_REGISTRY:
            if name not in ('fibonacci'):
                logger.warning(f"Unknown indicator '{name}', skipping")
            continue
        try:
            indicator = INDICATOR_REGISTRY[name]
            params = indicator.get_params(ind_config)
            if None in params and name not in ('macd', 'stochastic', 'volume'):
                logger.warning(f"Missing parameters for '{name}', skipping")
                continue
            y_keys = [s['y_key'] for s in ind_config['plot']['series']]
            df = indicator.calculate(data_set, *params)[y_keys]
            df = df.loc[:, ~df.columns.duplicated(keep='last')]
            # Convert to float32
            df = df.astype(np.float32)
            indicator_dfs.append(df)
        except Exception as e:
            logger.error(f"Error processing '{name}' for ticker '{ticker}': {e}")
            raise IndicatorCalculationError(f"Error in indicator '{name}' for ticker '{ticker}': {e}")

    if indicator_dfs:
        tech_data_set = pd.concat([tech_data_set] + indicator_dfs, axis=1, copy=False)
        tech_data_set = tech_data_set.loc[:, ~tech_data_set.columns.duplicated(keep='last')]

    del data_set
    gc.collect()  # Free memory
    return tech_data_set

def main():
    """
    Main function to run the analysis for multiple tickers with minimal memory usage.
    """
    env = load_env()
    tickers = env["TICKER_LIST"].split(',')
    
    config = load_json('config_plot.json')
    # sentiment_data = load_sentiment_json()
    sentiment_data = load_json('market_sentiment.json')
    vix = get_multiple_days_price('^VIX', env["DAYS"])["Close"].iloc[-1]
    print_condition(vix)

    # Determine the current market sentiment based on VIX
    current_sentiment = next((sent for sent in sentiment_data['market_sentiment'] if check_vix_condition(vix, sent['vix'])), None)
    watch_indicators = current_sentiment['watch_indicator'] if current_sentiment else []

    # Process and plot each ticker sequentially to reduce memory
    for ticker in tickers:
        # logger.info(f"Processing ticker '{ticker}'")
        ticker_df = get_data_for_ticker(ticker, config)
        if not ticker_df.empty:
            # Remove duplicate indices
            ticker_df = ticker_df.loc[~ticker_df.index.duplicated(keep='last')]
            draw(ticker, ticker_df, config, watch_indicators)  # Pass watch_indicators
        else:
            logger.warning(f"No data available for ticker '{ticker}'")
        del ticker_df
        gc.collect()  # Free memory after each ticker

if __name__ == "__main__":
    main()