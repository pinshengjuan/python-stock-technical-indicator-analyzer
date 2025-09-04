import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple
import gc
from signal_helper_utilities.utils import wilder_smoothing

class Indicator(ABC):
    """Abstract base class for technical indicators."""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, *args) -> pd.DataFrame:
        """Calculate the indicator values."""
        pass
    
    @abstractmethod
    def get_params(self, config: dict) -> tuple:
        """Extract parameters from config."""
        pass

class SMAIndicator(Indicator):
    def calculate(self, data: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
        """Calculate Simple Moving Averages for specified periods."""
        sma_df = pd.DataFrame(index=data['Close'].index)
        for period in periods:
            sma_df[f'sma_{period}'] = data['Close'].rolling(window=period).mean()
        return sma_df
    
    def get_params(self, config: dict) -> tuple:
        return ([config.get('periods')],)

class RSIIndicator(Indicator):
    def calculate(self, data: pd.DataFrame, periods: int) -> pd.DataFrame:
        """Calculate Relative Strength Index for the given period."""
        if len(data) <= periods:
            raise ValueError(f"Data length ({len(data)}) must be > periods ({periods})")
        delta: pd.Series = data['Close'].diff()
        gain: pd.Series = delta.where(delta > 0, 0).rolling(window=periods).mean()
        loss: pd.Series = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs: pd.Series = gain / loss
        rsi_values: pd.Series = 100 - (100 / (1 + rs))
        rsi_values = rsi_values.where(loss != 0, 100.0)
        del delta, gain, loss, rs
        gc.collect()
        return pd.DataFrame({f'rsi_{periods}': rsi_values})
    
    def get_params(self, config: dict) -> tuple:
        return (config.get('periods'),)

class VolumeIndicator(Indicator):
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract volume data."""
        return pd.DataFrame({'Volume': data['Volume']})
    
    def get_params(self, config: dict) -> tuple:
        return ()

class ADXIndicator(Indicator):
    def calculate(self, data: pd.DataFrame, periods: int) -> pd.DataFrame:
        """Calculate Average Directional Index for the given period."""
        tr_df = pd.DataFrame(index=data['High'].index)
        tr_df['h_l'] = (data['High'] - data['Low']).round(2)
        tr_df['h_pc'] = abs((data['High'] - data['Close'].shift(1)).round(2))
        tr_df['l_pc'] = abs((data['Low'] - data['Close'].shift(1)).round(2))
        tr_df['tr'] = tr_df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
        atr_series = wilder_smoothing(tr_df['tr'], periods)
        
        dm_plus: pd.Series = data['High'] - data['High'].shift(1)
        dm_minus: pd.Series = data['Low'].shift(1) - data['Low']
        dm_plus = dm_plus.where((dm_plus > 0) & (dm_plus > dm_minus), 0)
        dm_minus = dm_minus.where((dm_minus > 0) & (dm_minus > dm_plus), 0)
        smoothed_dm_plus = wilder_smoothing(dm_plus, periods)
        smoothed_dm_minus = wilder_smoothing(dm_minus, periods)
        di_plus = (smoothed_dm_plus / atr_series) * 100
        di_minus = (smoothed_dm_minus / atr_series) * 100
        dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
        adx_series = wilder_smoothing(dx.fillna(0), periods)
        del tr_df, atr_series, dm_plus, dm_minus, smoothed_dm_plus, smoothed_dm_minus, di_plus, di_minus, dx
        gc.collect()
        return pd.DataFrame({f'adx_{periods}': adx_series})
    
    def get_params(self, config: dict) -> tuple:
        return (config.get('periods'),)

class MACDIndicator(Indicator):
    def calculate(self, data: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
        """Calculate MACD, signal line, and histogram."""
        ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return pd.DataFrame({
            'macd_line': macd_line,
            'signal_line': signal_line,
            'macd_histogram': macd_histogram,
        })
    
    def get_params(self, config: dict) -> tuple:
        return tuple(config.get('macd_periods', [12, 26, 9]))

class BollingerBandsIndicator(Indicator):
    def calculate(self, data: pd.DataFrame, window: int, deviations: float) -> pd.DataFrame:
        """Calculate Bollinger Bands for the given window and deviations."""
        if window <= 0 or deviations <= 0:
            raise ValueError("Window and deviations must be positive")
        sma = data['Close'].rolling(window=window).mean()
        std_n = data['Close'].rolling(window=window).std()
        upper_band = sma + (std_n * deviations)
        lower_band = sma - (std_n * deviations)
        return pd.DataFrame({
            'upper_band': upper_band,
            'middle_band': sma,
            'lower_band': lower_band,
        })
    
    def get_params(self, config: dict) -> tuple:
        return (config.get('periods'), config.get('deviations', 2))

class StochasticIndicator(Indicator):
    def calculate(self, data: pd.DataFrame, k_period: int, k_smooth_period: int, d_period: int) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        result_df = pd.DataFrame(index=data['High'].index)
        result_df['Lowest_Low'] = data['Low'].rolling(window=k_period).min()
        result_df['Highest_High'] = data['High'].rolling(window=k_period).max()
        result_df['%K'] = 100 * (data['Close'] - result_df['Lowest_Low']) / (result_df['Highest_High'] - result_df['Lowest_Low'])
        result_df['%K'] = result_df['%K'].rolling(window=k_smooth_period).mean()
        result_df['%D'] = result_df['%K'].rolling(window=d_period).mean()
        return result_df.drop(['Lowest_Low', 'Highest_High'], axis=1)
    
    def get_params(self, config: dict) -> tuple:
        return tuple(config.get('stochastic_periods', [14, 3, 3]))

class AroonIndicator(Indicator):
    def calculate(self, data: pd.DataFrame, periods: int) -> pd.DataFrame:
        """Calculate Aroon indicator."""
        aroon_up = pd.Series(index=data['High'].index, dtype=float)
        aroon_down = pd.Series(index=data['Low'].index, dtype=float)
        aroon_osc = pd.Series(index=data['High'].index, dtype=float)
        for i in range(periods, len(data['High'])):
            high_window = data['High'][i-periods:i+1]
            low_window = data['Low'][i-periods:i+1]
            high_idx = high_window.idxmax()
            low_idx = low_window.idxmin()
            periods_since_high = i - data['High'].index.get_loc(high_idx)
            periods_since_low = i - data['Low'].index.get_loc(low_idx)
            aroon_up.iloc[i] = ((periods - periods_since_high) / periods) * 100
            aroon_down.iloc[i] = ((periods - periods_since_low) / periods) * 100
            aroon_osc.iloc[i] = aroon_up.iloc[i] - aroon_down.iloc[i]
        return pd.DataFrame({
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_osc': aroon_osc,
        })
    
    def get_params(self, config: dict) -> tuple:
        return (config.get('periods'),)

INDICATOR_REGISTRY = {
    'sma': SMAIndicator(),
    'rsi': RSIIndicator(),
    'volume': VolumeIndicator(),
    'adx': ADXIndicator(),
    'macd': MACDIndicator(),
    'bollinger-bands': BollingerBandsIndicator(),
    'stochastic': StochasticIndicator(),
    'aroon': AroonIndicator(),
}