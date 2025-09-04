import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
from indicator_calculation.retracement_levels import fib_retracement
import gc
import warnings
from typing import List

class LatestValueLabelStrategy:
    """
    Strategy to add the latest value as a y-tick label on the right y-axis (ax.twinx()).
    """
    def apply(self, ax, df, series: dict, indicator: dict, fib_data):
        """
        Apply the latest value label to the right y-axis.
        
        Args:
            ax: Matplotlib axis to plot on.
            df: DataFrame with indicator data.
            series: Series configuration from config.json.
            indicator: Indicator configuration from config.json.
        """
        y_key = series.get('y_key', '')
        ticker = df.get('Ticker', 'unknown').iloc[0]
        name = indicator['name'].lower()
        color = series.get('color', 'gray')

        if name != 'fibonacci' and y_key not in df.columns:
            print(f"Warning: y_key '{y_key}' not found in DataFrame columns for ticker '{ticker}'.")
            return

        # Create right y-axis
        ax_right = ax.twinx()
        ax_right.set_ylim(ax.get_ylim())

        if name == 'fibonacci' and fib_data is not None:
            fib_levels, fib_labels, _, _ = fib_data
            ax_right.set_yticks(fib_levels)
            ax_right.set_yticklabels(fib_labels)
            ax_right.set_ylabel('Fibonacci Levels', color=color)
        else:
            # Get the latest value
            last_value = df[y_key].dropna().iloc[-1]
            if isinstance(last_value, pd.Series):
                print(f"Warning: Multiple values for last index in y_key '{y_key}' for ticker '{ticker}', using first.")
                last_value = last_value.iloc[0]

            # Format the label based on indicator type
            label = self._format_label(name, last_value)

            if name == 'volume':
                twenty_days_avg = df[y_key].rolling(window=20).mean().iloc[-1]
                twenty_days_avg_label = self._format_label(name, twenty_days_avg)
                twenty_days_avg_label = twenty_days_avg_label.replace('(last)', '(20-days avg.)')
                ax_right.set_yticks([last_value, twenty_days_avg])
                ax_right.set_yticklabels([label, twenty_days_avg_label])
            else:
                ax_right.set_yticks([last_value])
                ax_right.set_yticklabels([label])
        ax_right.tick_params(axis='y', labelcolor=color, labelsize=6)

    def _format_label(self, indicator_name: str, value: float) -> str:
        """
        Format the latest value based on the indicator type.
        
        Args:
            indicator_name (str): Name of the indicator (e.g., 'volume', 'rsi', 'close').
            value (float): The latest value to format.
        
        Returns:
            str: Formatted label string.
        """
        if indicator_name == 'volume':
            if value >= 1e6:
                return f'{value/1e6:.1f}M (last)'
            return f'{value/1e3:.1f}K (last)'
        elif indicator_name in ('rsi', 'stochastic'):
            return f'{value:.1f}'
        elif indicator_name in ('close', 'sma', 'bollinger-bands', 'fibonacci'):
            return f'${value:.2f}'
        else:
            return f'{value:.2f}'  # Default formatting

class PlotStrategy:
    """Base class for plotting strategies."""
    def plot(self, ax, df, series: dict, indicator: dict, x_index):
        pass

class LinePlotStrategy(PlotStrategy):
    def plot(self, ax, df, series: dict, indicator: dict, x_index):
        y_key = series.get('y_key', '')
        if y_key not in df.columns:
            print(f"Warning: y_key '{y_key}' not found in DataFrame columns for ticker '{df.get('Ticker', 'unknown').iloc[0]}'.")
            return
        y_data = df[y_key]
        if y_data.ndim > 1:
            print(f"Warning: y_key '{y_key}' returned multi-dimensional data for ticker '{df.get('Ticker', 'unknown').iloc[0]}', using first column.")
            y_data = y_data.iloc[:, 0] if isinstance(y_data, pd.DataFrame) else y_data
        sns.lineplot(
            x=x_index,
            y=y_data,
            ax=ax,
            color=series.get('color', 'blue'),
            label=series.get('label', '') if series.get('legend', True) else '_nolegend_',
            linewidth=0.8
        )

class BarPlotStrategy(PlotStrategy):
    def plot(self, ax, df, series: dict, indicator: dict, x_index):
        name = indicator['name'].lower()
        ticker = df.get('Ticker', 'unknown').iloc[0]
        y_key = series.get('y_key', '')
        if y_key not in df.columns:
            print(f"Warning: y_key '{y_key}' not found in DataFrame columns for ticker '{ticker}'.")
            return
        y_data = df[y_key]
        if y_data.ndim > 1:
            print(f"Warning: y_key '{y_key}' returned multi-dimensional data for ticker '{ticker}', using first column.")
            y_data = y_data.iloc[:, 0] if isinstance(y_data, pd.DataFrame) else y_data
        colors = [series.get('color', 'blue')] * len(y_data)
        if name == 'volume':
            colors = ['green' if change > 0 else 'red' for change in df['Price_Change']]
        elif name == 'macd':
            colors = ['green' if h > 0 else 'red' for h in y_data]
        ax.bar(
            x_index,
            y_data,
            color=colors,
            label=series.get('label', '') if series.get('legend', True) else '_nolegend_',
            width=0.84
        )

class FibonacciPlotStrategy(PlotStrategy):
    def plot(self, ax, df, series: dict, indicator: dict, x_index, fib_data: tuple = None):
        """
        Plot Fibonacci retracement levels.
        
        Args:
            ax: Matplotlib axis to plot on.
            df: DataFrame with indicator data.
            series: Series configuration from config.json.
            indicator: Indicator configuration from config.json.
            x_index: Index for x-axis (dates).
            fib_data: Precomputed Fibonacci data (levels, labels, start_date, end_date).
                      If None, compute it; otherwise, use provided data.
        
        Returns:
            tuple: Fibonacci data (levels, labels, start_date, end_date) or None if computation fails.
        """
        fib_levels, fib_labels, fib_start_date, fib_end_date = fib_data
        color = series.get('color', '#ffa500')
        label = series.get('label', '') if series.get('legend', True) else '_nolegend_'
        for level in fib_levels:
            ax.hlines(
                  y=level,
                  xmin=fib_start_date,
                  xmax=fib_end_date,
                  color=color,
                  linestyle='--',
                  alpha=0.5,
                  label=label
              )

class SubplotGenerator:
    def __init__(self, ticker: str, df: pd.DataFrame, config: dict, watch_indicators: List[str] = []):
        self.ticker = ticker
        self.x_index = df.index.strftime('%Y-%m-%d')
        self.today = datetime.now().strftime('%Y-%m-%d')
        self.df = df
        self.config = config
        self.watch_indicators = watch_indicators
        self.today = datetime.now().strftime('%Y-%m-%d')
        self.fig = plt.figure(figsize=(12, 8), facecolor=config['figure']['background'])
        self.axes = []
        self.grouped = defaultdict(list)

        # Filter indicators that are active and in watch_indicators
        for i, ind in enumerate(self.config['tech_indicators']):
            if not ind['plot']['active']:
                continue
            name = ind['name'].lower()
            # Check if any series' y_key is in watch_indicators or if it's fibonacci
            valid_series = []
            for series in ind['plot']['series']:
                y_key = series.get('y_key')
                if y_key and y_key in self.watch_indicators:
                    valid_series.append(series)
            if name == 'fibonacci' and 'fibonacci' in self.watch_indicators:
                valid_series = ind['plot']['series']  # Include all series for fibonacci
            if valid_series:
                # Update the indicator's series to only include valid ones
                ind['plot']['series'] = valid_series
                pos = ind['plot']['subplot_position']
                self.grouped[pos].append(i)

        if not self.grouped:
            print(f"No indicators to plot for ticker '{self.ticker}' based on current market sentiment.")
            plt.close(self.fig)
            return

        unique_positions = sorted(self.grouped.keys())
        self.pos_to_ax_idx = {pos: i for i, pos in enumerate(unique_positions)}

        heights = []
        for pos in unique_positions:
            max_h = max(self.config['tech_indicators'][i]['plot']['h_ratio'] for i in self.grouped[pos])
            heights.append(max_h)

        total_height = sum(heights)
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(total_height, 1, figure=self.fig)

        start_row = 0
        for idx, pos in enumerate(unique_positions):
            h = heights[idx]
            ax = self.fig.add_subplot(gs[start_row:start_row + h, 0])
            self.axes.append(ax)
            start_row += h

        self.fib_data = None
        self.price_linewidth = 1.2
        self.plot_strategies = {
            'line': LinePlotStrategy(),
            'bar': BarPlotStrategy(),
            'fibonacci': FibonacciPlotStrategy(),
        }
        self.latest_value_strategy = LatestValueLabelStrategy()

    def _get_fib_data(self, period: int, ratio):
        fib_data = None
        try:
            fib_data = fib_retracement(self.df, period, ratio)
            fib_levels, fib_labels, fib_start_date, fib_end_date = fib_data
            if fib_start_date not in self.df.index or fib_end_date not in self.df.index:
                print(f"Warning: Fibonacci dates out of range for ticker '{self.ticker}'.")
                return None
        except Exception as e:
            print(f"Error computing Fibonacci lines for ticker '{self.ticker}': {e}")
            return None
        return fib_data

    def plot(self, idx: int):
        """
        Plots the indicator for the given index in the configuration.
        """
        if len(self.axes) == 0:
            return
        ind = self.config['tech_indicators'][idx]
        pos = ind['plot']['subplot_position']
        subplot_idx = self.pos_to_ax_idx[pos]
        ax = self.axes[subplot_idx]
        plot_cfg = ind['plot']
        name = ind['name'].lower()
        
        if plot_cfg.get('add_price_chart'):
            if 'Close' in self.df.columns:
                sns.lineplot(x=self.x_index, y=self.df['Close'], ax=ax, color='blue', label='_nolegend_', linewidth=self.price_linewidth)
                latest_close = self.df['Close'].iloc[-1]
                ax.minorticks_on()
                ax.set_yticks([latest_close], minor=True)
                ax.set_yticklabels([f'${latest_close:.2f}'], minor=True)
                ax.tick_params(axis='y', which='minor', labelcolor='blue', labelsize=8, pad=5)
            else:
                print(f"Warning: 'Close' column missing for ticker '{self.ticker}'.")
        
        if 'overbought' in ind:
            ax.axhline(y=ind['overbought'], linestyle='--', color='red', label='_nolegend_', alpha=0.5)
        if 'oversold' in ind:
            ax.axhline(y=ind['oversold'], linestyle='--', color='green', label='_nolegend_', alpha=0.5)
        
        for series in plot_cfg.get('series', []):
            s_type = series['type']
            if s_type in self.plot_strategies:
                if s_type == 'fibonacci':
                    self.fib_data = self._get_fib_data(ind.get('periods', 90), ind.get('ratios', [0, 0.382, 0.5, 0.618, 1.0]))
                    self.plot_strategies[s_type].plot(ax, self.df, series, ind, self.x_index, self.fib_data)
                else:
                    self.plot_strategies[s_type].plot(ax, self.df, series, ind, self.x_index)
            else:
                raise ValueError(f"Unsupported plot type: {s_type} for ticker '{self.ticker}'")
            
            # Apply latest value label if enabled in series config
            if series.get('mark_value', False):
                self.latest_value_strategy.apply(ax, self.df, series, ind, self.fib_data)
        
        if name in ('rsi', 'stochastic'):
            ax.set_ylim(0, 100)
        if name == 'macd':
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')
        
        if name != 'fibonacci':
            ax.set_ylabel(name.replace('-', ' ').title())
        ax.set_xticks([])
        ax.grid(True)
        ax.grid(axis='x', visible=False)
        
        valid_labels = []
        has_legend = False
        for ind_idx in self.grouped[pos]:
            ind_config = self.config['tech_indicators'][ind_idx]
            for series in ind_config['plot']['series']:
                if series.get('legend', True):
                    valid_labels.append(series['label'])
                    has_legend = True
        lines = [line for line in ax.get_lines() if line.get_label() in valid_labels and line.get_label() != '_nolegend_']
        if has_legend:
            framealpha = 1
            facecolor = 'white'
            edgecolor = 'white'
        else:
            framealpha = 0
            facecolor = 'none'
            edgecolor = 'none'

        if lines:
            ax.legend(handles=lines, loc='upper left', framealpha=framealpha, facecolor=facecolor, edgecolor=edgecolor)

    def finalize(self):
        """Finalize the plot with title and display, then clear memory."""
        if len(self.axes) > 0:
            self.fig.suptitle(f'{self.ticker} {self.today}')
            # plt.tight_layout()
            plt.subplots_adjust(bottom=0.05)
            plt.show()
            plt.close(self.fig)  # Close figure to free memory
        gc.collect()  # Free memory

def draw(ticker: str, df: pd.DataFrame, config: dict, watch_indicators: List[str] = []):
    """
    Draw a plot of the given DataFrame and technical indicators for a specific ticker.
    
    Args:
        ticker (str): Stock ticker symbol.
        df (pd.DataFrame): DataFrame containing indicator data for the ticker.
        config (dict): Configuration with technical indicator data.
        watch_indicators (List[str]): List of indicators to watch based on market sentiment.
    """
    if df.empty:
        print(f"No data available for ticker '{ticker}'")
        return
    # Avoid copy by filtering in-place if possible
    if len(df['Ticker'].unique()) > 1:
        print(f"Warning: Multiple tickers {df['Ticker'].unique()} found in DataFrame for '{ticker}', filtering to '{ticker}'.")
        df = df[df['Ticker'] == ticker]
    plt.set_loglevel('WARNING')
    generator = SubplotGenerator(ticker, df, config, watch_indicators)
    active_indices = sorted(set(i for pos in generator.grouped for i in generator.grouped[pos]))
    for idx in active_indices:
        generator.plot(idx)
    generator.finalize()