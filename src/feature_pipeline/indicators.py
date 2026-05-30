import pandas as pd
import numpy as np
from typing import Tuple


def calculate_sma(prices: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average."""
    return prices.rolling(window=period).mean()


def calculate_ema(prices: pd.Series, period: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    return prices.ewm(span=period, adjust=False).mean()


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0-100)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_volatility(returns: pd.Series, period: int = 20) -> pd.Series:
    """Rolling standard deviation of returns."""
    return returns.rolling(window=period).std()


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Log returns."""
    return np.log(prices / prices.shift(1))


def calculate_zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Z-score normalized by rolling mean/std."""
    rolling_mean = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    
    zscore = (series - rolling_mean) / (rolling_std + 1e-8)  # Avoid division by zero
    
    return zscore


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
    """Bollinger Bands: (upper, lower)."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper = sma + (num_std * std)
    lower = sma - (num_std * std)
    
    return upper, lower


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """MACD: (macd_line, signal_line)."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    return macd_line, signal_line