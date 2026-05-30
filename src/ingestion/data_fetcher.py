import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class YahooFinanceFetcher:
    """Fetch EURUSD data with fallback to synthetic data."""
    
    SYMBOL = "EURUSD=X"
    
    def __init__(self, symbol: str = SYMBOL):
        self.symbol = symbol
    
    def fetch_latest(self, period: str = "1d", interval: str = "1m") -> pd.DataFrame:
        """
        Fetch forex data. Falls back to synthetic if API fails.
        
        Args:
            period: '1d', '5d', '1mo', etc.
            interval: '1m', '5m', '1h', '1d'
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Attempting to fetch {self.symbol}...")
            df = self._fetch_from_yfinance(period, interval)
            
            if df is not None and len(df) > 0:
                logger.info(f"✅ Fetched {len(df)} real records for {self.symbol}")
                return df
        
        except Exception as e:
            logger.warning(f"⚠️  Real data fetch failed: {e}")
        
        # Fallback to synthetic data
        logger.info("📊 Using synthetic data for testing/development")
        return self._generate_synthetic_data(period, interval)
    
    def _fetch_from_yfinance(self, period: str, interval: str) -> pd.DataFrame:
        """Try to fetch from yfinance."""
        df = yf.download(
            self.symbol,
            period=period,
            interval=interval,
            progress=False
        )
        
        if df is None or len(df) == 0:
            return None
        
        df = df.reset_index()
        df.columns = df.columns.str.lower()
        df = df.rename(columns={"date": "timestamp"})
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.dropna(subset=["close"])
        df = df.reset_index(drop=True)
        
        return df
    
    def _generate_synthetic_data(self, period: str, interval: str) -> pd.DataFrame:
        """Generate realistic synthetic forex data for testing."""
        # Determine number of points
        if period == "1d":
            n_points = 240  # ~4 hours of 1-min data
        elif period == "5d":
            n_points = 288  # 5 days of 5-min data
        elif period == "1mo":
            n_points = 160  # 1 month of hourly data
        else:
            n_points = 60
        
        # Generate timestamps
        now = datetime.now()
        if interval == "1m":
            timestamps = [now - timedelta(minutes=i) for i in range(n_points, 0, -1)]
        elif interval == "5m":
            timestamps = [now - timedelta(minutes=5*i) for i in range(n_points, 0, -1)]
        elif interval == "1h":
            timestamps = [now - timedelta(hours=i) for i in range(n_points, 0, -1)]
        else:
            timestamps = [now - timedelta(days=i) for i in range(n_points, 0, -1)]
        
        # Generate realistic price movement (EUR/USD typically ~1.08)
        base_price = 1.0850
        returns = np.random.normal(0.0001, 0.005, n_points)
        close_prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close prices
        df = pd.DataFrame({
            "timestamp": timestamps,
            "close": close_prices,
        })
        
        # Generate open, high, low from close
        df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
        df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.002, n_points)))
        df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.002, n_points)))
        df["volume"] = np.random.randint(1000000, 5000000, n_points)
        
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.reset_index(drop=True)
        
        return df


def get_forex_data(symbol: str = "EURUSD=X", days: int = 1) -> pd.DataFrame:
    """
    Fetch forex data with fallback to synthetic.
    
    Args:
        symbol: Forex pair
        days: Number of days
    
    Returns:
        DataFrame with OHLCV data
    """
    fetcher = YahooFinanceFetcher(symbol)
    
    if days == 1:
        period, interval = "1d", "1m"
    elif days <= 5:
        period, interval = "5d", "5m"
    elif days <= 30:
        period, interval = "1mo", "1h"
    else:
        period, interval = "3mo", "1d"
    
    return fetcher.fetch_latest(period=period, interval=interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = get_forex_data(days=1)
    print("\n" + "="*60)
    print(df.head(10))
    print("\n" + "="*60)
    print(f"Total records: {len(df)}")
    print(f"Price range: {df['low'].min():.4f} - {df['high'].max():.4f}")
    print(f"Last close: {df['close'].iloc[-1]:.4f}")