import pytest
import pandas as pd
from src.ingestion.data_fetcher import YahooFinanceFetcher, get_forex_data


def test_fetcher_initialization():
    """Test that fetcher initializes correctly."""
    fetcher = YahooFinanceFetcher()
    assert fetcher.symbol == "EURUSD=X"


def test_get_forex_data():
    """Test that we can fetch real data."""
    df = get_forex_data(days=1)
    
    # Should have these columns
    assert "timestamp" in df.columns
    assert "close" in df.columns
    assert "open" in df.columns
    
    # Should have rows
    assert len(df) > 0
    
    # Close prices should be positive (forex rates)
    assert (df["close"] > 0).all()


if __name__ == "__main__":
    test_get_forex_data()
    print("✅ All tests passed")