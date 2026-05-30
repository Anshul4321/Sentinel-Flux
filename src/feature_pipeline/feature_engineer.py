import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

from src.feature_pipeline.indicators import (
    calculate_sma, calculate_ema, calculate_rsi,
    calculate_volatility, calculate_atr, calculate_returns,
    calculate_zscore, calculate_bollinger_bands, calculate_macd
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Transform raw OHLCV data into ML-ready features."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from OHLCV data.
        
        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
        
        Returns:
            DataFrame with original OHLCV + computed features
        """
        df = df.copy()
        
        # Ensure close prices are valid
        if df["close"].isna().all():
            raise ValueError("No valid close prices in data")
        
        # Basic price features
        df["returns"] = calculate_returns(df["close"])
        df["log_volume"] = np.log(df["volume"] + 1)
        
        # Moving averages
        df["sma_20"] = calculate_sma(df["close"], 20)
        df["ema_20"] = calculate_ema(df["close"], 20)
        df["sma_50"] = calculate_sma(df["close"], 50)
        
        # Momentum indicators
        df["rsi_14"] = calculate_rsi(df["close"], 14)
        df["atr_14"] = calculate_atr(df["high"], df["low"], df["close"], 14)
        
        # Volatility
        df["volatility_20"] = calculate_volatility(df["returns"], 20)
        
        # Z-scores (anomaly detection relevant)
        df["zscore_returns"] = calculate_zscore(df["returns"], 20)
        df["zscore_volume"] = calculate_zscore(df["log_volume"], 20)
        
        # Bollinger Bands
        bb_upper, bb_lower = calculate_bollinger_bands(df["close"], 20)
        df["bb_upper"] = bb_upper
        df["bb_lower"] = bb_lower
        df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        # MACD
        macd_line, signal_line = calculate_macd(df["close"])
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = macd_line - signal_line
        
        # Price changes
        df["price_change_pct"] = df["close"].pct_change() * 100
        df["high_low_pct"] = ((df["high"] - df["low"]) / df["low"]) * 100
        
        logger.info(f"✅ Engineered {len(df.columns) - 6} features from raw data")
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get names of all engineered features (exclude OHLCV)."""
        base_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        return [col for col in df.columns if col not in base_cols]
    
    def normalize_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Normalize all features to zero mean, unit variance.
        
        Args:
            df: DataFrame with features
            fit: If True, fit scaler on this data (for training)
        
        Returns:
            DataFrame with normalized features
        """
        df = df.copy()
        feature_cols = self.get_feature_columns(df)
        
        # Drop NaN rows before normalization
        df_clean = df.dropna(subset=feature_cols)
        
        if len(df_clean) == 0:
            logger.warning("⚠️  No valid features after dropping NaN")
            return df
        
        if fit:
            self.scaler.fit(df_clean[feature_cols])
            logger.info("📊 Fitted scaler on training data")
        
        # Normalize only clean rows
        df_clean[feature_cols] = self.scaler.transform(df_clean[feature_cols])
        
        logger.info(f"✅ Normalized {len(feature_cols)} features")
        
        return df_clean
    
    def prepare_for_ml(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Complete pipeline: engineer features → normalize → drop NaN.
        
        Args:
            df: Raw OHLCV DataFrame
            fit: If True, fit scaler on this data
        
        Returns:
            ML-ready DataFrame with normalized features
        """
        # Engineer features
        df_features = self.engineer_features(df)
        
        # Normalize
        df_normalized = self.normalize_features(df_features, fit=fit)
        
        # Drop rows with any NaN
        df_clean = df_normalized.dropna()
        
        logger.info(f"✅ Prepared {len(df_clean)} rows for ML")
        
        return df_clean