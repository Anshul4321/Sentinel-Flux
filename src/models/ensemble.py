import pandas as pd
import numpy as np
from src.models.anomaly_detector import AnomalyDetector
import logging

logger = logging.getLogger(__name__)


class EnsembleManager:
    """Manage training and inference with anomaly detector."""
    
    def __init__(self, contamination: float = 0.05):
        self.detector = AnomalyDetector(contamination=contamination)
        self.feature_cols = None
    
    def train_on_history(self, df: pd.DataFrame) -> None:
        """
        Train detector on historical data.
        
        Args:
            df: DataFrame with all features (from feature pipeline)
        """
        # Get all non-OHLCV columns as features
        base_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        self.feature_cols = [col for col in df.columns if col not in base_cols]
        
        if len(self.feature_cols) == 0:
            raise ValueError("No feature columns found")
        
        logger.info(f"🎓 Training on {len(self.feature_cols)} features: {self.feature_cols[:5]}...")
        self.detector.train(df, self.feature_cols)
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict on batch of data.
        
        Args:
            df: DataFrame with features
        
        Returns:
            Original DataFrame with added columns:
            - anomaly_score: 0-1 confidence
            - is_anomaly: 0/1 label
        """
        if self.feature_cols is None:
            raise ValueError("Model not trained")
        
        scores, labels = self.detector.detect(df, self.feature_cols)
        
        df = df.copy()
        df["anomaly_score"] = scores
        df["is_anomaly"] = labels
        
        n_anomalies = labels.sum()
        logger.info(f"📊 Detected {n_anomalies} anomalies in {len(df)} samples")
        
        return df
    
    def predict_realtime(self, latest_row: pd.Series) -> dict:
        """
        Predict on single new data point (for streaming).
        
        Args:
            latest_row: Single row with all features
        
        Returns:
            Dict with anomaly result
        """
        if self.feature_cols is None:
            raise ValueError("Model not trained")
        
        result = self.detector.detect_realtime(latest_row, self.feature_cols)
        
        if result["alert"]:
            logger.warning(f"🚨 ANOMALY DETECTED: confidence={result['confidence']:.3f}")
        
        return result
    
    def get_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics of predictions."""
        if "is_anomaly" not in df.columns:
            raise ValueError("Run predict_batch first")
        
        n_total = len(df)
        n_anomalies = df["is_anomaly"].sum()
        anomaly_rate = (n_anomalies / n_total * 100) if n_total > 0 else 0
        
        summary = {
            "total_samples": n_total,
            "anomalies_detected": int(n_anomalies),
            "anomaly_rate_pct": round(anomaly_rate, 2),
            "avg_confidence": round(df["anomaly_score"].mean(), 3),
            "max_confidence": round(df["anomaly_score"].max(), 3),
        }
        
        return summary