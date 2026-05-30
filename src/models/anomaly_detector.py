import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Ensemble anomaly detector combining Isolation Forest + Z-score.
    
    Returns anomaly scores (0-1) and binary anomaly labels (0=normal, 1=anomaly).
    """
    
    def __init__(self, 
                 contamination: float = 0.05,
                 zscore_threshold: float = 2.5,
                 random_state: int = 42):
        """
        Args:
            contamination: Expected proportion of anomalies (IF hyperparameter)
            zscore_threshold: Z-score threshold for statistical detection
            random_state: For reproducibility
        """
        self.contamination = contamination
        self.zscore_threshold = zscore_threshold
        self.random_state = random_state
        
        self.isolation_forest = None
        self.is_trained = False
    
    def train(self, df: pd.DataFrame, feature_cols: list) -> None:
        """
        Train Isolation Forest on normal data.
        
        Args:
            df: Training DataFrame (should contain mostly normal data)
            feature_cols: List of feature column names to use
        """
        if len(df) < 10:
            raise ValueError("Need at least 10 samples to train")
        
        X = df[feature_cols].values
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        self.isolation_forest.fit(X)
        
        self.is_trained = True
        logger.info(f"✅ Trained Isolation Forest on {len(df)} samples")
    
    def detect(self, df: pd.DataFrame, feature_cols: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using ensemble method.
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature columns
        
        Returns:
            Tuple of (anomaly_scores, anomaly_labels)
            - anomaly_scores: 0-1 (0=normal, 1=anomaly)
            - anomaly_labels: 0/1 binary labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X = df[feature_cols].values
        
        # Isolation Forest predictions (-1=anomaly, 1=normal)
        if_predictions = self.isolation_forest.predict(X)
        if_scores = -self.isolation_forest.score_samples(X)
        if_scores = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
        
        # Z-score method: flag if ANY feature is above threshold
        feature_zscores = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8))
        zscore_anomalies = (feature_zscores > self.zscore_threshold).any(axis=1).astype(int)
        
        # Ensemble: vote
        if_binary = (if_predictions == -1).astype(int)
        ensemble_votes = if_binary + zscore_anomalies
        ensemble_labels = (ensemble_votes >= 1).astype(int)  # 1+ vote = anomaly
        
        # Confidence score: average of both methods
        ensemble_scores = (if_scores + zscore_anomalies) / 2.0
        
        return ensemble_scores, ensemble_labels
    
    def detect_realtime(self, 
                       latest_row: pd.Series, 
                       feature_cols: list,
                       confidence_threshold: float = 0.65) -> Dict:
        """
        Detect anomaly on a single new data point.
        
        Args:
            latest_row: Single row (pd.Series) with features
            feature_cols: Feature column names
            confidence_threshold: Alert if confidence > this
        
        Returns:
            Dict with keys: is_anomaly, confidence, if_score, zscore_max
        """
        df_point = pd.DataFrame([latest_row])
        scores, labels = self.detect(df_point, feature_cols)
        
        confidence = scores[0]
        is_anomaly = labels[0]
        
        # Get max z-score for diagnostics
        X = df_point[feature_cols].values
        zscores = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8))
        max_zscore = zscores.max()
        
        result = {
            "is_anomaly": bool(is_anomaly),
            "confidence": float(confidence),
            "alert": confidence > confidence_threshold,
            "max_zscore": float(max_zscore),
            "timestamp": latest_row.get("timestamp", None)
        }
        
        return result