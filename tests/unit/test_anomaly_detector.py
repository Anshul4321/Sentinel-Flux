import pytest
import pandas as pd
from src.ingestion.data_fetcher import get_forex_data
from src.feature_pipeline.feature_engineer import FeatureEngineer
from src.models.ensemble import EnsembleManager


@pytest.fixture
def prepared_data():
    """Get prepared ML-ready data."""
    df = get_forex_data(days=1)
    engineer = FeatureEngineer()
    return engineer.prepare_for_ml(df, fit=True)


def test_ensemble_training(prepared_data):
    """Test that ensemble trains."""
    manager = EnsembleManager()
    manager.train_on_history(prepared_data)
    
    assert manager.detector.is_trained
    assert manager.feature_cols is not None
    print(f"✅ Trained on {len(manager.feature_cols)} features")


def test_batch_prediction(prepared_data):
    """Test batch anomaly detection."""
    manager = EnsembleManager()
    manager.train_on_history(prepared_data)
    
    df_predictions = manager.predict_batch(prepared_data)
    
    assert "anomaly_score" in df_predictions.columns
    assert "is_anomaly" in df_predictions.columns
    assert (df_predictions["anomaly_score"] >= 0).all()
    assert (df_predictions["anomaly_score"] <= 1).all()
    
    n_anomalies = df_predictions["is_anomaly"].sum()
    print(f"✅ Detected {n_anomalies} anomalies in {len(df_predictions)} samples")


def test_realtime_prediction(prepared_data):
    """Test single-point anomaly detection."""
    manager = EnsembleManager()
    manager.train_on_history(prepared_data)
    
    latest_row = prepared_data.iloc[-1]
    result = manager.predict_realtime(latest_row)
    
    assert "is_anomaly" in result
    assert "confidence" in result
    assert "alert" in result
    assert 0 <= result["confidence"] <= 1
    
    print(f"✅ Realtime prediction: is_anomaly={result['is_anomaly']}, confidence={result['confidence']:.3f}")


def test_summary_stats(prepared_data):
    """Test prediction summary."""
    manager = EnsembleManager()
    manager.train_on_history(prepared_data)
    
    df_predictions = manager.predict_batch(prepared_data)
    summary = manager.get_summary(df_predictions)
    
    assert summary["total_samples"] == len(df_predictions)
    assert summary["anomaly_rate_pct"] >= 0
    
    print(f"✅ Summary: {summary}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])