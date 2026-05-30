import pytest
import pandas as pd
import numpy as np
from src.ingestion.data_fetcher import get_forex_data
from src.feature_pipeline.feature_engineer import FeatureEngineer


@pytest.fixture
def sample_data():
    """Get sample forex data."""
    return get_forex_data(days=1)


def test_engineer_features(sample_data):
    """Test that features are computed."""
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(sample_data)
    
    # Should have more columns
    assert len(df_features.columns) > len(sample_data.columns)
    
    # Should have key features
    assert "sma_20" in df_features.columns
    assert "rsi_14" in df_features.columns
    assert "volatility_20" in df_features.columns
    
    print(f"✅ Created {len(engineer.get_feature_columns(df_features))} features")


def test_normalize_features(sample_data):
    """Test feature normalization."""
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(sample_data)
    df_normalized = engineer.normalize_features(df_features, fit=True)
    
    # Should have normalized values (close to 0 mean)
    feature_cols = engineer.get_feature_columns(df_normalized)
    for col in feature_cols:
        if not df_normalized[col].isna().all():
            mean = df_normalized[col].mean()
            assert abs(mean) < 1.0, f"Feature {col} not normalized"


def test_prepare_for_ml(sample_data):
    """Test complete ML preparation pipeline."""
    engineer = FeatureEngineer()
    df_ml = engineer.prepare_for_ml(sample_data, fit=True)
    
    # Should have no NaN
    assert df_ml.isna().sum().sum() == 0
    
    # Should be ready for sklearn
    assert len(df_ml) > 0
    
    print(f"✅ Prepared {len(df_ml)} rows for ML")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])