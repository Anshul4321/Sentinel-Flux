import logging
import sys
sys.path.insert(0, '.')

from src.ingestion.data_fetcher import get_forex_data
from src.feature_pipeline.feature_engineer import FeatureEngineer

logging.basicConfig(level=logging.INFO)

# Fetch data
df = get_forex_data(days=1)
print(f"\n📥 Raw data shape: {df.shape}")
print(df[["timestamp", "close"]].head())

# Engineer features
engineer = FeatureEngineer()
df_features = engineer.engineer_features(df)
print(f"\n⚙️  Featured data shape: {df_features.shape}")
print(f"Features created: {engineer.get_feature_columns(df_features)[:5]}...")

# Prepare for ML
df_ml = engineer.prepare_for_ml(df, fit=True)
print(f"\n🤖 ML-ready data shape: {df_ml.shape}")
print(df_ml.iloc[:5, :6])