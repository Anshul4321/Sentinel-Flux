import logging
import sys
sys.path.insert(0, '.')

from src.ingestion.data_fetcher import get_forex_data
from src.feature_pipeline.feature_engineer import FeatureEngineer
from src.models.ensemble import EnsembleManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("\n" + "="*70)
print("SENTINEL FLUX - Anomaly Detection Demo")
print("="*70)

# Step 1: Fetch data
print("\n📥 Fetching forex data...")
df = get_forex_data(days=1)

# Step 2: Engineer features
print("\n⚙️  Engineering features...")
engineer = FeatureEngineer()
df_ml = engineer.prepare_for_ml(df, fit=True)

# Step 3: Train anomaly detector
print("\n🎓 Training ensemble anomaly detector...")
manager = EnsembleManager(contamination=0.05)
manager.train_on_history(df_ml)

# Step 4: Batch predictions
print("\n🔍 Running batch predictions...")
df_results = manager.predict_batch(df_ml)
summary = manager.get_summary(df_results)

print(f"\n{'─'*70}")
print(f"Detection Summary:")
print(f"{'─'*70}")
print(f"Total samples:        {summary['total_samples']}")
print(f"Anomalies detected:   {summary['anomalies_detected']}")
print(f"Anomaly rate:         {summary['anomaly_rate_pct']:.2f}%")
print(f"Avg confidence:       {summary['avg_confidence']:.3f}")
print(f"Max confidence:       {summary['max_confidence']:.3f}")

# Step 5: Show top anomalies
print(f"\n{'─'*70}")
print("🚨 Top 5 Anomalies (by confidence):")
print(f"{'─'*70}")
top_anomalies = df_results[df_results["is_anomaly"] == 1].nlargest(5, "anomaly_score")
for idx, row in top_anomalies.iterrows():
    print(f"  {row['timestamp']} | Score: {row['anomaly_score']:.3f} | Close: {row['close']:.4f}")

# Step 6: Realtime prediction on latest point
print(f"\n{'─'*70}")
print("📊 Realtime prediction (latest data point):")
print(f"{'─'*70}")
latest = df_results.iloc[-1]
realtime_result = manager.predict_realtime(latest)
print(f"Timestamp:   {realtime_result['timestamp']}")
print(f"Confidence:  {realtime_result['confidence']:.3f}")
print(f"Is Anomaly:  {realtime_result['is_anomaly']}")
print(f"Alert:       {realtime_result['alert']}")
print(f"Max Z-score: {realtime_result['max_zscore']:.3f}")

print("\n" + "="*70)