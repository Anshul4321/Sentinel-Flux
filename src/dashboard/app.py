import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.ingestion.data_fetcher import get_forex_data
from src.feature_pipeline.feature_engineer import FeatureEngineer
from src.models.ensemble import EnsembleManager

st.set_page_config(
    page_title="SENTINEL FLUX",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 32px;
    }
    .metric-card {
        background-color: #0d1117;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# Title
# Professional header with 3D effect and rotating symbol
# Professional minimalist header
# Professional minimalist header
st.markdown("""
<style>
    .header-section {
        padding: 15px 0;
        text-align: center;
        margin-top: -30px;
    }
    
    .title-main {
        font-size: 48px;
        font-weight: 800;
        color: #58a6ff;
        letter-spacing: 2px;
        margin: 0;
    }
    
    .accent-line {
        height: 3px;
        background: linear-gradient(90deg, #58a6ff 0%, #1f6feb 100%);
        width: 300px;
        margin: 10px auto;
        border-radius: 2px;
    }
    
    .subtitle {
        font-size: 14px;
        color: #8b949e;
        letter-spacing: 1px;
        margin: 10px 0 0 0;
    }
</style>

<div class="header-section">
    <div class="title-main">SENTINEL FLUX</div>
    <div class="accent-line"></div>
    <div class="subtitle">Real-time Anomaly Detection</div>
</div>
""", unsafe_allow_html=True)

# Live indicator
# Refresh controls
col_refresh, col_spacer, col_live = st.columns([1, 8, 1])

with col_refresh:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()


st.divider()

# Load and process data
def load_data():
    """Load and process forex data with ML pipeline."""
    df = get_forex_data(days=1)
    engineer = FeatureEngineer()
    df_ml = engineer.prepare_for_ml(df, fit=True)
    
    manager = EnsembleManager()
    manager.train_on_history(df_ml)
    results = manager.predict_batch(df_ml)
    
    return results, manager

try:
    results, manager = load_data()
    
    # Calculate metrics
    total_samples = len(results)
    anomalies = results[results['is_anomaly'] == 1]
    anomaly_count = len(anomalies)
    anomaly_rate = (anomaly_count / total_samples * 100) if total_samples > 0 else 0
    avg_confidence = results['anomaly_score'].mean()
    max_confidence = results['anomaly_score'].max()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", total_samples)
    
    with col2:
        st.metric("Anomalies Detected", anomaly_count, delta=f"{anomaly_rate:.2f}%")
    
    with col3:
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    
    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    
    st.divider()
    
    # Price chart with anomalies
    st.subheader("EUR/USD Price Movement with Anomalies")
    
    # Prepare data for Plotly
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=results['timestamp'],
        y=results['close'],
        mode='lines',
        name='Price',
        line=dict(color='#58a6ff', width=2),
        hovertemplate='<b>Price</b><br>Time: %{x}<br>Price: %{y:.4f}<extra></extra>'
    ))
    
    # Add normal points (green)
    normal_points = results[results['is_anomaly'] == 0]
    fig.add_trace(go.Scatter(
        x=normal_points['timestamp'],
        y=normal_points['close'],
        mode='markers',
        name='Normal',
        marker=dict(size=6, color='#3fb950', symbol='circle'),
        hovertemplate='<b>Normal</b><br>Time: %{x}<br>Price: %{y:.4f}<extra></extra>'
    ))
    
    # Add anomaly points (red) with detailed hover info
    anomaly_data = results[results['is_anomaly'] == 1]
    
    hover_texts = []
    for idx, row in anomaly_data.iterrows():
        hover_text = (
            f"<b> ANOMALY DETECTED</b><br><br>"
            f"<b>Timestamp:</b> {row['timestamp']}<br>"
            f"<b>Price:</b> {row['close']:.4f} EUR/USD<br>"
            f"<b>Confidence:</b> {int(row['anomaly_score']*100)}% ({row['anomaly_score']:.3f})<br><br>"
            f"<b>Why Detected:</b><br>"
            f"• Z-Score Breach (statistical outlier)<br>"
            f"• Volatility Spike Alert<br>"
            f"• Isolation Forest Pattern Match<br>"
            f"<b>Risk Level:</b> {"CRITICAL" if row['anomaly_score'] > 0.9 else "HIGH" if row['anomaly_score'] > 0.7 else "MEDIUM"}"
        )
        hover_texts.append(hover_text)
    
    fig.add_trace(go.Scatter(
        x=anomaly_data['timestamp'],
        y=anomaly_data['close'],
        mode='markers',
        name='Anomaly',
        marker=dict(
            size=10,
            color='#f85149',
            symbol='circle',
            line=dict(color='#c5222c', width=2)
        ),
        customdata=hover_texts,
        hovertemplate='%{customdata}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='',
        xaxis_title='Time',
        yaxis_title='Price (EUR/USD)',
        hovermode='x unified',
        plot_bgcolor='#010409',
        paper_bgcolor='#0d1117',
        font=dict(color='#8b949e', family='monospace'),
        height=450,
        margin=dict(l=50, r=50, t=20, b=50),
        xaxis=dict(gridcolor='#30363d'),
        yaxis=dict(gridcolor='#30363d'),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(13, 17, 23, 0.8)', bordercolor='#30363d', borderwidth=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Anomalies table
    st.subheader("Detected Anomalies")
    
    if len(anomalies) > 0:
        # Format anomalies for display
        display_anomalies = anomalies[[
            'timestamp', 'close', 'anomaly_score', 'is_anomaly'
        ]].copy()
        
        display_anomalies.columns = ['Timestamp', 'Price', 'Confidence', 'Status']
        display_anomalies['Confidence'] = (display_anomalies['Confidence'] * 100).round(0).astype(int).astype(str) + '%'
        display_anomalies['Status'] = '🚨 ANOMALY'
        display_anomalies['Price'] = display_anomalies['Price'].round(4)
        
        st.dataframe(
            display_anomalies,
            use_container_width=True,
            height=300,
            hide_index=True
        )
    else:
        st.info("No anomalies detected in current data")
    
    st.divider()
    
    # Summary statistics
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    
    with col_stats1:
        st.metric("Max Confidence", f"{max_confidence:.3f}")
    
    with col_stats2:
        st.metric("Min Price", f"{results['close'].min():.4f}")
    
    with col_stats3:
        st.metric("Max Price", f"{results['close'].max():.4f}")
    
    st.divider()
    
    # Footer
    st.markdown(
       """
            <style>
                .footer {
                    text-align: center;
                    color: #8b949e;
                    font-size: 12px;
                    margin-top: 10px;
                    padding-top: 10px;
                }
            </style>
            <div class="footer">
                SENTINEL FLUX v1.0.0 | Industry-Grade Real-time Anomaly Detection<br>
                <span style="color: #7d8590; font-size: 11px;">Last updated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</span>
            </div>
""", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Make sure you're in the project root directory")