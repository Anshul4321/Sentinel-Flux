import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.ingestion.data_fetcher import get_forex_data
from src.feature_pipeline.feature_engineer import FeatureEngineer
from src.models.ensemble import EnsembleManager
from src.agents.technical_analyzer import TechnicalAnalyzer
from src.agents.llm_analyzer import LLMAnalyzer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SENTINEL FLUX",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 32px; }
    .trigger-card {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .deep-box {
        background: #0d1117;
        border: 1px solid #1f6feb;
        border-radius: 8px;
        padding: 18px 22px;
        margin-top: 10px;
        line-height: 1.7;
    }
    .risk-badge {
        display: inline-block;
        padding: 5px 16px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .header-section { padding: 15px 0; text-align: center; margin-top: -30px; }
    .title-main { font-size: 48px; font-weight: 800; color: #58a6ff; letter-spacing: 2px; margin: 0; }
    .accent-line {
        height: 3px;
        background: linear-gradient(90deg, #58a6ff 0%, #1f6feb 100%);
        width: 300px; margin: 10px auto; border-radius: 2px;
    }
    .subtitle { font-size: 14px; color: #8b949e; letter-spacing: 1px; margin: 10px 0 0 0; }
</style>
<div class="header-section">
    <div class="title-main">SENTINEL FLUX</div>
    <div class="accent-line"></div>
    <div class="subtitle">Real-time Anomaly Detection</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: Ollama status ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Deep Analysis")

    ollama_ok = LLMAnalyzer.is_available()
    models = LLMAnalyzer.list_models() if ollama_ok else []

    if ollama_ok and models:
        selected_model = st.selectbox("Model", models, key="ollama_model")
        st.success(f"Ollama ready — {len(models)} model(s)")
    elif ollama_ok:
        selected_model = "llama3.2"
        st.warning("Ollama running, no models pulled.\nRun: `ollama pull llama3.2`")
    else:
        selected_model = "llama3.2"
        st.error("Ollama not running")
        st.markdown(
            "**Free local setup:**\n"
            "1. [ollama.ai](https://ollama.ai) → Download\n"
            "2. `ollama pull llama3.2`\n"
            "3. Restart this dashboard"
        )

# ── Refresh button ────────────────────────────────────────────────────────────
col_refresh, *_ = st.columns([1, 9])
with col_refresh:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        for k in list(st.session_state.keys()):
            if k.startswith("deep_"):
                del st.session_state[k]
        st.rerun()

st.divider()

# ── Constants ─────────────────────────────────────────────────────────────────
RISK_COLORS = {'CRITICAL': '#f85149', 'HIGH': '#d29922', 'MEDIUM': '#3fb950', 'LOW': '#58a6ff'}
SEV_ICONS   = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🔵'}


@st.cache_data(ttl=300)
def load_data():
    df = get_forex_data(days=1)
    engineer = FeatureEngineer()
    df_raw  = engineer.engineer_features(df)
    df_norm = engineer.normalize_features(df_raw, fit=True)
    df_ml   = df_norm.dropna()
    mgr = EnsembleManager()
    mgr.train_on_history(df_ml)
    results = mgr.predict_batch(df_ml)
    return results, df_raw.loc[df_ml.index]


# ── Main ──────────────────────────────────────────────────────────────────────
try:
    results, raw_for_ml = load_data()

    anomalies = results[results['is_anomaly'] == 1]
    total     = len(results)
    n_anom    = len(anomalies)
    rate      = (n_anom / total * 100) if total else 0
    avg_conf  = results['anomaly_score'].mean()
    max_conf  = results['anomaly_score'].max()

    # ── Metrics ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", total)
    c2.metric("Anomalies Detected", n_anom, delta=f"{rate:.2f}%")
    c3.metric("Anomaly Rate", f"{rate:.2f}%")
    c4.metric("Avg Confidence", f"{avg_conf:.3f}")
    st.divider()

    # ── Price chart (view only — no click events) ─────────────────────────────
    st.subheader("EUR/USD Price Movement with Anomalies")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results['timestamp'], y=results['close'],
        mode='lines', name='Price',
        line=dict(color='#58a6ff', width=2),
        hovertemplate='<b>Price</b><br>%{x}<br>%{y:.4f}<extra></extra>',
    ))

    normals = results[results['is_anomaly'] == 0]
    fig.add_trace(go.Scatter(
        x=normals['timestamp'], y=normals['close'],
        mode='markers', name='Normal',
        marker=dict(size=6, color='#3fb950'),
        hovertemplate='<b>Normal</b><br>%{x}<br>%{y:.4f}<extra></extra>',
    ))

    anom_data = results[results['is_anomaly'] == 1]
    hover_texts = []
    for _, row in anom_data.iterrows():
        risk_label = "CRITICAL" if row['anomaly_score'] > 0.9 else "HIGH" if row['anomaly_score'] > 0.7 else "MEDIUM"
        hover_texts.append(
            f"<b>ANOMALY</b><br>{row['timestamp']}<br>"
            f"Price: {row['close']:.4f}<br>"
            f"Confidence: {int(row['anomaly_score']*100)}%<br>"
            f"Risk: {risk_label}"
        )
    fig.add_trace(go.Scatter(
        x=anom_data['timestamp'], y=anom_data['close'],
        mode='markers', name='Anomaly',
        marker=dict(size=10, color='#f85149', symbol='circle',
                    line=dict(color='#c5222c', width=2)),
        customdata=hover_texts,
        hovertemplate='%{customdata}<extra></extra>',
    ))

    fig.update_layout(
        xaxis_title='Time', yaxis_title='Price (EUR/USD)',
        hovermode='x unified',
        plot_bgcolor='#010409', paper_bgcolor='#0d1117',
        font=dict(color='#8b949e', family='monospace'),
        height=450, margin=dict(l=50, r=50, t=20, b=50),
        xaxis=dict(gridcolor='#30363d'), yaxis=dict(gridcolor='#30363d'),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(13,17,23,0.8)',
                    bordercolor='#30363d', borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    # ── Anomaly table + selector ──────────────────────────────────────────────
    st.subheader("Detected Anomalies")

    if n_anom == 0:
        st.info("No anomalies detected in current data")
    else:
        # Build display table (positional index so iloc works cleanly below)
        anom_list = []
        for pos, (orig_idx, row) in enumerate(anomalies.iterrows()):
            ts = row['timestamp']
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
            conf_pct = int(row['anomaly_score'] * 100)
            risk_label = ("CRITICAL" if row['anomaly_score'] > 0.9
                          else "HIGH" if row['anomaly_score'] > 0.7
                          else "MEDIUM")
            anom_list.append({
                '_orig_idx': orig_idx,
                '_ts_str': ts_str,
                'Status': '🚨',
                'Timestamp': ts_str,
                'Price': round(row['close'], 4),
                'Confidence': f"{conf_pct}%",
                'Risk': risk_label,
            })

        display_df = pd.DataFrame(anom_list)[['Status', 'Timestamp', 'Price', 'Confidence', 'Risk']]
        st.dataframe(display_df, use_container_width=True, height=260, hide_index=True)

        # Row selector — numbered to match table rows
        selector_options = ["— select a row to investigate —"] + [
            f"Row {i+1}  ·  {a['_ts_str']}  ·  {a['Price']}  ·  {a['Confidence']}"
            for i, a in enumerate(anom_list)
        ]
        selected_label = st.selectbox(
            "Select a row to investigate:",
            selector_options,
            key="row_selector",
        )

        # ── Investigation panel ───────────────────────────────────────────────
        if selected_label != "— select a row to investigate —":
            # Parse position from "Row N · ..."
            sel_pos = int(selected_label.split("·")[0].replace("Row", "").strip()) - 1
            sel_entry  = anom_list[sel_pos]
            orig_idx   = sel_entry['_orig_idx']
            ts_str     = sel_entry['_ts_str']
            anomaly_row = results.loc[orig_idx]
            raw_row     = raw_for_ml.loc[orig_idx]

            st.divider()
            st.subheader("🔍 Investigation")

            # Header line
            st.markdown(
                f"**{ts_str}** &nbsp;·&nbsp; "
                f"Price `{anomaly_row['close']:.4f}` EUR/USD &nbsp;·&nbsp; "
                f"Confidence `{int(anomaly_row['anomaly_score']*100)}%`"
            )

            # Run technical analysis
            try:
                analyzer = TechnicalAnalyzer()
                analysis = analyzer.analyze(raw_row, raw_for_ml)
                risk = analysis['risk_level']
                risk_color = RISK_COLORS.get(risk, '#58a6ff')

                col_left, col_right = st.columns([3, 2])

                with col_left:
                    st.markdown(
                        f"<span class='risk-badge' style='background:{risk_color}22;"
                        f"color:{risk_color};border:1px solid {risk_color}55;'>"
                        f"RISK LEVEL: {risk}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("")

                    for t in analysis['triggers']:
                        icon = SEV_ICONS.get(t['severity'], '⚪')
                        st.markdown(
                            f"<div class='trigger-card'>"
                            f"<b>{icon} {t['name']}</b><br>"
                            f"<span style='color:#58a6ff;font-size:13px;'>{t['value']}</span><br>"
                            f"<span style='color:#8b949e;font-size:12px;'>{t['detail']}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                with col_right:
                    st.markdown("**Indicator Readings**")
                    readings_df = pd.DataFrame(
                        list(analysis['indicator_readings'].items()),
                        columns=['Indicator', 'Value'],
                    )
                    st.dataframe(readings_df, use_container_width=True,
                                 hide_index=True, height=300)

                # ── Deep Analysis (Ollama) ────────────────────────────────────
                st.markdown("")
                deep_key = f"deep_{orig_idx}"

                col_btn, col_clear, _ = st.columns([2, 1, 5])
                with col_btn:
                    run_deep = st.button(
                        "🤖 Deep Analysis (Ollama)",
                        use_container_width=True,
                        key=f"deep_btn_{orig_idx}",
                    )
                with col_clear:
                    if deep_key in st.session_state:
                        if st.button("✕ Clear", use_container_width=True, key="clear_deep"):
                            del st.session_state[deep_key]
                            st.rerun()

                if run_deep:
                    if not ollama_ok:
                        st.error("Ollama is not running. See sidebar for setup instructions.")
                    elif not models:
                        st.warning("No models pulled. Run: `ollama pull llama3.2`")
                    else:
                        with st.spinner(f"Analyzing with {selected_model}..."):
                            try:
                                llm = LLMAnalyzer(model=selected_model)
                                result = llm.analyze(
                                    timestamp=ts_str,
                                    price=float(anomaly_row['close']),
                                    confidence=float(anomaly_row['anomaly_score']),
                                    indicator_readings=analysis['indicator_readings'],
                                    triggers=analysis['triggers'],
                                    risk_level=risk,
                                )
                                st.session_state[deep_key] = result
                            except (ConnectionError, TimeoutError) as e:
                                st.error(str(e))
                            except Exception as e:
                                st.error(f"Unexpected error: {e}")

                if deep_key in st.session_state:
                    st.markdown(
                        f"<div class='deep-box'>"
                        f"<b style='color:#58a6ff;font-size:14px;'>"
                        f"🤖 Deep Analysis — {selected_model}</b><br><br>"
                        f"{st.session_state[deep_key].replace(chr(10), '<br>')}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"Technical analysis error: {e}")

    st.divider()

    # ── Summary stats ─────────────────────────────────────────────────────────
    s1, s2, s3 = st.columns(3)
    s1.metric("Max Confidence", f"{max_conf:.3f}")
    s2.metric("Min Price", f"{results['close'].min():.4f}")
    s3.metric("Max Price", f"{results['close'].max():.4f}")
    st.divider()

    st.markdown(
        "<style>.footer{text-align:center;color:#8b949e;font-size:12px;}</style>"
        "<div class='footer'>SENTINEL FLUX v1.1.0 · Real-time Anomaly Detection + Investigation Agents<br>"
        "<span style='color:#7d8590;font-size:11px;'>Last updated: "
        + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        + "</span></div>",
        unsafe_allow_html=True,
    )

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Run from project root: streamlit run src/dashboard/app.py")
    import traceback
    st.code(traceback.format_exc())
