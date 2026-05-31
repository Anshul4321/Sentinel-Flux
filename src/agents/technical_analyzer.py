import pandas as pd
import numpy as np
from typing import Dict, Any, List


class TechnicalAnalyzer:
    """Analyze which technical indicators triggered a detected anomaly."""

    def analyze(self, raw_row: pd.Series, all_raw_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a single anomaly using raw (un-normalized) feature values.

        Args:
            raw_row: Raw feature values for the anomaly point (un-normalized)
            all_raw_df: Raw features for all ML-ready rows (for dataset statistics)

        Returns:
            Dict with 'triggers', 'risk_level', 'indicator_readings', 'trigger_count'
        """
        triggers: List[Dict] = []
        readings: Dict[str, str] = {}

        # ── RSI breach ──────────────────────────────────────────────────────
        if 'rsi_14' in raw_row.index and pd.notna(raw_row['rsi_14']):
            rsi = float(raw_row['rsi_14'])
            readings['RSI (14)'] = f"{rsi:.1f}"
            if rsi > 70:
                triggers.append({
                    'name': 'RSI Overbought',
                    'value': f"RSI = {rsi:.1f} (>70)",
                    'detail': "Overbought conditions — potential reversal or momentum exhaustion",
                    'severity': 'HIGH',
                })
            elif rsi < 30:
                triggers.append({
                    'name': 'RSI Oversold',
                    'value': f"RSI = {rsi:.1f} (<30)",
                    'detail': "Oversold conditions — potential reversal or bounce",
                    'severity': 'HIGH',
                })

        # ── Bollinger Band breach ────────────────────────────────────────────
        bb_cols = ['bb_upper', 'bb_lower', 'close']
        if all(c in raw_row.index for c in bb_cols):
            close = float(raw_row['close'])
            bb_upper = raw_row['bb_upper']
            bb_lower = raw_row['bb_lower']
            if pd.notna(bb_upper) and pd.notna(bb_lower):
                bb_upper = float(bb_upper)
                bb_lower = float(bb_lower)
                readings['BB Upper'] = f"{bb_upper:.4f}"
                readings['BB Lower'] = f"{bb_lower:.4f}"
                if close > bb_upper:
                    triggers.append({
                        'name': 'BB Upper Band Breach',
                        'value': f"Price {close:.4f} > Upper {bb_upper:.4f}",
                        'detail': "Price exceeded upper Bollinger Band — extreme upward deviation (2 std devs)",
                        'severity': 'HIGH',
                    })
                elif close < bb_lower:
                    triggers.append({
                        'name': 'BB Lower Band Breach',
                        'value': f"Price {close:.4f} < Lower {bb_lower:.4f}",
                        'detail': "Price broke below lower Bollinger Band — extreme downward deviation (2 std devs)",
                        'severity': 'HIGH',
                    })

        # ── Z-score returns breach ───────────────────────────────────────────
        if 'zscore_returns' in raw_row.index and pd.notna(raw_row['zscore_returns']):
            zr = float(raw_row['zscore_returns'])
            readings['Z-Score Returns'] = f"{zr:.2f} std devs"
            if abs(zr) > 2.0:
                direction = "upward spike" if zr > 0 else "downward drop"
                severity = 'CRITICAL' if abs(zr) > 3.0 else 'HIGH'
                triggers.append({
                    'name': 'Abnormal Return',
                    'value': f"Z = {zr:.2f} std devs",
                    'detail': f"Statistically rare {direction} — {abs(zr):.1f} std devs from 20-period rolling mean",
                    'severity': severity,
                })

        # ── Z-score volume breach ────────────────────────────────────────────
        if 'zscore_volume' in raw_row.index and pd.notna(raw_row['zscore_volume']):
            zv = float(raw_row['zscore_volume'])
            readings['Z-Score Volume'] = f"{zv:.2f} std devs"
            if abs(zv) > 2.0:
                direction = "surge" if zv > 0 else "drought"
                triggers.append({
                    'name': 'Volume Anomaly',
                    'value': f"Z = {zv:.2f} std devs",
                    'detail': f"Unusual volume {direction} ({abs(zv):.1f} std devs from rolling mean)",
                    'severity': 'MEDIUM',
                })

        # ── Volatility spike ─────────────────────────────────────────────────
        if 'volatility_20' in raw_row.index and 'volatility_20' in all_raw_df.columns:
            vol = raw_row['volatility_20']
            if pd.notna(vol):
                vol = float(vol)
                vol_series = all_raw_df['volatility_20'].dropna()
                if len(vol_series) > 1:
                    vol_mean = float(vol_series.mean())
                    vol_std = float(vol_series.std())
                    vol_z = (vol - vol_mean) / (vol_std + 1e-8)
                    pct_above = ((vol - vol_mean) / (vol_mean + 1e-8)) * 100
                    readings['Volatility (20)'] = f"{vol:.5f} ({pct_above:+.0f}% vs avg)"
                    if vol_z > 1.5:
                        severity = 'CRITICAL' if vol_z > 2.5 else 'HIGH'
                        triggers.append({
                            'name': 'Volatility Spike',
                            'value': f"+{pct_above:.0f}% above dataset average",
                            'detail': f"Rolling volatility is {vol_z:.1f} std devs above the dataset mean",
                            'severity': severity,
                        })

        # ── MACD crossover (sign change vs prior row) ────────────────────────
        if 'macd_hist' in raw_row.index and 'macd_hist' in all_raw_df.columns:
            macd_h = raw_row['macd_hist']
            if pd.notna(macd_h):
                macd_h = float(macd_h)
                readings['MACD Histogram'] = f"{macd_h:.6f}"
                try:
                    if raw_row.name in all_raw_df.index:
                        loc = all_raw_df.index.get_loc(raw_row.name)
                        if loc > 0:
                            prev_macd = float(all_raw_df.iloc[loc - 1]['macd_hist'])
                            if pd.notna(prev_macd):
                                crossed = (prev_macd > 0 and macd_h < 0) or (prev_macd < 0 and macd_h > 0)
                                if crossed:
                                    direction = "bullish" if macd_h > 0 else "bearish"
                                    triggers.append({
                                        'name': f'MACD Crossover ({direction.capitalize()})',
                                        'value': f"Hist: {prev_macd:.6f} → {macd_h:.6f}",
                                        'detail': f"{direction.capitalize()} momentum signal — MACD histogram sign change",
                                        'severity': 'MEDIUM',
                                    })
                except Exception:
                    pass

        # ── SMA-20 deviation ─────────────────────────────────────────────────
        if all(c in raw_row.index for c in ['close', 'sma_20']):
            close = float(raw_row['close'])
            sma20 = raw_row['sma_20']
            if pd.notna(sma20):
                sma20 = float(sma20)
                dev_pct = abs(close - sma20) / (sma20 + 1e-8) * 100
                readings['SMA-20'] = f"{sma20:.4f} ({dev_pct:+.2f}% dev)"
                if dev_pct > 0.15:
                    direction = "above" if close > sma20 else "below"
                    triggers.append({
                        'name': 'SMA-20 Deviation',
                        'value': f"Price {dev_pct:.2f}% {direction} SMA-20",
                        'detail': "Significant deviation from 20-period simple moving average",
                        'severity': 'LOW',
                    })

        # ── Isolation Forest (always present for confirmed anomalies) ────────
        triggers.append({
            'name': 'Isolation Forest',
            'value': 'ML ensemble vote',
            'detail': "Model identified this point as structurally isolated from the normal data cluster",
            'severity': 'HIGH',
        })

        # Add SMA-50 and ATR to readings even if not triggers
        if 'sma_50' in raw_row.index and pd.notna(raw_row['sma_50']):
            readings['SMA-50'] = f"{float(raw_row['sma_50']):.4f}"
        if 'atr_14' in raw_row.index and pd.notna(raw_row['atr_14']):
            readings['ATR (14)'] = f"{float(raw_row['atr_14']):.6f}"

        # ── Overall risk level ───────────────────────────────────────────────
        severities = [t['severity'] for t in triggers]
        if severities.count('CRITICAL') >= 1 or severities.count('HIGH') >= 3:
            risk_level = 'CRITICAL'
        elif severities.count('HIGH') >= 2:
            risk_level = 'HIGH'
        elif 'HIGH' in severities:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

        human_triggers = [t for t in triggers if t['name'] != 'Isolation Forest']

        return {
            'triggers': triggers,
            'risk_level': risk_level,
            'indicator_readings': readings,
            'trigger_count': len(human_triggers),
        }
