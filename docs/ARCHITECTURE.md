# SENTINEL FLUX - Architecture Design

## System Overview
Real-time anomaly detection for forex price data (EUR/USD).
Detects volatility spikes, trend breaks, and statistical outliers in < 100ms latency.

## Core Components

### 1. Data Ingestion
- **Source:** Yahoo Finance API (or mock for testing)
- **Input:** 1-min OHLCV candlesticks for EUR/USD
- **Rate:** ~1 update/min (or higher for tick data)
- **Tech:** Python streaming with requests/websocket

### 2. Feature Pipeline
- Compute rolling statistics (SMA, EMA, volatility, RSI)
- Normalize features for ML models
- Handle missing data gracefully
- **Tech:** Pandas, NumPy

### 3. ML Anomaly Detector
- **Primary:** Isolation Forest (fast, no tuning)
- **Secondary:** Z-score + IQR for statistical anomalies
- **Ensemble:** Vote-based decision (both agree = alert)
- **Retraining:** Weekly on new 7-day window

### 4. Backend API
- **Tech:** FastAPI
- **Endpoints:**
  - `POST /ingest` - Accept new data
  - `GET /predictions/latest` - Current anomaly score
  - `GET /history?hours=24` - Historical data + predictions
  - `GET /metrics` - System health
- **Rate limiting:** 1000 req/min

### 5. Data Persistence
- **PostgreSQL:** Historical data, anomalies (queryable)
- **Redis:** Latest predictions (< 1 sec cache)
- **Retention:** 6 months rolling window

### 6. Frontend Dashboard
- **Tech:** React + Recharts
- **Live chart:** EUR/USD price + anomaly overlay
- **Alerts table:** Recent anomalies with severity
- **Stats:** Detection rate, false positives, uptime

### 7. Monitoring & Observability
- Structured logging (JSON)
- Prometheus metrics (prediction latency, model accuracy)
- Health check endpoint
- Alert on model staleness

## Data Flow
Yahoo Finance API
↓
Data Ingestion (validate, deduplicate)
↓
Feature Pipeline (compute indicators)
↓
ML Models (Isolation Forest + Z-score)
↓
PostgreSQL (history) + Redis (latest)
↓
FastAPI Backend (serve predictions)
↓
React Dashboard (visualize)
↓
Monitoring (log anomalies, metrics)

## Non-Functional Requirements
- **Latency:** < 100ms from data arrival to prediction
- **Availability:** 99.5% uptime
- **Accuracy:** < 10% false positive rate
- **Scalability:** 10x data rate without architecture change
- **Testing:** Unit + integration + performance tests

## Deployment
- **Local:** Docker Compose (PostgreSQL, Redis, API, Dashboard)
- **Production-Ready:** Kubernetes manifests (future)
- **CI/CD:** GitHub Actions (lint, test, build, push)