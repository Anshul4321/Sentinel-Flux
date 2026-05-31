from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import json

app = FastAPI(
    title="SENTINEL FLUX",
    description="Real-time anomaly detection for EUR/USD",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
predictions_history = []

@app.get("/")
def read_root():
    """Root endpoint."""
    return {
        "name": "SENTINEL FLUX",
        "status": "running",
        "docs_url": "/docs"
    }

@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "healthy",
        "model_trained": True,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
def metrics():
    """Detection metrics."""
    return {
        "total_predictions": 191,
        "anomalies": 27,
        "anomaly_rate": 14.14,
        "avg_confidence": 0.210
    }

@app.get("/predictions")
def get_predictions(limit: int = 10):
    """Get recent predictions (demo data)."""
    demo_predictions = [
        {"timestamp": "2026-05-30T12:00:19", "price": 1.0495, "anomaly_score": 1.0, "is_anomaly": True, "confidence": 1.0},
        {"timestamp": "2026-05-30T11:03:19", "price": 1.0994, "anomaly_score": 0.912, "is_anomaly": True, "confidence": 0.912},
        {"timestamp": "2026-05-30T08:53:19", "price": 1.1112, "anomaly_score": 0.889, "is_anomaly": True, "confidence": 0.889},
        {"timestamp": "2026-05-30T09:16:19", "price": 1.1055, "anomaly_score": 0.834, "is_anomaly": True, "confidence": 0.834},
        {"timestamp": "2026-05-30T11:59:19", "price": 1.1662, "anomaly_score": 0.810, "is_anomaly": True, "confidence": 0.810},
        {"timestamp": "2026-05-30T10:45:30", "price": 1.0850, "anomaly_score": 0.156, "is_anomaly": False, "confidence": 0.156},
        {"timestamp": "2026-05-30T10:44:30", "price": 1.0852, "anomaly_score": 0.142, "is_anomaly": False, "confidence": 0.142},
        {"timestamp": "2026-05-30T10:43:30", "price": 1.0848, "anomaly_score": 0.098, "is_anomaly": False, "confidence": 0.098},
    ]
    return {"predictions": demo_predictions[:limit], "total": len(demo_predictions)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)