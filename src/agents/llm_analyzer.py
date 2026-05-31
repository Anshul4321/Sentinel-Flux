import requests
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
DEFAULT_MODEL = "llama3.2"

_SYSTEM = (
    "You are an expert forex market analyst specializing in algorithmic anomaly detection. "
    "Provide clear, data-driven, actionable insights in prose. No bullet points."
)


class LLMAnalyzer:
    """Generate deep analysis of anomalies using a local Ollama model (free, offline)."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    @staticmethod
    def is_available() -> bool:
        """Return True if Ollama is reachable."""
        try:
            requests.get("http://localhost:11434/", timeout=3)
            return True
        except Exception:
            return False

    @staticmethod
    def list_models() -> list:
        """Return list of pulled Ollama model names, or [] if unreachable."""
        try:
            resp = requests.get(OLLAMA_TAGS_URL, timeout=5)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            pass
        return []

    def analyze(
        self,
        timestamp: str,
        price: float,
        confidence: float,
        indicator_readings: Dict[str, str],
        triggers: List[Dict],
        risk_level: str,
    ) -> str:
        """
        Send anomaly context to the local Ollama model and return a prose analysis.
        Raises ConnectionError if Ollama is not running.
        """
        trigger_lines = "\n".join(
            f"- {t['name']} ({t['severity']}): {t['value']} — {t['detail']}"
            for t in triggers
        )
        reading_lines = "\n".join(f"- {k}: {v}" for k, v in indicator_readings.items())

        prompt = f"""An ensemble anomaly detection system flagged this EUR/USD data point:

Timestamp : {timestamp}
Price     : {price:.4f} EUR/USD
Confidence: {confidence:.0%}
Risk Level: {risk_level}

Technical Indicators:
{reading_lines}

Triggered Detection Rules:
{trigger_lines}

Provide a concise professional analysis (4–6 sentences) covering:
1. What likely caused this anomaly in market terms
2. What the combined indicator pattern reveals about current conditions
3. Whether this looks like a risk event, trading opportunity, or statistical noise
4. A brief recommendation for how to interpret this signal

Be specific, cite the data, write for an experienced algorithmic trader."""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }

        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Ollama. "
                "Install: https://ollama.ai — then run: ollama pull llama3.2"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError("Ollama request timed out. The model may still be loading.")
