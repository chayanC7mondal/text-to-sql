from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import time

from prometheus_client import Counter, Histogram, Gauge
import mlflow

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class ObservabilityService:
    """
    Comprehensive observability service with Prometheus metrics and MLflow tracking.
    Tracks model performance, query patterns, and system health.
    """

    def __init__(self):
        # Prometheus metrics
        self.query_counter = Counter(
            'text_to_sql_queries_total',
            'Total number of queries processed',
            ['status', 'model']
        )
        self.query_duration = Histogram(
            'text_to_sql_query_duration_seconds',
            'Query processing duration',
            ['stage']
        )
        self.cache_hit_rate = Gauge(
            'text_to_sql_cache_hit_rate',
            'Cache hit rate'
        )
        self.model_accuracy = Gauge(
            'text_to_sql_model_accuracy',
            'Model accuracy by provider',
            ['provider']
        )
        self.active_sessions = Gauge(
            'text_to_sql_active_sessions',
            'Number of active sessions'
        )

        # In-memory metrics for MLflow
        self.query_history: deque = deque(maxlen=10000)
        self.model_performance: Dict[str, List[float]] = defaultdict(list)
        self.anomaly_count: int = 0
        self.anomaly_history: deque = deque(maxlen=1000)

    async def initialize(self):
        """Initialize observability service."""
        try:
            if settings.mlflow_tracking_uri:
                mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                logger.info(f"MLflow tracking initialized: {settings.mlflow_tracking_uri}")

            logger.info("Observability service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize observability: {e}", exc_info=True)
            raise

    async def close(self):
        """Close observability service."""
        logger.info("Observability service closed")

    def record_query(
        self,
        query_id: str,
        status: str,
        model: str,
        duration_ms: float,
        is_cached: bool = False
    ):
        """Record query metrics."""
        self.query_counter.labels(status=status, model=model).inc()
        self.query_duration.labels(stage='total').observe(duration_ms / 1000.0)

        self.query_history.append({
            "query_id": query_id,
            "status": status,
            "model": model,
            "duration_ms": duration_ms,
            "is_cached": is_cached,
            "timestamp": datetime.utcnow().isoformat()
        })

    def record_model_performance(
        self,
        provider: str,
        accuracy: float,
        confidence: float
    ):
        """Record model performance metrics."""
        self.model_performance[provider].append(accuracy)
        self.model_accuracy.labels(provider=provider).set(accuracy)

        # Log to MLflow
        try:
            with mlflow.start_run(nested=True):
                mlflow.log_metric(f"{provider}_accuracy", accuracy)
                mlflow.log_metric(f"{provider}_confidence", confidence)
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    def record_cache_hit(self, hit: bool):
        """Record cache hit/miss."""
        # Update cache hit rate gauge
        history = list(self.query_history)[-100:]  # Last 100 queries
        if history:
            hits = sum(1 for q in history if q.get("is_cached", False))
            rate = hits / len(history)
            self.cache_hit_rate.set(rate)

    def record_anomaly(self, anomaly_type: str, score: float):
        """Record detected anomaly."""
        self.anomaly_count += 1
        self.anomaly_history.append({
            "type": anomaly_type,
            "score": score,
            "timestamp": datetime.utcnow().isoformat()
        })

        logger.warning(f"Anomaly detected: {anomaly_type} (score: {score:.2f})")

    def update_active_sessions(self, count: int):
        """Update active sessions gauge."""
        self.active_sessions.set(count)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        # Calculate average query duration
        durations = [q["duration_ms"] for q in self.query_history if "duration_ms" in q]
        avg_duration = sum(durations) / len(durations) if durations else 0

        # Calculate success rate
        if self.query_history:
            successful = sum(1 for q in self.query_history if q["status"] == "success")
            success_rate = successful / len(self.query_history)
        else:
            success_rate = 0.0

        # Calculate cache hit rate
        if self.query_history:
            cached = sum(1 for q in self.query_history if q.get("is_cached", False))
            cache_rate = cached / len(self.query_history)
        else:
            cache_rate = 0.0

        # Calculate model accuracies
        model_accuracies = {}
        for provider, accuracies in self.model_performance.items():
            if accuracies:
                model_accuracies[provider] = sum(accuracies) / len(accuracies)

        # Recent anomalies (last hour)
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_anomalies = [
            a for a in self.anomaly_history
            if datetime.fromisoformat(a["timestamp"]) > one_hour_ago
        ]

        return {
            "total_queries": len(self.query_history),
            "avg_duration_ms": avg_duration,
            "success_rate": success_rate,
            "cache_hit_rate": cache_rate,
            "model_accuracies": model_accuracies,
            "anomaly_count_last_hour": len(recent_anomalies),
            "total_anomalies": self.anomaly_count
        }

    def log_model_to_mlflow(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        params: Optional[Dict[str, Any]] = None
    ):
        """Log model to MLflow."""
        try:
            with mlflow.start_run():
                mlflow.log_metrics(metrics)
                if params:
                    mlflow.log_params(params)
                mlflow.sklearn.log_model(model, model_name)
                logger.info(f"Model {model_name} logged to MLflow")
        except Exception as e:
            logger.error(f"Failed to log model to MLflow: {e}", exc_info=True)

    def get_query_patterns(self) -> Dict[str, Any]:
        """Analyze query patterns for insights."""
        if not self.query_history:
            return {}

        # Query frequency by hour
        hourly_counts = defaultdict(int)
        for query in self.query_history:
            hour = datetime.fromisoformat(query["timestamp"]).hour
            hourly_counts[hour] += 1

        # Most common errors
        errors = defaultdict(int)
        for query in self.query_history:
            if query["status"] == "error":
                errors[query.get("model", "unknown")] += 1

        return {
            "hourly_distribution": dict(hourly_counts),
            "error_distribution": dict(errors),
            "peak_hour": max(hourly_counts.items(), key=lambda x: x[1])[0] if hourly_counts else None
        }
