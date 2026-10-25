from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Anomaly detection system for query patterns and security threats.
    Uses isolation forest and time-series analysis.
    """

    def __init__(self):
        self.isolation_forest: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.query_history: deque = deque(maxlen=10000)
        self.feature_history: List[np.ndarray] = []
        self.anomaly_history: deque = deque(maxlen=1000)
        self.attack_patterns: List[str] = [
            "DROP TABLE",
            "DELETE FROM",
            "TRUNCATE",
            "ALTER TABLE",
            "UPDATE SET",
            "INSERT INTO",
            "EXEC",
            "xp_cmdshell",
            "sp_executesql",
            "--",
            "/*",
            "*/",
            "OR 1=1",
            "UNION SELECT",
            "script:",
            "javascript:",
            "<script"
        ]

    async def initialize(self):
        """Initialize anomaly detection models."""
        logger.info("Initializing anomaly detector...")

        try:
            # Load pre-trained model if available
            try:
                self.isolation_forest = joblib.load(settings.anomaly_detector_model)
                logger.info("Loaded pre-trained anomaly detector")
            except FileNotFoundError:
                # Initialize new model
                self.isolation_forest = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=100
                )
                logger.info("Initialized new anomaly detector")

            self.scaler = StandardScaler()

            logger.info("Anomaly detector initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize anomaly detector: {e}", exc_info=True)
            raise

    def extract_features(self, sql: str, question: str) -> np.ndarray:
        """
        Extract features from SQL and question for anomaly detection.

        Args:
            sql: Generated SQL query
            question: Original natural language question

        Returns:
            Feature vector
        """
        sql_lower = sql.lower()
        question_lower = question.lower()

        features = [
            # SQL structure features
            len(sql),
            sql.count("join"),
            sql.count("where"),
            sql.count("select"),
            sql.count("from"),
            sql.count("group by"),
            sql.count("order by"),
            sql.count("having"),

            # Question features
            len(question),
            question.count("?"),
            question.count("what"),
            question.count("how"),
            question.count("which"),
            question.count("show"),

            # Complexity features
            sql.count("("),
            sql.count(")"),
            sql.count(","),
            sql.count("and"),
            sql.count("or"),

            # Potential attack indicators
            any(pattern.lower() in sql_lower for pattern in self.attack_patterns),
            sql.count(";"),
            sql.count("--"),
            sql.count("/*"),
        ]

        return np.array(features, dtype=float)

    def detect_sql_injection(self, sql: str) -> Tuple[bool, List[str]]:
        """
        Detect potential SQL injection attacks.

        Args:
            sql: SQL query to check

        Returns:
            Tuple of (is_attack, detected_patterns)
        """
        sql_lower = sql.lower()
        detected_patterns = []

        for pattern in self.attack_patterns:
            if pattern.lower() in sql_lower:
                detected_patterns.append(pattern)

        is_attack = len(detected_patterns) > 0

        if is_attack:
            logger.warning(f"Potential SQL injection detected: {detected_patterns}")

        return is_attack, detected_patterns

    def detect_anomaly(
        self,
        sql: str,
        question: str,
        query_embedding: Optional[np.ndarray] = None
    ) -> Tuple[bool, float, str]:
        """
        Detect anomalous queries using isolation forest.

        Args:
            sql: Generated SQL query
            question: Original natural language question
            query_embedding: Query embedding for additional detection

        Returns:
            Tuple of (is_anomaly, anomaly_score, reason)
        """
        # Check for SQL injection first
        is_attack, patterns = self.detect_sql_injection(sql)
        if is_attack:
            return True, 1.0, f"SQL injection pattern detected: {patterns}"

        # Extract features
        features = self.extract_features(sql, question)

        # If we don't have enough history, use rule-based detection
        if len(self.feature_history) < 50:
            return self._rule_based_detection(sql, question, features)

        # Scale features
        if self.scaler is not None:
            features_scaled = self.scaler.transform([features])[0]
        else:
            features_scaled = features

        # Predict anomaly
        try:
            prediction = self.isolation_forest.predict([features_scaled])[0]
            score = self.isolation_forest.score_samples([features_scaled])[0]

            is_anomaly = prediction == -1
            anomaly_score = abs(score)

            if is_anomaly:
                reason = "Statistical anomaly detected in query pattern"
            else:
                reason = "No anomaly detected"

            # Record for training
            self.feature_history.append(features)
            if len(self.feature_history) > 1000:
                self.feature_history = self.feature_history[-1000:]

            return is_anomaly, anomaly_score, reason

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}", exc_info=True)
            return False, 0.0, "Detection failed"

    def _rule_based_detection(
        self,
        sql: str,
        question: str,
        features: np.ndarray
    ) -> Tuple[bool, float, str]:
        """
        Rule-based anomaly detection when model is not trained.

        Args:
            sql: SQL query
            question: Natural language question
            features: Feature vector

        Returns:
            Tuple of (is_anomaly, anomaly_score, reason)
        """
        reasons = []

        # Check for overly complex queries
        if features[1] > 5:  # Too many joins
            reasons.append("Too many joins")
        if features[0] > 5000:  # Too long
            reasons.append("Query too long")

        # Check for suspicious patterns
        if "DROP" in sql.upper() or "DELETE" in sql.upper():
            reasons.append("Destructive operation detected")

        # Check for mismatch between question complexity and SQL complexity
        if len(question) < 20 and features[0] > 1000:
            reasons.append("Complex SQL for simple question")

        is_anomaly = len(reasons) > 0
        anomaly_score = min(len(reasons) * 0.2, 1.0)
        reason = "; ".join(reasons) if reasons else "No anomaly detected"

        return is_anomaly, anomaly_score, reason

    def detect_time_series_anomaly(
        self,
        query_times: List[float],
        window_size: int = 10
    ) -> Tuple[bool, float]:
        """
        Detect anomalies in query execution time time-series.

        Args:
            query_times: List of recent query execution times
            window_size: Size of rolling window

        Returns:
            Tuple of (is_anomaly, z_score)
        """
        if len(query_times) < window_size:
            return False, 0.0

        # Calculate rolling statistics
        recent_times = query_times[-window_size:]
        mean = np.mean(recent_times)
        std = np.std(recent_times)

        if std == 0:
            return False, 0.0

        # Calculate z-score for latest query
        latest_time = query_times[-1]
        z_score = abs((latest_time - mean) / std)

        is_anomaly = z_score > 3.0  # 3-sigma rule

        if is_anomaly:
            logger.warning(f"Time-series anomaly detected: z-score={z_score:.2f}")

        return is_anomaly, z_score

    def record_anomaly(
        self,
        query_id: str,
        anomaly_type: str,
        score: float,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Record detected anomaly for analysis.

        Args:
            query_id: Query identifier
            anomaly_type: Type of anomaly
            score: Anomaly score
            details: Additional details
        """
        anomaly_entry = {
            "query_id": query_id,
            "type": anomaly_type,
            "score": score,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }

        self.anomaly_history.append(anomaly_entry)

        logger.warning(f"Anomaly recorded: {anomaly_type} (score: {score:.2f})")

    def get_anomaly_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get anomaly statistics for the last N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            Anomaly statistics
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        recent_anomalies = [
            a for a in self.anomaly_history
            if datetime.fromisoformat(a["timestamp"]) > cutoff
        ]

        # Count by type
        by_type = defaultdict(int)
        for anomaly in recent_anomalies:
            by_type[anomaly["type"]] += 1

        return {
            "total_anomalies": len(recent_anomalies),
            "by_type": dict(by_type),
            "avg_score": np.mean([a["score"] for a in recent_anomalies]) if recent_anomalies else 0.0,
            "max_score": max([a["score"] for a in recent_anomalies]) if recent_anomalies else 0.0
        }

    def retrain_model(self):
        """Retrain the anomaly detection model with accumulated data."""
        if len(self.feature_history) < 100:
            logger.warning("Not enough data to retrain anomaly detector")
            return

        try:
            # Prepare training data
            X = np.array(self.feature_history)

            # Fit scaler
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)

            # Retrain isolation forest
            self.isolation_forest.fit(X_scaled)

            # Save model
            joblib.dump(self.isolation_forest, settings.anomaly_detector_model)

            logger.info("Anomaly detector retrained successfully")

        except Exception as e:
            logger.error(f"Failed to retrain anomaly detector: {e}", exc_info=True)
