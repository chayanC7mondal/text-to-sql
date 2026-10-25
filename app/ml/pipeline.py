from typing import Optional, Dict, List, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
import logging
from sklearn.ensemble import IsolationForest
from collections import defaultdict

from app.core.config import get_settings
from app.models.schemas import AmbiguityDetection, AmbiguityType, QueryPerformance

settings = get_settings()
logger = logging.getLogger(__name__)


class MLPipeline:
    """
    ML Pipeline for Text-to-SQL system with advanced features:
    - Embedding-based ambiguity detection
    - Query performance prediction
    - Anomaly detection
    - Similarity-based caching
    """

    def __init__(self):
        self.embedding_model: Optional[SentenceTransformer] = None
        self.ambiguity_classifier: Optional[Any] = None
        self.performance_predictor: Optional[Any] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.ambiguity_embeddings: Optional[np.ndarray] = None
        self.ambiguity_labels: Optional[List[str]] = None

        # Domain-specific ambiguity patterns
        self.ambiguous_terms = {
            "metric": ["best", "top", "worst", "highest", "lowest", "active", "churned", "valuable"],
            "timeframe": ["recent", "last", "current", "previous", "ytd", "qtd", "mtd"],
            "entity": ["customer", "product", "category", "region", "segment"],
            "aggregation": ["total", "average", "sum", "count", "distinct"]
        }

        # Query complexity features
        self.complexity_weights = {
            "joins": 0.3,
            "subqueries": 0.25,
            "aggregations": 0.2,
            "filters": 0.15,
            "group_by": 0.1
        }

    async def initialize(self):
        """Initialize ML models and load pre-trained components."""
        logger.info("Initializing ML pipeline...")

        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer(settings.embedding_model)
            logger.info(f"Loaded embedding model: {settings.embedding_model}")

            # Initialize anomaly detector
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            logger.info("Initialized anomaly detector")

            # Try to load pre-trained models if they exist
            try:
                self.ambiguity_classifier = joblib.load(settings.ambiguity_detector_model)
                logger.info("Loaded pre-trained ambiguity classifier")
            except FileNotFoundError:
                logger.warning("Ambiguity classifier not found, will use rule-based approach")

            try:
                self.performance_predictor = joblib.load(settings.query_performance_model)
                logger.info("Loaded pre-trained performance predictor")
            except FileNotFoundError:
                logger.warning("Performance predictor not found, will use heuristic approach")

            # Initialize ambiguity pattern embeddings
            await self._initialize_ambiguity_patterns()

            logger.info("ML pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ML pipeline: {e}", exc_info=True)
            raise

    async def _initialize_ambiguity_patterns(self):
        """Initialize embeddings for common ambiguity patterns."""
        ambiguity_examples = [
            "What is our best selling product?",  # metric ambiguity
            "Who are our top customers?",  # metric ambiguity
            "Show me recent sales",  # timeframe ambiguity
            "What is the total revenue?",  # metric ambiguity
            "Which category performs best?",  # metric ambiguity
            "How many active users do we have?",  # metric ambiguity
            "What's the average order value?",  # metric ambiguity
            "Who churned last month?",  # metric + timeframe ambiguity
        ]

        self.ambiguity_embeddings = self.embedding_model.encode(
            ambiguity_examples,
            show_progress_bar=False
        )
        self.ambiguity_labels = ambiguity_examples
        logger.info(f"Initialized {len(ambiguity_examples)} ambiguity pattern embeddings")

    async def detect_ambiguity(
        self,
        question: str,
        threshold: float = 0.75
    ) -> AmbiguityDetection:
        """
        Detect ambiguity in natural language questions using embedding similarity.

        Args:
            question: User's natural language question
            threshold: Similarity threshold for ambiguity detection

        Returns:
            AmbiguityDetection object with detection results
        """
        question_embedding = self.embedding_model.encode(
            [question],
            show_progress_bar=False
        )[0]

        # Calculate similarity with known ambiguity patterns
        similarities = np.dot(
            self.ambiguity_embeddings,
            question_embedding
        ) / (
            np.linalg.norm(self.ambiguity_embeddings, axis=1) *
            np.linalg.norm(question_embedding)
        )

        max_similarity = np.max(similarities)
        max_idx = np.argmax(similarities)

        # Check for ambiguous terms in the question
        detected_terms = []
        ambiguity_type = None

        question_lower = question.lower()
        for term_type, terms in self.ambiguous_terms.items():
            for term in terms:
                if term in question_lower:
                    detected_terms.append(term)
                    if ambiguity_type is None:
                        ambiguity_type = AmbiguityType(term_type)

        is_ambiguous = max_similarity > threshold or len(detected_terms) > 0
        confidence = max(max_similarity if is_ambiguous else 1.0 - max_similarity, 0.5)

        if is_ambiguous and detected_terms:
            # Generate clarification options
            clarification_options = self._generate_clarification_options(
                detected_terms,
                ambiguity_type
            )
        else:
            clarification_options = None

        return AmbiguityDetection(
            is_ambiguous=is_ambiguous,
            ambiguity_type=ambiguity_type,
            confidence=confidence,
            clarification_options=clarification_options,
            detected_terms=detected_terms if detected_terms else None
        )

    def _generate_clarification_options(
        self,
        detected_terms: List[str],
        ambiguity_type: AmbiguityType
    ) -> List[Dict[str, str]]:
        """Generate clarification options based on detected ambiguous terms."""
        options = []

        if ambiguity_type == AmbiguityType.METRIC:
            options = [
                {
                    "id": "metric_revenue",
                    "label": "Total Revenue",
                    "description": "Based on total monetary value of sales",
                    "example_sql": "SELECT SUM(amount) FROM sales"
                },
                {
                    "id": "metric_units",
                    "label": "Units Sold",
                    "description": "Based on number of items sold",
                    "example_sql": "SELECT COUNT(*) FROM sales"
                },
                {
                    "id": "metric_orders",
                    "label": "Number of Orders",
                    "description": "Based on count of unique orders",
                    "example_sql": "SELECT COUNT(DISTINCT order_id) FROM sales"
                },
                {
                    "id": "metric_profit",
                    "label": "Profit Margin",
                    "description": "Based on revenue minus costs",
                    "example_sql": "SELECT SUM(revenue - cost) FROM sales"
                }
            ]
        elif ambiguity_type == AmbiguityType.TIMEFRAME:
            options = [
                {
                    "id": "time_7d",
                    "label": "Last 7 Days",
                    "description": "Rolling 7-day window",
                    "example_sql": "WHERE date >= NOW() - INTERVAL '7 days'"
                },
                {
                    "id": "time_30d",
                    "label": "Last 30 Days",
                    "description": "Rolling 30-day window",
                    "example_sql": "WHERE date >= NOW() - INTERVAL '30 days'"
                },
                {
                    "id": "time_ytd",
                    "label": "Year to Date",
                    "description": "From January 1st of current year",
                    "example_sql": "WHERE date >= DATE_TRUNC('year', NOW())"
                },
                {
                    "id": "time_custom",
                    "label": "Custom Range",
                    "description": "Specify exact date range",
                    "example_sql": "WHERE date BETWEEN 'start' AND 'end'"
                }
            ]

        return options

    async def predict_query_performance(
        self,
        sql: str,
        schema_stats: Optional[Dict[str, Any]] = None
    ) -> QueryPerformance:
        """
        Predict query execution time and resource usage.

        Args:
            sql: SQL query to analyze
            schema_stats: Statistics about database schema

        Returns:
            QueryPerformance prediction
        """
        # Extract features from SQL
        features = self._extract_sql_features(sql)

        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(features)

        # Use ML model if available, otherwise use heuristics
        if self.performance_predictor is not None:
            try:
                feature_vector = self._build_feature_vector(features, schema_stats)
                prediction = self.performance_predictor.predict([feature_vector])[0]
                predicted_time = max(prediction, 10)  # Minimum 10ms
            except Exception as e:
                logger.warning(f"Performance prediction failed, using heuristics: {e}")
                predicted_time = self._heuristic_prediction(features, complexity_score)
        else:
            predicted_time = self._heuristic_prediction(features, complexity_score)

        # Predict rows (rough estimate based on complexity)
        predicted_rows = int(1000 * (1 + complexity_score * 10))

        # Determine if query is recommended
        recommended = predicted_time < settings.max_query_execution_time * 1000
        warning = None if recommended else f"Query may take longer than {settings.max_query_execution_time}s"

        return QueryPerformance(
            predicted_execution_time_ms=predicted_time,
            predicted_rows=predicted_rows,
            complexity_score=complexity_score,
            recommended=recommended,
            warning=warning
        )

    def _extract_sql_features(self, sql: str) -> Dict[str, int]:
        """Extract structural features from SQL query."""
        sql_lower = sql.lower()
        return {
            "joins": sql_lower.count("join"),
            "subqueries": sql_lower.count("select") - 1,  # Subtract main SELECT
            "aggregations": sum(
                sql_lower.count(func) for func in ["sum(", "count(", "avg(", "max(", "min("]
            ),
            "filters": sql_lower.count("where") + sql_lower.count("having"),
            "group_by": sql_lower.count("group by"),
            "order_by": sql_lower.count("order by"),
            "unions": sql_lower.count("union"),
            "length": len(sql)
        }

    def _calculate_complexity_score(self, features: Dict[str, int]) -> float:
        """Calculate normalized complexity score (0-1)."""
        score = 0.0
        for feature, weight in self.complexity_weights.items():
            feature_value = features.get(feature, 0)
            # Normalize (cap at 5 for most features)
            normalized = min(feature_value / 5.0, 1.0)
            score += normalized * weight
        return min(score, 1.0)

    def _build_feature_vector(
        self,
        features: Dict[str, int],
        schema_stats: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """Build feature vector for ML model."""
        vector = [
            features.get("joins", 0),
            features.get("subqueries", 0),
            features.get("aggregations", 0),
            features.get("filters", 0),
            features.get("group_by", 0),
            features.get("order_by", 0),
            features.get("unions", 0),
            features.get("length", 0) / 1000.0,  # Normalize length
        ]

        if schema_stats:
            vector.extend([
                schema_stats.get("total_rows", 0) / 1e6,  # Normalize to millions
                schema_stats.get("avg_table_size", 0) / 1e6,
                schema_stats.get("table_count", 0) / 20.0,  # Normalize
            ])

        return vector

    def _heuristic_prediction(
        self,
        features: Dict[str, int],
        complexity_score: float
    ) -> float:
        """Heuristic prediction when ML model is not available."""
        base_time = 50  # Base 50ms
        complexity_multiplier = 1 + complexity_score * 10
        join_penalty = features.get("joins", 0) * 20
        subquery_penalty = features.get("subqueries", 0) * 30

        predicted = base_time * complexity_multiplier + join_penalty + subquery_penalty
        return min(predicted, 30000)  # Cap at 30 seconds

    async def detect_anomaly(
        self,
        query_embedding: np.ndarray,
        historical_embeddings: Optional[np.ndarray] = None
    ) -> Tuple[bool, float]:
        """
        Detect anomalous queries using isolation forest.

        Args:
            query_embedding: Embedding of the current query
            historical_embeddings: Historical query embeddings for context

        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        if historical_embeddings is None or len(historical_embeddings) < 10:
            # Not enough data for anomaly detection
            return False, 0.0

        try:
            # Fit anomaly detector on historical data
            self.anomaly_detector.fit(historical_embeddings)

            # Predict anomaly
            prediction = self.anomaly_detector.predict([query_embedding])[0]
            score = self.anomaly_detector.score_samples([query_embedding])[0]

            # prediction: -1 for anomaly, 1 for normal
            is_anomaly = prediction == -1

            return is_anomaly, float(abs(score))

        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            return False, 0.0

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using the loaded model."""
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not initialized")
        return self.embedding_model.encode([text], show_progress_bar=False)[0]

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return float(np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        ))
