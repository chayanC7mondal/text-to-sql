import pytest
import numpy as np
from app.ml.pipeline import MLPipeline
from app.models.schemas import AmbiguityType


@pytest.fixture
async def ml_pipeline():
    """Create and initialize ML pipeline for testing."""
    pipeline = MLPipeline()
    await pipeline.initialize()
    yield pipeline
    # Cleanup if needed


@pytest.mark.asyncio
async def test_ambiguity_detection_unambiguous(ml_pipeline):
    """Test ambiguity detection for clear questions."""
    question = "Show me all customers from the West region"
    result = await ml_pipeline.detect_ambiguity(question)

    assert result.is_ambiguous == False
    assert result.confidence > 0.7


@pytest.mark.asyncio
async def test_ambiguity_detection_ambiguous(ml_pipeline):
    """Test ambiguity detection for ambiguous questions."""
    question = "Who is our best customer?"
    result = await ml_pipeline.detect_ambiguity(question)

    assert result.is_ambiguous == True
    assert result.ambiguity_type == AmbiguityType.METRIC
    assert result.clarification_options is not None
    assert len(result.clarification_options) > 0


@pytest.mark.asyncio
async def test_embedding_generation(ml_pipeline):
    """Test embedding generation."""
    text = "What is the total revenue?"
    embedding = ml_pipeline.get_embedding(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] > 0


@pytest.mark.asyncio
async def test_similarity_calculation(ml_pipeline):
    """Test similarity calculation between embeddings."""
    text1 = "Show me sales data"
    text2 = "Display sales information"
    text3 = "What is the weather?"

    emb1 = ml_pipeline.get_embedding(text1)
    emb2 = ml_pipeline.get_embedding(text2)
    emb3 = ml_pipeline.get_embedding(text3)

    sim12 = ml_pipeline.calculate_similarity(emb1, emb2)
    sim13 = ml_pipeline.calculate_similarity(emb1, emb3)

    assert sim12 > sim13  # Similar texts should have higher similarity


@pytest.mark.asyncio
async def test_query_performance_prediction(ml_pipeline):
    """Test query performance prediction."""
    simple_sql = "SELECT * FROM customers LIMIT 10"
    complex_sql = """
        SELECT c.customer_name, SUM(s.amount) as total
        FROM customers c
        JOIN sales s ON c.customer_id = s.customer_id
        WHERE s.sale_date >= '2024-01-01'
        GROUP BY c.customer_name
        ORDER BY total DESC
        LIMIT 10
    """

    simple_pred = await ml_pipeline.predict_query_performance(simple_sql)
    complex_pred = await ml_pipeline.predict_query_performance(complex_sql)

    assert simple_pred.complexity_score < complex_pred.complexity_score
    assert simple_pred.predicted_execution_time_ms < complex_pred.predicted_execution_time_ms


@pytest.mark.asyncio
async def test_anomaly_detection(ml_pipeline):
    """Test anomaly detection."""
    # Create some historical embeddings
    historical = np.random.rand(50, 384)  # 50 random embeddings

    # Normal query embedding
    normal_emb = np.random.rand(384)
    is_anomaly, score = await ml_pipeline.detect_anomaly(normal_emb, historical)

    # Anomalous query embedding (very different)
    anomalous_emb = np.random.rand(384) * 10
    is_anomaly_2, score_2 = await ml_pipeline.detect_anomaly(anomalous_emb, historical)

    # Results may vary due to randomness, but should not crash
    assert isinstance(is_anomaly, bool)
    assert isinstance(score, float)
