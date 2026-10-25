from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class QuerySource(str, Enum):
    """Source of the query for A/B testing."""
    CONTROL = "control"
    TREATMENT = "treatment"


class AmbiguityType(str, Enum):
    """Types of ambiguity detected in queries."""
    METRIC = "metric"
    TIMEFRAME = "timeframe"
    ENTITY = "entity"
    AGGREGATION = "aggregation"


class QueryRequest(BaseModel):
    """Request model for natural language to SQL conversion."""
    question: str = Field(..., min_length=1, max_length=1000, description="Natural language question")
    session_id: Optional[str] = Field(None, description="Session identifier for context")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the query")
    enable_cache: Optional[bool] = Field(True, description="Enable caching for this query")
    source: Optional[QuerySource] = Field(None, description="Query source for A/B testing")

    @validator('question')
    def question_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()


class ClarificationOption(BaseModel):
    """Single clarification option for ambiguous queries."""
    id: str = Field(..., description="Unique identifier for the option")
    label: str = Field(..., description="Human-readable label")
    description: str = Field(..., description="Detailed description of what this option means")
    example_sql: Optional[str] = Field(None, description="Example SQL for this interpretation")


class AmbiguityDetection(BaseModel):
    """Ambiguity detection result."""
    is_ambiguous: bool = Field(..., description="Whether the query is ambiguous")
    ambiguity_type: Optional[AmbiguityType] = Field(None, description="Type of ambiguity detected")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of ambiguity detection")
    clarification_options: Optional[List[ClarificationOption]] = Field(None, description="Options for clarification")
    detected_terms: Optional[List[str]] = Field(None, description="Terms that triggered ambiguity detection")


class SQLGeneration(BaseModel):
    """SQL generation result."""
    sql: str = Field(..., description="Generated SQL query")
    model_used: str = Field(..., description="Which model generated this SQL")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the generated SQL")
    explanation: Optional[str] = Field(None, description="Natural language explanation of the SQL")
    is_cached: bool = Field(False, description="Whether this result came from cache")
    cache_similarity: Optional[float] = Field(None, description="Similarity score if cached")


class QueryPerformance(BaseModel):
    """Query performance prediction."""
    predicted_execution_time_ms: float = Field(..., description="Predicted execution time in milliseconds")
    predicted_rows: Optional[int] = Field(None, description="Predicted number of rows returned")
    complexity_score: float = Field(..., ge=0.0, le=1.0, description="Query complexity score")
    recommended: bool = Field(True, description="Whether query is recommended to execute")
    warning: Optional[str] = Field(None, description="Warning if query is not recommended")


class QueryResult(BaseModel):
    """Query execution result."""
    columns: List[str] = Field(..., description="Column names")
    rows: List[List[Any]] = Field(..., description="Result rows")
    row_count: int = Field(..., description="Number of rows returned")
    execution_time_ms: float = Field(..., description="Actual execution time in milliseconds")
    summary: Optional[str] = Field(None, description="Natural language summary of results")


class QueryResponse(BaseModel):
    """Complete query response."""
    query_id: str = Field(..., description="Unique query identifier")
    question: str = Field(..., description="Original question")
    ambiguity: Optional[AmbiguityDetection] = Field(None, description="Ambiguity detection result")
    sql_generation: Optional[SQLGeneration] = Field(None, description="SQL generation result")
    performance: Optional[QueryPerformance] = Field(None, description="Performance prediction")
    result: Optional[QueryResult] = Field(None, description="Query execution result")
    status: Literal["success", "needs_clarification", "error"] = Field(..., description="Query status")
    error: Optional[str] = Field(None, description="Error message if status is error")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class ClarificationRequest(BaseModel):
    """Request to provide clarification for ambiguous query."""
    query_id: str = Field(..., description="Query identifier from previous response")
    selected_option_id: str = Field(..., description="Selected clarification option ID")


class FeedbackRequest(BaseModel):
    """Request to provide feedback on query results."""
    query_id: str = Field(..., description="Query identifier")
    rating: int = Field(..., ge=1, le=5, description="User rating (1-5)")
    is_correct: bool = Field(..., description="Whether the result was correct")
    comments: Optional[str] = Field(None, description="Additional comments")
    correct_sql: Optional[str] = Field(None, description="Correct SQL if the generated one was wrong")


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    model_name: str = Field(..., description="Model identifier")
    total_queries: int = Field(..., description="Total number of queries")
    successful_queries: int = Field(..., description="Number of successful queries")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Accuracy score")
    avg_execution_time_ms: float = Field(..., description="Average execution time")
    cache_hit_rate: float = Field(..., ge=0.0, le=1.0, description="Cache hit rate")


class SystemMetrics(BaseModel):
    """System-wide metrics."""
    total_queries: int = Field(..., description="Total queries processed")
    active_sessions: int = Field(..., description="Number of active sessions")
    cache_size: int = Field(..., description="Current cache size")
    cache_hit_rate: float = Field(..., ge=0.0, le=1.0, description="Overall cache hit rate")
    avg_response_time_ms: float = Field(..., description="Average response time")
    model_metrics: List[ModelMetrics] = Field(..., description="Per-model metrics")
    anomaly_count: int = Field(..., description="Number of anomalies detected in last hour")
