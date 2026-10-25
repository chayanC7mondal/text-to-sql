from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Optional, Tuple, Any, List
import uuid
import logging
from datetime import datetime

from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    ClarificationRequest,
    FeedbackRequest,
    QueryResult,
    SQLGeneration,
    QueryPerformance
)
from app.ml.pipeline import MLPipeline
from app.ml.ensemble import SQLEnsemble
from app.ml.feedback_loop import FeedbackLoop
from app.services.database import DatabaseService
from app.services.cache_service import CacheService
from app.services.observability import ObservabilityService
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

router = APIRouter()


# Global dependencies
async def get_ml_pipeline(request: Request) -> MLPipeline:
    return request.app.state.ml_pipeline


async def get_ensemble(request: Request) -> SQLEnsemble:
    if not hasattr(request.app.state, 'ensemble'):
        request.app.state.ensemble = SQLEnsemble()
        await request.app.state.ensemble.initialize()
    return request.app.state.ensemble


async def get_database(request: Request) -> DatabaseService:
    return request.app.state.database


async def get_cache(request: Request) -> CacheService:
    return request.app.state.cache_service


async def get_observability(request: Request) -> ObservabilityService:
    return request.app.state.observability


@router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    ml_pipeline: MLPipeline = Depends(get_ml_pipeline),
    database: DatabaseService = Depends(get_database),
    cache: CacheService = Depends(get_cache),
    observability: ObservabilityService = Depends(get_observability)
):
    """
    Process natural language query and return SQL or clarification.

    This endpoint:
    1. Detects ambiguity in the question
    2. Generates SQL if unambiguous
    3. Validates SQL for safety
    4. Predicts query performance
    5. Executes query against database
    6. Returns formatted results
    """
    query_id = str(uuid.uuid4())
    start_time = datetime.utcnow()

    try:
        # Get query embedding for caching
        query_embedding = ml_pipeline.get_embedding(request.question)

        # Check cache
        if request.enable_cache and settings.enable_cache:
            cached_result = await cache.get(query_id, query_embedding)
            if cached_result:
                observability.record_cache_hit(True)
                return QueryResponse(
                    query_id=query_id,
                    question=request.question,
                    sql_generation=SQLGeneration(
                        sql=cached_result["sql"],
                        model_used=cached_result["model"],
                        confidence=cached_result["confidence"],
                        is_cached=True,
                        cache_similarity=cached_result.get("cache_similarity")
                    ),
                    result=QueryResult(**cached_result["result"]),
                    status="success",
                    timestamp=datetime.utcnow()
                )

        observability.record_cache_hit(False)

        # Detect ambiguity
        ambiguity = await ml_pipeline.detect_ambiguity(request.question)

        if ambiguity.is_ambiguous:
            return QueryResponse(
                query_id=query_id,
                question=request.question,
                ambiguity=ambiguity,
                status="needs_clarification",
                timestamp=datetime.utcnow()
            )

        # Generate SQL using ensemble
        schema_context = await database.get_schema_context()
        ensemble = await get_ensemble(request)
        sql, model_used, confidence = await ensemble.generate_sql(
            request.question,
            schema_context
        )

        # Validate SQL for safety
        is_safe, safety_reason = await validate_sql_safety(sql)
        if not is_safe:
            raise HTTPException(status_code=400, detail=f"Unsafe SQL: {safety_reason}")

        # Predict performance
        performance = await ml_pipeline.predict_query_performance(sql)

        if not performance.recommended:
            logger.warning(f"Query not recommended: {performance.warning}")

        # Execute query
        columns, rows, row_count, execution_time = await database.execute_query(sql)

        # Format result
        result = QueryResult(
            columns=columns,
            rows=rows,
            row_count=row_count,
            execution_time_ms=execution_time,
            summary=generate_summary(request.question, columns, rows, row_count)
        )

        # Cache result
        if request.enable_cache and settings.enable_cache:
            await cache.set(
                query_id,
                {
                    "sql": sql,
                    "model": model_used,
                    "confidence": confidence,
                    "result": result.dict()
                },
                query_embedding
            )

        # Record metrics
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        observability.record_query(
            query_id=query_id,
            status="success",
            model=model_used,
            duration_ms=duration_ms,
            is_cached=False
        )

        return QueryResponse(
            query_id=query_id,
            question=request.question,
            sql_generation=SQLGeneration(
                sql=sql,
                model_used=model_used,
                confidence=confidence,
                is_cached=False
            ),
            performance=performance,
            result=result,
            status="success",
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        observability.record_query(
            query_id=query_id,
            status="error",
            model="unknown",
            duration_ms=duration_ms,
            is_cached=False
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clarification")
async def provide_clarification(
    request: ClarificationRequest,
    database: DatabaseService = Depends(get_database),
    ml_pipeline: MLPipeline = Depends(get_ml_pipeline),
    cache: CacheService = Depends(get_cache),
    observability: ObservabilityService = Depends(get_observability)
):
    """
    Provide clarification for ambiguous query and proceed with SQL generation.
    """
    try:
        # In a real implementation, this would:
        # 1. Retrieve the original question from session/storage
        # 2. Apply the clarification to refine the question
        # 3. Re-run the query pipeline

        # For now, return a placeholder response
        return {
            "status": "success",
            "message": "Clarification received, query will be reprocessed",
            "query_id": request.query_id,
            "selected_option": request.selected_option_id
        }

    except Exception as e:
        logger.error(f"Clarification processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop),
    observability: ObservabilityService = Depends(get_observability)
):
    """
    Submit feedback on query results for continuous learning.
    """
    try:
        feedback_loop.record_feedback(request)

        # Record model performance
        observability.record_model_performance(
            provider="ensemble",  # Would need to extract from query_id
            accuracy=1.0 if request.is_correct else 0.0,
            confidence=request.rating / 5.0
        )

        return {
            "status": "success",
            "message": "Feedback recorded successfully"
        }

    except Exception as e:
        logger.error(f"Feedback submission failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def validate_sql_safety(sql: str) -> Tuple[bool, Optional[str]]:
    """
    Validate SQL for safety using AST parsing.

    Args:
        sql: SQL query to validate

    Returns:
        Tuple of (is_safe, reason)
    """
    try:
        # Parse SQL using sqlglot
        parsed = sqlglot.parse_one(sql, dialect="postgres")

        # Check if it's a SELECT statement
        if parsed.key != "select":
            return False, f"Query must be SELECT, got {parsed.key}"

        # Check for dangerous operations
        sql_lower = sql.lower()
        dangerous_keywords = [
            "drop", "delete", "truncate", "alter", "insert", "update",
            "create", "grant", "revoke", "exec", "execute"
        ]

        for keyword in dangerous_keywords:
            if keyword in sql_lower:
                return False, f"Dangerous keyword detected: {keyword}"

        return True, None

    except Exception as e:
        logger.error(f"SQL validation failed: {e}")
        return False, f"SQL parsing error: {str(e)}"


def generate_summary(
    question: str,
    columns: List[str],
    rows: List[List[Any]],
    row_count: int
) -> str:
    """Generate natural language summary of query results."""
    if row_count == 0:
        return "No results found for your query."

    if row_count == 1:
        return f"Found 1 result: {', '.join(str(v) for v in rows[0][:3])}"

    # Sample first few values
    sample_values = [str(rows[0][i]) for i in range(min(3, len(columns)))]

    return f"Found {row_count:,} results. Sample data: {', '.join(sample_values)}"


# Dependency for feedback loop (would be added to app state)
async def get_feedback_loop(request: Request) -> FeedbackLoop:
    return request.app.state.feedback_loop
