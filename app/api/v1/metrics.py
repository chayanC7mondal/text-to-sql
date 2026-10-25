from fastapi import APIRouter, Depends, Request
from typing import Dict, Any, List
import logging

from app.models.schemas import ModelMetrics, SystemMetrics
from app.services.observability import ObservabilityService
from app.services.cache_service import CacheService
from app.ml.feedback_loop import FeedbackLoop
from app.ml.ab_testing import ABTestFramework

logger = logging.getLogger(__name__)

router = APIRouter()


async def get_observability(request: Request) -> ObservabilityService:
    return request.app.state.observability


async def get_cache(request: Request) -> CacheService:
    return request.app.state.cache_service


async def get_feedback_loop(request: Request) -> FeedbackLoop:
    return request.app.state.feedback_loop


async def get_ab_testing(request: Request) -> ABTestFramework:
    return request.app.state.ab_testing


@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics(
    observability: ObservabilityService = Depends(get_observability),
    cache: CacheService = Depends(get_cache),
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop)
) -> SystemMetrics:
    """
    Get comprehensive system metrics including ML model performance.
    """
    # Get observability metrics
    obs_metrics = observability.get_metrics_summary()

    # Get cache stats
    cache_stats = await cache.get_stats()

    # Get feedback stats
    feedback_stats = feedback_loop.get_feedback_stats()

    # Build model metrics
    model_metrics_list = []
    for model_name, accuracy in obs_metrics["model_accuracies"].items():
        model_metrics_list.append(
            ModelMetrics(
                model_name=model_name,
                total_queries=obs_metrics["total_queries"],
                successful_queries=int(obs_metrics["total_queries"] * accuracy),
                accuracy=accuracy,
                avg_execution_time_ms=obs_metrics["avg_duration_ms"],
                cache_hit_rate=cache_stats.get("hit_rate", 0.0)
            )
        )

    return SystemMetrics(
        total_queries=obs_metrics["total_queries"],
        active_sessions=0,  # Would need session tracking
        cache_size=cache_stats.get("cached_embeddings", 0),
        cache_hit_rate=cache_stats.get("hit_rate", 0.0),
        avg_response_time_ms=obs_metrics["avg_duration_ms"],
        model_metrics=model_metrics_list,
        anomaly_count=obs_metrics["anomaly_count_last_hour"]
    )


@router.get("/metrics/query-patterns")
async def get_query_patterns(
    observability: ObservabilityService = Depends(get_observability)
) -> Dict[str, Any]:
    """
    Get query pattern analysis.
    """
    return observability.get_query_patterns()


@router.get("/metrics/feedback")
async def get_feedback_metrics(
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop)
) -> Dict[str, Any]:
    """
    Get feedback and continuous learning metrics.
    """
    return feedback_loop.get_feedback_stats()


@router.get("/metrics/ab-tests")
async def get_ab_test_metrics(
    ab_testing: ABTestFramework = Depends(get_ab_testing)
) -> List[Dict[str, Any]]:
    """
    Get A/B test experiment metrics.
    """
    return ab_testing.get_all_experiments()


@router.get("/metrics/cache")
async def get_cache_metrics(
    cache: CacheService = Depends(get_cache)
) -> Dict[str, Any]:
    """
    Get cache performance metrics.
    """
    return await cache.get_stats()
