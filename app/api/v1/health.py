from fastapi import APIRouter, Depends, Request
from typing import Dict, Any
import logging
from datetime import datetime

from app.services.database import DatabaseService
from app.services.cache_service import CacheService
from app.services.observability import ObservabilityService

logger = logging.getLogger(__name__)

router = APIRouter()


async def get_database(request: Request) -> DatabaseService:
    return request.app.state.database


async def get_cache(request: Request) -> CacheService:
    return request.app.state.cache_service


async def get_observability(request: Request) -> ObservabilityService:
    return request.app.state.observability


@router.get("/health")
async def health_check(
    database: DatabaseService = Depends(get_database),
    cache: CacheService = Depends(get_cache)
) -> Dict[str, Any]:
    """
    Health check endpoint.
    Checks database and cache connectivity.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }

    # Check database
    db_healthy = await database.check_connection()
    health_status["services"]["database"] = {
        "status": "healthy" if db_healthy else "unhealthy",
        "connected": db_healthy
    }

    # Check cache
    try:
        await cache.redis.ping()
        cache_healthy = True
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        cache_healthy = False

    health_status["services"]["cache"] = {
        "status": "healthy" if cache_healthy else "unhealthy",
        "connected": cache_healthy
    }

    # Overall status
    if not db_healthy or not cache_healthy:
        health_status["status"] = "degraded"

    return health_status


@router.get("/health/ready")
async def readiness_check(
    database: DatabaseService = Depends(get_database),
    cache: CacheService = Depends(get_cache)
) -> Dict[str, Any]:
    """
    Readiness check - more stringent than health check.
    """
    db_healthy = await database.check_connection()

    try:
        await cache.redis.ping()
        cache_healthy = True
    except Exception:
        cache_healthy = False

    is_ready = db_healthy and cache_healthy

    return {
        "ready": is_ready,
        "database": db_healthy,
        "cache": cache_healthy
    }
