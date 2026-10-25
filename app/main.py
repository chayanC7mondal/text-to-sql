from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import get_settings
from app.api.v1 import query, health, metrics
from app.ml.pipeline import MLPipeline
from app.ml.ensemble import SQLEnsemble
from app.ml.ab_testing import ABTestFramework
from app.ml.feedback_loop import FeedbackLoop
from app.ml.anomaly_detection import AnomalyDetector
from app.services.database import DatabaseService
from app.services.cache_service import CacheService
from app.services.observability import ObservabilityService

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    logger.info("Starting application...")

    # Initialize ML pipeline
    app.state.ml_pipeline = MLPipeline()
    await app.state.ml_pipeline.initialize()

    # Initialize SQL ensemble
    app.state.ensemble = SQLEnsemble()
    await app.state.ensemble.initialize()

    # Initialize A/B testing
    app.state.ab_testing = ABTestFramework()

    # Initialize feedback loop
    app.state.feedback_loop = FeedbackLoop()

    # Initialize anomaly detector
    app.state.anomaly_detector = AnomalyDetector()
    await app.state.anomaly_detector.initialize()

    # Initialize database service
    app.state.database = DatabaseService()
    await app.state.database.initialize()

    # Initialize cache service
    app.state.cache_service = CacheService()
    await app.state.cache_service.initialize()

    # Initialize observability
    app.state.observability = ObservabilityService()
    await app.state.observability.initialize()

    logger.info("Application started successfully")
    yield

    logger.info("Shutting down application...")
    await app.state.database.close()
    await app.state.cache_service.close()
    await app.state.observability.close()
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Advanced Text-to-SQL API",
    description="Production-grade NL2SQL with ML-driven intelligence",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add Prometheus metrics
if settings.enable_prometheus:
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)
        Instrumentator().instrument(app).expose(app, endpoint="/metrics/internal")
    except ImportError:
        logger.warning("prometheus-fastapi-instrumentator not installed, skipping metrics")

# Include routers
app.include_router(query.router, prefix="/api/v1", tags=["query"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing."""
    import time
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Time: {process_time:.3f}s"
    )

    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )
