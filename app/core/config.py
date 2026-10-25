from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://readonly_user:readonly_password@localhost:5432/ecommerce"
    database_pool_size: int = 20
    database_max_overflow: int = 10

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl: int = 3600
    redis_max_connections: int = 20

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo-preview"
    openai_temperature: float = 0.1
    openai_max_tokens: int = 2048

    # Anthropic
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-opus-20240229"
    anthropic_temperature: float = 0.1
    anthropic_max_tokens: int = 2048

    # ML Settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    query_performance_model: str = "models/query_performance.pkl"
    ambiguity_detector_model: str = "models/ambiguity_detector.pkl"
    anomaly_detector_model: str = "models/anomaly_detector.pkl"

    # Ensemble Weights
    model_weights_openai: float = 0.4
    model_weights_anthropic: float = 0.4
    model_weights_local: float = 0.2

    # Caching
    enable_cache: bool = True
    cache_similarity_threshold: float = 0.85
    cache_max_size: int = 10000

    # A/B Testing
    enable_ab_testing: bool = True
    ab_test_ratio: float = 0.1
    ab_test_min_samples: int = 100

    # Observability
    mlflow_tracking_uri: str = "http://localhost:5000"
    enable_prometheus: bool = True
    prometheus_port: int = 8000

    # Query Safety
    max_query_execution_time: int = 30
    max_query_rows_returned: int = 10000
    enable_query_cost_prediction: bool = True

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
