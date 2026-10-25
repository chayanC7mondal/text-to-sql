from typing import Dict, List, Optional, Any, Tuple
import logging
from collections import defaultdict
import numpy as np

from app.core.config import get_settings
from app.ml.llm_providers import OpenAIProvider, AnthropicProvider, LocalProvider

settings = get_settings()
logger = logging.getLogger(__name__)


class SQLEnsemble:
    """
    Ensemble model for SQL generation with multiple LLM providers.
    Implements weighted averaging and dynamic model selection.
    """

    def __init__(self):
        self.providers: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {
            "openai": settings.model_weights_openai,
            "anthropic": settings.model_weights_anthropic,
            "local": settings.model_weights_local
        }
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self.current_weights = self.weights.copy()

    async def initialize(self):
        """Initialize all LLM providers."""
        logger.info("Initializing SQL ensemble...")

        if settings.openai_api_key:
            self.providers["openai"] = OpenAIProvider(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                temperature=settings.openai_temperature,
                max_tokens=settings.openai_max_tokens
            )
            logger.info("Initialized OpenAI provider")

        if settings.anthropic_api_key:
            self.providers["anthropic"] = AnthropicProvider(
                api_key=settings.anthropic_api_key,
                model=settings.anthropic_model,
                temperature=settings.anthropic_temperature,
                max_tokens=settings.anthropic_max_tokens
            )
            logger.info("Initialized Anthropic provider")

        # Local provider (always available, uses smaller model)
        self.providers["local"] = LocalProvider()
        logger.info("Initialized local provider")

        logger.info(f"Ensemble initialized with {len(self.providers)} providers")

    async def generate_sql(
        self,
        question: str,
        schema_context: str,
        complexity_score: float = 0.5
    ) -> Tuple[str, str, float]:
        """
        Generate SQL using ensemble of models.

        Args:
            question: Natural language question
            schema_context: Database schema context
            complexity_score: Query complexity (0-1)

        Returns:
            Tuple of (sql, model_used, confidence)
        """
        # Select provider based on complexity and availability
        selected_provider = self._select_provider(complexity_score)

        logger.info(f"Selected provider: {selected_provider} for complexity {complexity_score:.2f}")

        try:
            provider = self.providers[selected_provider]
            sql, confidence = await provider.generate_sql(question, schema_context)

            # Record performance
            self.performance_history[selected_provider].append(confidence)

            return sql, selected_provider, confidence

        except Exception as e:
            logger.error(f"Provider {selected_provider} failed: {e}", exc_info=True)

            # Fallback to next best provider
            for provider_name in self._get_provider_priority():
                if provider_name != selected_provider and provider_name in self.providers:
                    try:
                        logger.info(f"Falling back to {provider_name}")
                        provider = self.providers[provider_name]
                        sql, confidence = await provider.generate_sql(question, schema_context)
                        return sql, provider_name, confidence * 0.9  # Penalty for fallback
                    except Exception as fallback_error:
                        logger.error(f"Fallback provider {provider_name} also failed: {fallback_error}")

            raise RuntimeError("All providers failed to generate SQL")

    def _select_provider(self, complexity_score: float) -> str:
        """
        Select provider based on query complexity and historical performance.

        For high complexity queries, prefer more capable models.
        For low complexity queries, use faster models.
        """
        available_providers = list(self.providers.keys())

        if not available_providers:
            raise RuntimeError("No providers available")

        # High complexity: prefer OpenAI or Anthropic
        if complexity_score > 0.7:
            if "openai" in available_providers:
                return "openai"
            if "anthropic" in available_providers:
                return "anthropic"

        # Medium complexity: prefer Anthropic
        if complexity_score > 0.4:
            if "anthropic" in available_providers:
                return "anthropic"
            if "openai" in available_providers:
                return "openai"

        # Low complexity: use local or fastest available
        if "local" in available_providers:
            return "local"

        # Default to first available
        return available_providers[0]

    def _get_provider_priority(self) -> List[str]:
        """Get provider priority order."""
        return ["openai", "anthropic", "local"]

    async def generate_ensemble_sql(
        self,
        question: str,
        schema_context: str,
        complexity_score: float = 0.5
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate SQL using true ensemble (multiple providers + voting).

        This is more expensive but provides better accuracy for critical queries.
        """
        results = []
        total_weight = 0.0

        for provider_name, provider in self.providers.items():
            try:
                sql, confidence = await provider.generate_sql(question, schema_context)
                weight = self.current_weights.get(provider_name, 0.0)

                results.append({
                    "provider": provider_name,
                    "sql": sql,
                    "confidence": confidence,
                    "weight": weight
                })
                total_weight += weight

                logger.info(f"Provider {provider_name}: confidence={confidence:.2f}, weight={weight:.2f}")

            except Exception as e:
                logger.warning(f"Provider {provider_name} failed in ensemble: {e}")

        if not results:
            raise RuntimeError("No providers succeeded in ensemble")

        # Select best SQL based on weighted confidence
        best_result = max(results, key=lambda x: x["confidence"] * x["weight"])

        # Calculate ensemble confidence
        ensemble_confidence = sum(
            r["confidence"] * r["weight"] for r in results
        ) / total_weight if total_weight > 0 else 0.0

        logger.info(f"Ensemble selected {best_result['provider']} with confidence {ensemble_confidence:.2f}")

        return best_result["sql"], {
            "provider": best_result["provider"],
            "confidence": ensemble_confidence,
            "all_results": results
        }

    def update_weights(self, provider_name: str, success: bool):
        """
        Update provider weights based on success/failure.
        Implements simple reinforcement learning.
        """
        if provider_name not in self.performance_history:
            return

        learning_rate = 0.1

        if success:
            # Increase weight slightly
            self.current_weights[provider_name] = min(
                self.current_weights[provider_name] + learning_rate,
                1.0
            )
        else:
            # Decrease weight
            self.current_weights[provider_name] = max(
                self.current_weights[provider_name] - learning_rate,
                0.1
            )

        # Normalize weights
        total = sum(self.current_weights.values())
        for key in self.current_weights:
            self.current_weights[key] /= total

        logger.info(f"Updated weights: {self.current_weights}")

    def get_provider_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each provider."""
        stats = {}
        for provider_name, history in self.performance_history.items():
            if history:
                stats[provider_name] = {
                    "avg_confidence": np.mean(history),
                    "total_queries": len(history),
                    "current_weight": self.current_weights.get(provider_name, 0.0)
                }
        return stats
