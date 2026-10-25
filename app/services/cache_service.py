from typing import Optional, Dict, Any, List
import numpy as np
import json
import logging
from datetime import datetime, timedelta

import redis.asyncio as redis

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class CacheService:
    """
    Intelligent caching service with embedding-based similarity search.
    Implements ML-driven cache eviction and warm-up strategies.
    """

    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.access_patterns: Dict[str, List[datetime]] = {}

    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=settings.redis_max_connections
            )
            await self.redis.ping()
            logger.info("Cache service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize cache service: {e}", exc_info=True)
            raise

    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            logger.info("Cache service closed")

    async def get(
        self,
        query_id: str,
        query_embedding: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result by query ID or similarity search.

        Args:
            query_id: Unique query identifier
            query_embedding: Query embedding for similarity search

        Returns:
            Cached result if found, None otherwise
        """
        # Try exact match first
        cached = await self.redis.get(f"query:{query_id}")
        if cached:
            result = json.loads(cached)
            await self._record_access(query_id)
            logger.info(f"Cache hit (exact): {query_id}")
            return result

        # Try similarity search if embedding provided
        if query_embedding is not None and settings.enable_cache:
            similar_result = await self._similarity_search(query_embedding)
            if similar_result:
                logger.info(f"Cache hit (similarity): {similar_result['query_id']}")
                return similar_result

        logger.info(f"Cache miss: {query_id}")
        return None

    async def set(
        self,
        query_id: str,
        result: Dict[str, Any],
        query_embedding: Optional[np.ndarray] = None,
        ttl: Optional[int] = None
    ):
        """
        Cache a query result.

        Args:
            query_id: Unique query identifier
            result: Query result to cache
            query_embedding: Query embedding for similarity search
            ttl: Time to live in seconds
        """
        try:
            ttl = ttl or settings.redis_cache_ttl

            # Store exact match
            await self.redis.setex(
                f"query:{query_id}",
                ttl,
                json.dumps(result)
            )

            # Store embedding for similarity search
            if query_embedding is not None:
                await self._store_embedding(query_id, query_embedding, ttl)

            # Initialize access pattern
            self.access_patterns[query_id] = [datetime.utcnow()]

            logger.info(f"Cached result: {query_id}")

        except Exception as e:
            logger.error(f"Failed to cache result: {e}", exc_info=True)

    async def _store_embedding(
        self,
        query_id: str,
        embedding: np.ndarray,
        ttl: int
    ):
        """Store query embedding for similarity search."""
        try:
            # Store as binary
            embedding_bytes = embedding.tobytes()
            await self.redis.setex(
                f"embedding:{query_id}",
                ttl,
                embedding_bytes
            )

            # Store in local cache for faster access
            self.embedding_cache[query_id] = embedding

        except Exception as e:
            logger.error(f"Failed to store embedding: {e}", exc_info=True)

    async def _similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Search for similar queries using embedding similarity.

        Args:
            query_embedding: Query embedding to search for
            top_k: Number of similar queries to consider

        Returns:
            Most similar cached result if similarity above threshold
        """
        try:
            # Get all cached embeddings
            keys = await self.redis.keys("embedding:*")
            if not keys:
                return None

            similarities = []
            for key in keys:
                query_id = key.split(":")[1]

                # Try local cache first
                if query_id in self.embedding_cache:
                    cached_embedding = self.embedding_cache[query_id]
                else:
                    # Load from Redis
                    embedding_bytes = await self.redis.get(key)
                    if not embedding_bytes:
                        continue
                    cached_embedding = np.frombuffer(embedding_bytes)
                    self.embedding_cache[query_id] = cached_embedding

                # Calculate similarity
                similarity = float(np.dot(
                    query_embedding,
                    cached_embedding
                ) / (
                    np.linalg.norm(query_embedding) *
                    np.linalg.norm(cached_embedding)
                ))

                similarities.append((query_id, similarity))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Check if best match is above threshold
            if similarities and similarities[0][1] >= settings.cache_similarity_threshold:
                best_query_id = similarities[0][0]
                cached = await self.redis.get(f"query:{best_query_id}")
                if cached:
                    result = json.loads(cached)
                    result["query_id"] = best_query_id
                    result["cache_similarity"] = similarities[0][1]
                    await self._record_access(best_query_id)
                    return result

            return None

        except Exception as e:
            logger.error(f"Similarity search failed: {e}", exc_info=True)
            return None

    async def _record_access(self, query_id: str):
        """Record access pattern for ML-driven eviction."""
        if query_id not in self.access_patterns:
            self.access_patterns[query_id] = []

        self.access_patterns[query_id].append(datetime.utcnow())

        # Keep only last 100 accesses
        if len(self.access_patterns[query_id]) > 100:
            self.access_patterns[query_id] = self.access_patterns[query_id][-100:]

    async def evict_ml_driven(self):
        """
        ML-driven cache eviction based on access patterns.
        Evicts items with low access frequency and recency.
        """
        try:
            current_time = datetime.utcnow()
            to_evict = []

            for query_id, access_times in self.access_patterns.items():
                if not access_times:
                    continue

                # Calculate access frequency (accesses per hour)
                time_span = (current_time - access_times[0]).total_seconds() / 3600
                frequency = len(access_times) / max(time_span, 1)

                # Calculate recency (hours since last access)
                recency = (current_time - access_times[-1]).total_seconds() / 3600

                # Eviction criteria: low frequency AND high recency
                if frequency < 0.1 and recency > 24:
                    to_evict.append(query_id)

            # Evict identified items
            for query_id in to_evict:
                await self.redis.delete(f"query:{query_id}")
                await self.redis.delete(f"embedding:{query_id}")
                if query_id in self.embedding_cache:
                    del self.embedding_cache[query_id]
                if query_id in self.access_patterns:
                    del self.access_patterns[query_id]

            logger.info(f"Evicted {len(to_evict)} items using ML-driven policy")

        except Exception as e:
            logger.error(f"ML-driven eviction failed: {e}", exc_info=True)

    async def warm_up(self, common_queries: List[str]):
        """
        Warm up cache with common queries.

        Args:
            common_queries: List of common query embeddings to pre-cache
        """
        logger.info(f"Warming up cache with {len(common_queries)} queries")
        # Implementation would pre-generate and cache common queries
        # This is a placeholder for the actual implementation

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = await self.redis.info("stats")
            return {
                "total_keys": info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / max(
                    info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0),
                    1
                ),
                "cached_embeddings": len(self.embedding_cache),
                "tracked_patterns": len(self.access_patterns)
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}", exc_info=True)
            return {}
