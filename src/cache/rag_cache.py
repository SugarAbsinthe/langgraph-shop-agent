"""Redis-based RAG result cache for ProductRetriever.

Caches the final formatted retrieval context keyed by (query, top_k) hash.
On Redis miss or Redis-unavailable, falls through gracefully to normal retrieval.
"""

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RAGCache:
    """Thin Redis wrapper for caching product retrieval results.

    Design decisions:
      - Cache key is md5(query + str(top_k)) — the profile is already
        baked into the augmented query by _retrieve_node, so we don't need
        a separate profile hash in the key.
      - TTL defaults to 15 minutes. Profile changes invalidate naturally
        because a different profile produces a different augmented query,
        hence a different cache key.
      - If Redis is unreachable at init time, the cache is a no-op that
        silently falls through to retrieval. No hard dependency on Redis.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0",
                 ttl: int = 900, prefix: str = "rag"):
        self._ttl = ttl
        self._prefix = prefix
        self._redis = None

        try:
            import redis
            self._redis = redis.Redis.from_url(
                redis_url,
                socket_connect_timeout=2,
                socket_timeout=1,
                decode_responses=True,
            )
            self._redis.ping()
            logger.info("RAGCache connected to Redis at %s, TTL=%ds", redis_url, ttl)
        except Exception:
            logger.warning(
                "Redis unavailable at %s — RAG cache disabled, "
                "retrieval will always hit ChromaDB directly", redis_url
            )
            self._redis = None

    def _make_key(self, query: str, top_k: int) -> str:
        """Build a stable cache key from the retrieval parameters."""
        payload = f"{query}|{top_k}"
        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
        return f"{self._prefix}:{digest}"

    def get(self, query: str, top_k: int) -> Optional[str]:
        """Return cached retrieval result or None on miss/unavailable."""
        if self._redis is None:
            return None
        try:
            return self._redis.get(self._make_key(query, top_k))
        except Exception:
            return None

    def set(self, query: str, top_k: int, value: str) -> None:
        """Store retrieval result with TTL. Errors are silently swallowed."""
        if self._redis is None:
            return
        try:
            self._redis.setex(self._make_key(query, top_k), self._ttl, value)
        except Exception:
            pass
