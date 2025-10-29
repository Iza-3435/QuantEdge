"""Redis-based caching layer with fallback."""
import json
import pickle
from typing import Any, Optional, Callable
from functools import wraps
import hashlib
from src.core.config import settings
from src.core.logging import app_logger
from src.core.metrics import track_cache_access

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    app_logger.warning("Redis not available, using in-memory cache fallback")


class CacheBackend:
    """Abstract cache backend."""

    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        raise NotImplementedError

    def delete(self, key: str):
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError


class RedisCache(CacheBackend):
    """Redis-based cache implementation."""

    def __init__(self):
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis is not installed")

        self.client = redis.Redis(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.db,
            password=settings.redis.password,
            max_connections=settings.redis.max_connections,
            decode_responses=False,
        )
        self.default_ttl = settings.redis.cache_ttl

    def get(self, key: str) -> Optional[Any]:
        try:
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            app_logger.error(f"Redis get error for key {key}: {e}")
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        try:
            ttl = ttl or self.default_ttl
            data = pickle.dumps(value)
            self.client.setex(key, ttl, data)
        except Exception as e:
            app_logger.error(f"Redis set error for key {key}: {e}")

    def delete(self, key: str):
        try:
            self.client.delete(key)
        except Exception as e:
            app_logger.error(f"Redis delete error for key {key}: {e}")

    def exists(self, key: str) -> bool:
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            app_logger.error(f"Redis exists error for key {key}: {e}")
            return False


class MemoryCache(CacheBackend):
    """In-memory cache fallback."""

    def __init__(self, maxsize: int = 1000):
        self._cache = {}
        self.maxsize = maxsize

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        if len(self._cache) >= self.maxsize:
            # Simple LRU: remove first item
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value

    def delete(self, key: str):
        self._cache.pop(key, None)

    def exists(self, key: str) -> bool:
        return key in self._cache


class Cache:
    """Unified cache interface with automatic fallback."""

    def __init__(self):
        try:
            self.backend = RedisCache()
            self.backend_type = "redis"
            app_logger.info("Initialized Redis cache")
        except Exception as e:
            app_logger.warning(f"Failed to initialize Redis, using memory cache: {e}")
            self.backend = MemoryCache()
            self.backend_type = "memory"

    def _make_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from function arguments."""
        key_data = f"{prefix}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        value = self.backend.get(key)
        track_cache_access(self.backend_type, value is not None)
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        self.backend.set(key, value, ttl)

    def delete(self, key: str):
        """Delete key from cache."""
        self.backend.delete(key)

    def cached(
        self,
        prefix: str,
        ttl: Optional[int] = None,
        use_kwargs: bool = True
    ) -> Callable:
        """Decorator for caching function results."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_kwargs = kwargs if use_kwargs else {}
                cache_key = self._make_key(prefix, *args, **cache_kwargs)

                # Try to get from cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    app_logger.debug(f"Cache hit for {func.__name__}")
                    return cached_value

                # Compute and cache
                app_logger.debug(f"Cache miss for {func.__name__}")
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result

            return wrapper
        return decorator


# Global cache instance
cache = Cache()
