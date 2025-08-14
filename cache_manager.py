"""
Smart caching system for generated content
"""
import hashlib
import json
import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class CacheBackend:
    """Abstract cache backend"""

    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def set(self, key: str, value: Any, expire: int = None):
        raise NotImplementedError

    def delete(self, key: str):
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

class MemoryCache(CacheBackend):
    """In-memory cache backend with LRU eviction and thread safety"""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()  # Thread safety lock

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry['expire'] and entry['expire'] < time.time():
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
                    return None
                self.access_times[key] = time.time()
                return entry['value']
            return None

    def set(self, key: str, value: Any, expire: int = None):
        with self.lock:
            # LRU eviction if needed
            if len(self.cache) >= self.max_size and key not in self.cache:
                if self.access_times:  # Check if there are any items to evict
                    lru_key = min(self.access_times, key=self.access_times.get)
                    del self.cache[lru_key]
                    del self.access_times[lru_key]

            self.cache[key] = {
                'value': value,
                'expire': time.time() + expire if expire else None,
                'created': time.time()
            }
            self.access_times[key] = time.time()

    def delete(self, key: str):
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]

    def exists(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry['expire'] and entry['expire'] < time.time():
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
                    return False
                return True
            return False

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

class RedisCache(CacheBackend):
    """Redis cache backend"""

    def __init__(self, host='localhost', port=6379, db=0, password=None):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")

        self.client = redis.StrictRedis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False
        )
        self.logger = logging.getLogger(__name__)

    def get(self, key: str) -> Optional[Any]:
        try:
            value = self.client.get(key)
            if value:
                return pickle.loads(value)
        except Exception as e:
            self.logger.error(f"Redis get error: {e}")
        return None

    def set(self, key: str, value: Any, expire: int = None):
        try:
            serialized = pickle.dumps(value)
            if expire:
                self.client.setex(key, expire, serialized)
            else:
                self.client.set(key, serialized)
        except Exception as e:
            self.logger.error(f"Redis set error: {e}")

    def delete(self, key: str):
        try:
            self.client.delete(key)
        except Exception as e:
            self.logger.error(f"Redis delete error: {e}")

    def exists(self, key: str) -> bool:
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            self.logger.error(f"Redis exists error: {e}")
            return False

    def clear(self):
        try:
            self.client.flushdb()
        except Exception as e:
            self.logger.error(f"Redis clear error: {e}")

class FileCache(CacheBackend):
    """File-based cache backend"""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict:
        if self.index_file.exists():
            try:
                with open(self.index_file) as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_index(self):
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f)

    def _get_file_path(self, key: str) -> Path:
        # Use hash to avoid filesystem issues with special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> Optional[Any]:
        file_path = self._get_file_path(key)
        if file_path.exists():
            # Check expiration
            if key in self.index:
                expire = self.index[key].get('expire')
                if expire and expire < time.time():
                    self.delete(key)
                    return None

            try:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.error(f"File cache read error: {e}")
        return None

    def set(self, key: str, value: Any, expire: int = None):
        file_path = self._get_file_path(key)
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)

            self.index[key] = {
                'expire': time.time() + expire if expire else None,
                'created': time.time(),
                'file': str(file_path.name)
            }
            self._save_index()
        except Exception as e:
            self.logger.error(f"File cache write error: {e}")

    def delete(self, key: str):
        file_path = self._get_file_path(key)
        if file_path.exists():
            file_path.unlink()
        if key in self.index:
            del self.index[key]
            self._save_index()

    def exists(self, key: str) -> bool:
        if key in self.index:
            expire = self.index[key].get('expire')
            if expire and expire < time.time():
                self.delete(key)
                return False
            return self._get_file_path(key).exists()
        return False

    def clear(self):
        for file in self.cache_dir.glob("*.cache"):
            file.unlink()
        self.index.clear()
        self._save_index()

class CacheManager:
    """Smart cache manager with multiple backends and strategies"""

    def __init__(self, backend: str = 'memory', **backend_kwargs):
        """
        Initialize cache manager
        
        Args:
            backend: Cache backend ('memory', 'redis', 'file')
            **backend_kwargs: Backend-specific configuration
        """
        self.logger = logging.getLogger(__name__)

        if backend == 'redis' and REDIS_AVAILABLE:
            self.backend = RedisCache(**backend_kwargs)
        elif backend == 'file':
            self.backend = FileCache(**backend_kwargs)
        else:
            self.backend = MemoryCache(**backend_kwargs)

        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }

    def create_key(self, prefix: str, **params) -> str:
        """
        Create a cache key from parameters
        
        Args:
            prefix: Key prefix (e.g., 'title', 'chapter')
            **params: Parameters to include in key
            
        Returns:
            Cache key string
        """
        # Sort params for consistent keys
        sorted_params = sorted(params.items())
        param_str = json.dumps(sorted_params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:12]
        return f"{prefix}:{param_hash}"

    def get(self, key: str, default=None) -> Optional[Any]:
        """Get value from cache"""
        value = self.backend.get(key)
        if value is not None:
            self.stats['hits'] += 1
            self.logger.debug(f"Cache hit: {key}")
            return value
        else:
            self.stats['misses'] += 1
            self.logger.debug(f"Cache miss: {key}")
            return default

    def set(self, key: str, value: Any, expire: int = 3600):
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            expire: Expiration time in seconds (default 1 hour)
        """
        self.backend.set(key, value, expire)
        self.stats['sets'] += 1
        self.logger.debug(f"Cache set: {key} (expire: {expire}s)")

    def delete(self, key: str):
        """Delete value from cache"""
        self.backend.delete(key)
        self.stats['deletes'] += 1
        self.logger.debug(f"Cache delete: {key}")

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        return self.backend.exists(key)

    def clear(self):
        """Clear all cache"""
        self.backend.clear()
        self.logger.info("Cache cleared")

    def invalidate_pattern(self, pattern: str):
        """
        Invalidate cache keys matching pattern
        
        Args:
            pattern: Pattern to match (e.g., 'chapter:*')
        """
        # This is backend-specific and more complex
        # For now, just log
        self.logger.info(f"Pattern invalidation requested: {pattern}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total * 100) if total > 0 else 0

        return {
            **self.stats,
            'hit_rate': f"{hit_rate:.2f}%",
            'backend': self.backend.__class__.__name__
        }

# Decorator for caching function results
def cached(expire: int = 3600, key_prefix: str = None):
    """
    Decorator to cache function results
    
    Args:
        expire: Cache expiration in seconds
        key_prefix: Custom key prefix (default: function name)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get or create cache manager
            if not hasattr(wrapper, 'cache'):
                wrapper.cache = CacheManager()

            # Create cache key
            prefix = key_prefix or func.__name__
            cache_key = wrapper.cache.create_key(
                prefix,
                args=str(args),
                kwargs=str(kwargs)
            )

            # Try to get from cache
            result = wrapper.cache.get(cache_key)
            if result is not None:
                return result

            # Generate and cache
            result = func(*args, **kwargs)
            wrapper.cache.set(cache_key, result, expire)
            return result

        wrapper.__name__ = func.__name__
        return wrapper
    return decorator

# Global cache instance
cache_manager = None

def initialize_cache(backend: str = 'memory', **kwargs):
    """Initialize global cache manager"""
    global cache_manager
    cache_manager = CacheManager(backend, **kwargs)
    return cache_manager

def get_cache() -> CacheManager:
    """Get global cache manager"""
    global cache_manager
    if cache_manager is None:
        cache_manager = CacheManager()
    return cache_manager
