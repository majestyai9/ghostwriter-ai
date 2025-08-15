"""
Vector caching module for RAG system.

Provides LRU caching with TTL for frequently accessed vectors.
"""

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np


class VectorCache:
    """
    LRU cache for frequently accessed vectors.
    
    Features:
    - Thread-safe operations
    - TTL-based expiration
    - Memory-efficient storage
    - Performance metrics tracking
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize vector cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Tuple[Any, float, float]] = {}
        self.access_order: Deque[str] = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cached item if available and not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key in self.cache:
                value, score, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    self.hits += 1
                    # Update access order
                    if key in self.access_order:
                        self.access_order.remove(key)
                    self.access_order.append(key)
                    return value
                else:
                    # Expired, remove from cache
                    del self.cache[key]
                    if key in self.access_order:
                        self.access_order.remove(key)
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, score: float = 1.0):
        """
        Cache an item with optional score.
        
        Args:
            key: Cache key
            value: Value to cache
            score: Optional relevance score
        """
        with self._lock:
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Evict least recently used
                while self.access_order:
                    oldest = self.access_order[0]
                    if oldest in self.cache:
                        del self.cache[oldest]
                        self.access_order.popleft()
                        break
                    self.access_order.popleft()
            
            self.cache[key] = (value, score, time.time())
            if key not in self.access_order:
                self.access_order.append(key)
    
    def clear(self):
        """Clear all cached items."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache metrics
        """
        with self._lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.2%}",
                "total_requests": total
            }
    
    def evict_expired(self):
        """Remove all expired entries from cache."""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, (_, _, timestamp) in self.cache.items()
                if current_time - timestamp >= self.ttl
            ]
            for key in expired_keys:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)