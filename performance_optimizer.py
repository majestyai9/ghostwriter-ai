"""
Performance Optimizer for GhostWriter AI
Handles caching, streaming, and background task optimization
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Optional, Tuple, Union
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import queue

logger = logging.getLogger(__name__)


class CacheEntry:
    """Represents a single cache entry with metadata"""
    
    def __init__(self, key: str, value: Any, ttl: int = 3600):
        """
        Initialize cache entry
        
        Args:
            key: Cache key
            value: Cached value
            ttl: Time to live in seconds
        """
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.access_count = 0
        self.last_accessed = time.time()
        self.size = self._calculate_size(value)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of the cached value in bytes"""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (dict, list)):
                return len(json.dumps(value).encode('utf-8'))
            else:
                return len(pickle.dumps(value))
        except:
            return 1024  # Default size if calculation fails
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired"""
        return time.time() - self.created_at > self.ttl
    
    def access(self) -> Any:
        """Access the cached value and update metadata"""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value


class EnhancedCache:
    """
    Enhanced cache with TTL, size limits, and automatic invalidation
    Implements LRU eviction with size constraints
    """
    
    def __init__(self, max_size: int = 100, max_memory_mb: int = 512):
        """
        Initialize enhanced cache
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.total_size = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._lock = threading.RLock()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                
                if entry.is_expired():
                    self._remove_entry(key)
                    self.misses += 1
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return entry.access()
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds (defaults to 3600)
        """
        with self._lock:
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(key, value, ttl or 3600)
            
            # Check if we need to evict entries
            while self._should_evict(entry.size):
                self._evict_lru()
            
            # Add new entry
            self.cache[key] = entry
            self.total_size += entry.size
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was removed, False if not found
        """
        with self._lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all entries matching a pattern
        
        Args:
            pattern: Pattern to match (supports * wildcard)
            
        Returns:
            Number of entries invalidated
        """
        import fnmatch
        
        with self._lock:
            keys_to_remove = [
                key for key in self.cache.keys()
                if fnmatch.fnmatch(key, pattern)
            ]
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            return len(keys_to_remove)
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.total_size = 0
            self.hits = 0
            self.misses = 0
            self.evictions = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
            
            return {
                'entries': len(self.cache),
                'total_size_bytes': self.total_size,
                'total_size_mb': self.total_size / (1024 * 1024),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions,
                'max_size': self.max_size,
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024)
            }
    
    def _should_evict(self, new_size: int) -> bool:
        """Check if we need to evict entries"""
        return (
            len(self.cache) >= self.max_size or
            self.total_size + new_size > self.max_memory_bytes
        )
    
    def _evict_lru(self) -> None:
        """Evict the least recently used entry"""
        if self.cache:
            # Get first item (least recently used)
            key = next(iter(self.cache))
            self._remove_entry(key)
            self.evictions += 1
    
    def _remove_entry(self, key: str) -> None:
        """Remove an entry and update total size"""
        if key in self.cache:
            entry = self.cache[key]
            self.total_size -= entry.size
            del self.cache[key]
    
    def _periodic_cleanup(self) -> None:
        """Periodically clean up expired entries"""
        while True:
            time.sleep(60)  # Check every minute
            
            with self._lock:
                expired_keys = [
                    key for key, entry in self.cache.items()
                    if entry.is_expired()
                ]
                
                for key in expired_keys:
                    self._remove_entry(key)
                    logger.debug(f"Cleaned up expired cache entry: {key}")


def cached_with_ttl(ttl: int = 3600, cache_instance: Optional[EnhancedCache] = None):
    """
    Decorator for caching function results with TTL
    
    Args:
        ttl: Time to live in seconds
        cache_instance: Optional cache instance to use
    """
    cache = cache_instance or _default_cache
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = _create_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        # Add cache control methods
        wrapper.invalidate_cache = lambda: cache.invalidate_pattern(f"{func.__name__}:*")
        wrapper.get_cache_stats = lambda: cache.get_statistics()
        
        return wrapper
    return decorator


class StreamProcessor:
    """
    Handles streaming operations for large data processing
    """
    
    def __init__(self, chunk_size: int = 8192):
        """
        Initialize stream processor
        
        Args:
            chunk_size: Size of chunks for streaming
        """
        self.chunk_size = chunk_size
        self.active_streams: Dict[str, Dict[str, Any]] = {}
    
    async def stream_book_export(
        self,
        book_content: str,
        format: str = 'txt'
    ) -> AsyncIterator[bytes]:
        """
        Stream book export in chunks
        
        Args:
            book_content: Full book content
            format: Export format
            
        Yields:
            Chunks of exported content
        """
        stream_id = hashlib.md5(book_content[:100].encode()).hexdigest()
        
        try:
            # Register stream
            self.active_streams[stream_id] = {
                'started_at': time.time(),
                'bytes_sent': 0,
                'format': format
            }
            
            # Convert content to bytes
            if format == 'txt':
                content_bytes = book_content.encode('utf-8')
            else:
                # For other formats, would need conversion logic
                content_bytes = book_content.encode('utf-8')
            
            # Stream in chunks
            for i in range(0, len(content_bytes), self.chunk_size):
                chunk = content_bytes[i:i + self.chunk_size]
                self.active_streams[stream_id]['bytes_sent'] += len(chunk)
                
                # Simulate async I/O
                await asyncio.sleep(0.001)
                yield chunk
            
        finally:
            # Clean up stream
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
    
    def stream_large_file(self, file_path: Path) -> Iterator[bytes]:
        """
        Stream a large file in chunks
        
        Args:
            file_path: Path to file
            
        Yields:
            File chunks
        """
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                yield chunk
    
    async def stream_generation_progress(
        self,
        generation_id: str,
        progress_queue: asyncio.Queue
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream generation progress updates
        
        Args:
            generation_id: ID of the generation task
            progress_queue: Queue for progress updates
            
        Yields:
            Progress update dictionaries
        """
        stream_info = {
            'generation_id': generation_id,
            'started_at': time.time(),
            'updates_sent': 0
        }
        
        self.active_streams[generation_id] = stream_info
        
        try:
            while True:
                try:
                    # Wait for progress update with timeout
                    update = await asyncio.wait_for(
                        progress_queue.get(),
                        timeout=1.0
                    )
                    
                    if update is None:  # Sentinel for completion
                        break
                    
                    stream_info['updates_sent'] += 1
                    yield update
                    
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield {
                        'type': 'heartbeat',
                        'timestamp': time.time(),
                        'generation_id': generation_id
                    }
        finally:
            # Clean up
            if generation_id in self.active_streams:
                del self.active_streams[generation_id]


class TaskOptimizer:
    """
    Optimizes task execution with pooling and batching
    """
    
    def __init__(self, max_workers: int = 4, batch_size: int = 10):
        """
        Initialize task optimizer
        
        Args:
            max_workers: Maximum number of worker threads
            batch_size: Size for batch operations
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.batch_size = batch_size
        self.pending_batches: Dict[str, list] = {}
        self.batch_locks: Dict[str, threading.Lock] = {}
        self.task_futures: Dict[str, Future] = {}
    
    def submit_task(self, func: Callable, *args, **kwargs) -> Future:
        """
        Submit a task for optimized execution
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Future for the task result
        """
        future = self.executor.submit(func, *args, **kwargs)
        task_id = str(id(future))
        self.task_futures[task_id] = future
        
        # Clean up completed futures periodically
        self._cleanup_completed_futures()
        
        return future
    
    def batch_operation(
        self,
        operation_type: str,
        item: Any,
        batch_processor: Callable
    ) -> Future:
        """
        Add item to batch for processing
        
        Args:
            operation_type: Type of operation for batching
            item: Item to add to batch
            batch_processor: Function to process the batch
            
        Returns:
            Future for the batch result
        """
        if operation_type not in self.batch_locks:
            self.batch_locks[operation_type] = threading.Lock()
            self.pending_batches[operation_type] = []
        
        with self.batch_locks[operation_type]:
            batch = self.pending_batches[operation_type]
            batch.append(item)
            
            if len(batch) >= self.batch_size:
                # Process batch
                batch_to_process = batch.copy()
                self.pending_batches[operation_type] = []
                
                return self.submit_task(batch_processor, batch_to_process)
            else:
                # Return a future that will be completed when batch is processed
                future = Future()
                # Store future with item for later completion
                item._batch_future = future
                return future
    
    def force_batch_processing(self, operation_type: str, batch_processor: Callable) -> Optional[Future]:
        """
        Force processing of pending batch items
        
        Args:
            operation_type: Type of operation
            batch_processor: Function to process the batch
            
        Returns:
            Future for batch result or None if no items pending
        """
        if operation_type not in self.batch_locks:
            return None
        
        with self.batch_locks[operation_type]:
            batch = self.pending_batches.get(operation_type, [])
            if batch:
                batch_to_process = batch.copy()
                self.pending_batches[operation_type] = []
                return self.submit_task(batch_processor, batch_to_process)
        
        return None
    
    def _cleanup_completed_futures(self) -> None:
        """Remove completed futures from tracking"""
        completed = [
            task_id for task_id, future in self.task_futures.items()
            if future.done()
        ]
        
        for task_id in completed:
            del self.task_futures[task_id]
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the executor
        
        Args:
            wait: Whether to wait for pending tasks to complete
        """
        self.executor.shutdown(wait=wait)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get task execution statistics"""
        pending_tasks = sum(1 for f in self.task_futures.values() if not f.done())
        completed_tasks = sum(1 for f in self.task_futures.values() if f.done())
        
        batch_stats = {}
        for op_type, batch in self.pending_batches.items():
            batch_stats[op_type] = len(batch)
        
        return {
            'pending_tasks': pending_tasks,
            'completed_tasks': completed_tasks,
            'total_tasks': len(self.task_futures),
            'pending_batches': batch_stats,
            'worker_threads': self.executor._max_workers
        }


def _create_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Create a cache key from function name and arguments"""
    key_parts = [func_name]
    
    # Add args to key
    for arg in args:
        if isinstance(arg, (str, int, float, bool, type(None))):
            key_parts.append(str(arg))
        else:
            # For complex objects, use their hash
            key_parts.append(str(hash(str(arg))))
    
    # Add kwargs to key
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool, type(None))):
            key_parts.append(f"{k}={v}")
        else:
            key_parts.append(f"{k}={hash(str(v))}")
    
    return ":".join(key_parts)


# Global instances
_default_cache = EnhancedCache(max_size=1000, max_memory_mb=512)
_stream_processor = StreamProcessor()
_task_optimizer = TaskOptimizer()


def get_cache() -> EnhancedCache:
    """Get the global cache instance"""
    return _default_cache


def get_stream_processor() -> StreamProcessor:
    """Get the global stream processor instance"""
    return _stream_processor


def get_task_optimizer() -> TaskOptimizer:
    """Get the global task optimizer instance"""
    return _task_optimizer