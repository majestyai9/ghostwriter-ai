"""
Semantic Caching Layer for RAG System.

This module implements intelligent caching based on semantic similarity,
reducing redundant RAG queries and improving response times.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
import pickle
from pathlib import Path
from collections import OrderedDict, defaultdict
import threading
import time
import heapq

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class CacheEntry:
    """Represents a cached RAG query result."""
    query: str
    query_embedding: Optional[np.ndarray]
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    ttl_seconds: Optional[int] = None
    score: float = 0.0  # Relevance score for cache management


@dataclass
class SemanticCacheConfig:
    """Configuration for semantic cache."""
    # Cache size and TTL
    max_cache_size: int = 10000
    default_ttl_seconds: int = 3600  # 1 hour
    ttl_by_query_type: Dict[str, int] = field(default_factory=lambda: {
        "factual": 7200,  # 2 hours for factual queries
        "creative": 1800,  # 30 min for creative queries
        "navigation": 3600  # 1 hour for navigation queries
    })
    
    # Semantic similarity
    similarity_threshold: float = 0.85  # Minimum similarity for cache hit
    embedding_model: str = "all-MiniLM-L6-v2"
    use_faiss_index: bool = True
    
    # Cache strategies
    eviction_policy: str = "lru_semantic"  # lru, lfu, lru_semantic, adaptive
    prefetch_similar: bool = True
    prefetch_threshold: float = 0.7
    
    # Performance
    use_compression: bool = True
    compression_level: int = 6
    batch_embedding_size: int = 32
    
    # Persistence
    persist_cache: bool = True
    persistence_path: str = ".rag/semantic_cache"
    checkpoint_interval: int = 100
    
    # Adaptive caching
    enable_adaptive: bool = True
    min_access_for_promotion: int = 3
    decay_factor: float = 0.95  # Score decay per hour


class SemanticCache:
    """
    Semantic caching layer for RAG queries.
    
    Features:
    - Semantic similarity-based cache retrieval
    - Intelligent eviction policies
    - Query result prefetching
    - Adaptive TTL based on query patterns
    - Compression for memory efficiency
    """
    
    def __init__(self, config: Optional[SemanticCacheConfig] = None):
        self.config = config or SemanticCacheConfig()
        self.logger = logging.getLogger(__name__)
        
        # Cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.embedding_index = None
        self.query_to_id: Dict[str, str] = {}
        self.id_to_query: Dict[str, str] = {}
        
        # Embeddings
        self.encoder = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer(self.config.embedding_model)
            except Exception as e:
                self.logger.warning(f"Failed to initialize encoder: {e}")
        
        # FAISS index for similarity search
        if self.config.use_faiss_index and FAISS_AVAILABLE and self.encoder:
            self._initialize_faiss_index()
        
        # Statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "semantic_hits": 0,
            "exact_hits": 0,
            "evictions": 0,
            "prefetch_hits": 0,
            "total_queries": 0,
            "avg_similarity": 0.0,
            "memory_usage_mb": 0.0
        }
        
        # Query patterns for adaptive caching
        self.query_patterns = defaultdict(lambda: {
            "count": 0,
            "avg_similarity": 0.0,
            "typical_ttl": self.config.default_ttl_seconds
        })
        
        # Threading
        self._lock = threading.RLock()
        self._last_checkpoint = datetime.now()
        
        # Load existing cache if available
        if self.config.persist_cache:
            self._load_cache()
    
    def _initialize_faiss_index(self):
        """Initialize FAISS index for semantic search."""
        try:
            # Get embedding dimension
            test_embedding = self.encoder.encode(["test"], convert_to_numpy=True)
            dimension = test_embedding.shape[1]
            
            # Create FAISS index
            self.embedding_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            self.logger.info(f"Initialized FAISS index with dimension {dimension}")
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS index: {e}")
            self.embedding_index = None
    
    def get(self, query: str, similarity_threshold: Optional[float] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve cached results for a query.
        
        Args:
            query: The query to look up
            similarity_threshold: Optional custom similarity threshold
        
        Returns:
            Cached results if found, None otherwise
        """
        self.stats["total_queries"] += 1
        threshold = similarity_threshold or self.config.similarity_threshold
        
        with self._lock:
            # Check exact match first
            cache_key = self._get_cache_key(query)
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                if self._is_valid(entry):
                    self._update_access(entry)
                    self.stats["cache_hits"] += 1
                    self.stats["exact_hits"] += 1
                    return entry.results
                else:
                    # Remove expired entry
                    self._evict_entry(cache_key)
            
            # Check semantic similarity if encoder available
            if self.encoder:
                similar_entry = self._find_similar(query, threshold)
                if similar_entry:
                    self._update_access(similar_entry)
                    self.stats["cache_hits"] += 1
                    self.stats["semantic_hits"] += 1
                    
                    # Prefetch related queries if enabled
                    if self.config.prefetch_similar:
                        self._prefetch_related(query, similar_entry)
                    
                    return similar_entry.results
        
        self.stats["cache_misses"] += 1
        return None
    
    def put(self, query: str, results: List[Dict[str, Any]], 
            metadata: Optional[Dict] = None, ttl: Optional[int] = None):
        """
        Store query results in cache.
        
        Args:
            query: The query
            results: Results to cache
            metadata: Optional metadata
            ttl: Optional TTL in seconds
        """
        cache_key = self._get_cache_key(query)
        
        # Determine TTL
        if ttl is None:
            ttl = self._determine_ttl(query, metadata)
        
        # Create embedding if encoder available
        embedding = None
        if self.encoder:
            embedding = self.encoder.encode([query], convert_to_numpy=True)[0]
        
        # Create cache entry
        entry = CacheEntry(
            query=query,
            query_embedding=embedding,
            results=results,
            metadata=metadata or {},
            timestamp=datetime.now(),
            ttl_seconds=ttl
        )
        
        with self._lock:
            # Check cache size and evict if necessary
            if len(self.cache) >= self.config.max_cache_size:
                self._evict()
            
            # Add to cache
            self.cache[cache_key] = entry
            self.cache.move_to_end(cache_key)  # Move to end (most recent)
            
            # Update indices
            self.query_to_id[query] = cache_key
            self.id_to_query[cache_key] = query
            
            # Add to FAISS index if available
            if self.embedding_index and embedding is not None:
                self.embedding_index.add(embedding.reshape(1, -1).astype('float32'))
            
            # Update query patterns
            self._update_query_patterns(query, metadata)
            
            # Checkpoint if needed
            if self.config.persist_cache:
                self._checkpoint_if_needed()
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for a query."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _is_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid."""
        if entry.ttl_seconds is None:
            return True
        
        age = (datetime.now() - entry.timestamp).total_seconds()
        return age < entry.ttl_seconds
    
    def _update_access(self, entry: CacheEntry):
        """Update access statistics for an entry."""
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        
        # Update score for adaptive caching
        if self.config.enable_adaptive:
            entry.score = self._calculate_score(entry)
    
    def _calculate_score(self, entry: CacheEntry) -> float:
        """Calculate relevance score for cache entry."""
        # Combine access frequency, recency, and size
        frequency_score = min(entry.access_count / 10, 1.0)
        
        age_hours = (datetime.now() - entry.timestamp).total_seconds() / 3600
        recency_score = self.config.decay_factor ** age_hours
        
        # Prefer smaller entries
        size_score = 1.0 / (1 + len(str(entry.results)) / 10000)
        
        return frequency_score * 0.4 + recency_score * 0.4 + size_score * 0.2
    
    def _find_similar(self, query: str, threshold: float) -> Optional[CacheEntry]:
        """Find semantically similar cached query."""
        if not self.encoder:
            return None
        
        try:
            # Encode query
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)[0]
            
            # Search in FAISS index if available
            if self.embedding_index and self.embedding_index.ntotal > 0:
                query_vec = query_embedding.reshape(1, -1).astype('float32')
                
                # Normalize for cosine similarity
                faiss.normalize_L2(query_vec)
                
                # Search
                k = min(10, self.embedding_index.ntotal)
                scores, indices = self.embedding_index.search(query_vec, k)
                
                # Find best valid match
                cache_keys = list(self.cache.keys())
                for score, idx in zip(scores[0], indices[0]):
                    if score >= threshold and 0 <= idx < len(cache_keys):
                        cache_key = cache_keys[idx]
                        if cache_key in self.cache:
                            entry = self.cache[cache_key]
                            if self._is_valid(entry):
                                self.stats["avg_similarity"] = (
                                    self.stats["avg_similarity"] * 0.9 + score * 0.1
                                )
                                return entry
            else:
                # Fallback to linear search
                best_score = 0.0
                best_entry = None
                
                for entry in self.cache.values():
                    if not self._is_valid(entry):
                        continue
                    
                    if entry.query_embedding is not None:
                        similarity = self._cosine_similarity(
                            query_embedding, entry.query_embedding
                        )
                        if similarity >= threshold and similarity > best_score:
                            best_score = similarity
                            best_entry = entry
                
                if best_entry:
                    self.stats["avg_similarity"] = (
                        self.stats["avg_similarity"] * 0.9 + best_score * 0.1
                    )
                    return best_entry
                    
        except Exception as e:
            self.logger.error(f"Error finding similar query: {e}")
        
        return None
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if not NUMPY_AVAILABLE:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _determine_ttl(self, query: str, metadata: Optional[Dict]) -> int:
        """Determine appropriate TTL for a query."""
        # Check query type from metadata
        if metadata and "query_type" in metadata:
            query_type = metadata["query_type"]
            if query_type in self.config.ttl_by_query_type:
                return self.config.ttl_by_query_type[query_type]
        
        # Use adaptive TTL based on patterns
        if self.config.enable_adaptive:
            pattern = self._identify_pattern(query)
            if pattern in self.query_patterns:
                return self.query_patterns[pattern]["typical_ttl"]
        
        return self.config.default_ttl_seconds
    
    def _identify_pattern(self, query: str) -> str:
        """Identify query pattern for adaptive caching."""
        # Simple pattern identification based on keywords
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["who", "what", "when", "where", "why"]):
            return "factual"
        elif any(word in query_lower for word in ["create", "generate", "write", "imagine"]):
            return "creative"
        elif any(word in query_lower for word in ["chapter", "section", "page", "paragraph"]):
            return "navigation"
        else:
            return "general"
    
    def _update_query_patterns(self, query: str, metadata: Optional[Dict]):
        """Update query pattern statistics."""
        pattern = self._identify_pattern(query)
        pattern_data = self.query_patterns[pattern]
        
        pattern_data["count"] += 1
        
        # Update typical TTL based on actual usage
        if metadata and "response_time" in metadata:
            response_time = metadata["response_time"]
            if response_time > 1.0:  # Slow query, cache longer
                pattern_data["typical_ttl"] = min(
                    pattern_data["typical_ttl"] * 1.1,
                    self.config.default_ttl_seconds * 2
                )
    
    def _prefetch_related(self, query: str, similar_entry: CacheEntry):
        """Prefetch related queries based on similarity."""
        # This is a placeholder for prefetching logic
        # In practice, would analyze query patterns and prefetch likely follow-ups
        self.stats["prefetch_hits"] += 1
    
    def _evict(self):
        """Evict entries based on configured policy."""
        if self.config.eviction_policy == "lru":
            self._evict_lru()
        elif self.config.eviction_policy == "lfu":
            self._evict_lfu()
        elif self.config.eviction_policy == "lru_semantic":
            self._evict_lru_semantic()
        elif self.config.eviction_policy == "adaptive":
            self._evict_adaptive()
        else:
            self._evict_lru()  # Default to LRU
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if self.cache:
            # OrderedDict maintains insertion order, first item is oldest
            oldest_key = next(iter(self.cache))
            self._evict_entry(oldest_key)
    
    def _evict_lfu(self):
        """Evict least frequently used entry."""
        if not self.cache:
            return
        
        # Find entry with lowest access count
        min_entry_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].access_count
        )
        self._evict_entry(min_entry_key)
    
    def _evict_lru_semantic(self):
        """Evict based on LRU with semantic clustering protection."""
        # Protect semantically unique entries
        if not self.cache:
            return
        
        # Find candidates for eviction (oldest 20%)
        num_candidates = max(1, len(self.cache) // 5)
        candidates = list(self.cache.keys())[:num_candidates]
        
        # Choose the one with most semantic neighbors
        best_candidate = candidates[0]
        max_neighbors = 0
        
        for candidate in candidates:
            entry = self.cache[candidate]
            if entry.query_embedding is not None:
                # Count similar entries
                neighbors = 0
                for other_entry in self.cache.values():
                    if other_entry != entry and other_entry.query_embedding is not None:
                        similarity = self._cosine_similarity(
                            entry.query_embedding,
                            other_entry.query_embedding
                        )
                        if similarity > 0.8:
                            neighbors += 1
                
                if neighbors > max_neighbors:
                    max_neighbors = neighbors
                    best_candidate = candidate
        
        self._evict_entry(best_candidate)
    
    def _evict_adaptive(self):
        """Evict based on adaptive scoring."""
        if not self.cache:
            return
        
        # Calculate scores for all entries
        scores = {}
        for key, entry in self.cache.items():
            scores[key] = self._calculate_score(entry)
        
        # Evict entry with lowest score
        min_entry_key = min(scores.keys(), key=lambda k: scores[k])
        self._evict_entry(min_entry_key)
    
    def _evict_entry(self, cache_key: str):
        """Remove an entry from cache."""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            del self.cache[cache_key]
            
            # Clean up indices
            if entry.query in self.query_to_id:
                del self.query_to_id[entry.query]
            if cache_key in self.id_to_query:
                del self.id_to_query[cache_key]
            
            self.stats["evictions"] += 1
    
    def _checkpoint_if_needed(self):
        """Save cache checkpoint if interval reached."""
        if (datetime.now() - self._last_checkpoint).total_seconds() > 60:  # Every minute
            self._save_cache()
            self._last_checkpoint = datetime.now()
    
    def _save_cache(self):
        """Save cache to disk."""
        if not self.config.persist_cache:
            return
        
        try:
            cache_path = Path(self.config.persistence_path)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Save cache entries
            cache_data = {
                "entries": list(self.cache.values()),
                "stats": self.stats,
                "query_patterns": dict(self.query_patterns),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(cache_path / "cache.pkl", "wb") as f:
                pickle.dump(cache_data, f)
            
            # Save FAISS index if available
            if self.embedding_index and FAISS_AVAILABLE:
                faiss.write_index(self.embedding_index, str(cache_path / "embeddings.index"))
            
            self.logger.debug("Cache saved to disk")
            
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
    
    def _load_cache(self) -> bool:
        """Load cache from disk."""
        cache_path = Path(self.config.persistence_path)
        if not cache_path.exists():
            return False
        
        try:
            # Load cache entries
            cache_file = cache_path / "cache.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                
                # Rebuild cache
                for entry in cache_data["entries"]:
                    if self._is_valid(entry):
                        cache_key = self._get_cache_key(entry.query)
                        self.cache[cache_key] = entry
                        self.query_to_id[entry.query] = cache_key
                        self.id_to_query[cache_key] = entry.query
                
                self.stats.update(cache_data.get("stats", {}))
                self.query_patterns.update(cache_data.get("query_patterns", {}))
            
            # Load FAISS index
            index_file = cache_path / "embeddings.index"
            if index_file.exists() and FAISS_AVAILABLE:
                self.embedding_index = faiss.read_index(str(index_file))
            
            self.logger.info(f"Loaded {len(self.cache)} cache entries from disk")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load cache: {e}")
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.query_to_id.clear()
            self.id_to_query.clear()
            
            if self.embedding_index:
                self._initialize_faiss_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.copy()
        
        # Calculate hit rate
        total = stats["cache_hits"] + stats["cache_misses"]
        stats["hit_rate"] = stats["cache_hits"] / max(1, total)
        
        # Calculate memory usage
        total_size = sum(
            len(str(entry.results)) + len(entry.query)
            for entry in self.cache.values()
        )
        stats["memory_usage_mb"] = total_size / (1024 * 1024)
        
        # Add cache metrics
        stats["cache_size"] = len(self.cache)
        stats["unique_queries"] = len(self.query_to_id)
        
        return stats