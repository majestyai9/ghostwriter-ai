"""
Hybrid RAG-enhanced token optimization system for Ghostwriter AI.

This module implements a sophisticated context management system that combines:
1. Smart LLM-based summarization for chapter content
2. Semantic search using FAISS and sentence-transformers for relevant content retrieval
3. Intelligent token allocation between core context, RAG-retrieved content, and summaries

Features:
- IVF indexing for large-scale vector search
- GPU acceleration support
- Batch processing capabilities  
- Vector caching for frequent queries
- Performance metrics and logging
- Thread-safe implementation with proper resource management

Optimizations:
- IVF (Inverted File Index) for efficient large-scale search
- GPU acceleration for embeddings and FAISS operations
- Optimized chunking strategies for better retrieval accuracy
- Batch processing for multiple queries
- LRU caching with TTL for frequently accessed vectors
- Comprehensive performance metrics and logging

Usage:
    from token_optimizer_rag import create_hybrid_manager, RAGConfig, RAGMode
    
    # Create with default configuration
    manager = create_hybrid_manager(provider=llm_provider)
    
    # Or with custom configuration
    config = RAGConfig(
        mode=RAGMode.HYBRID,
        use_gpu=True,
        use_ivf=True,
        enable_batch_processing=True
    )
    manager = create_hybrid_manager(provider=llm_provider, config=config)
    
    # Prepare context for book generation
    context = manager.prepare_context(
        book=book_data,
        current_chapter=5,
        book_dir="/path/to/book",
        query="relevant search terms"
    )
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

# Check for optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. RAG features will be disabled.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Vector search will be disabled.")

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        logging.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
except ImportError:
    CUDA_AVAILABLE = False
    logging.info("PyTorch not available, GPU acceleration disabled")

# Import optimized modules
from rag_models import ChunkMetadata
from rag_retriever import SemanticContextRetriever
from rag_hybrid_manager import HybridContextManager as HybridManagerBase

from cache_manager import CacheManager
from providers.base import LLMProvider


class RAGMode(Enum):
    """RAG operation modes."""
    
    BASIC = "basic"  # Basic RAG with semantic search only
    HYBRID = "hybrid"  # Full hybrid mode with summarization + RAG
    FULL = "full"  # Maximum RAG features enabled


@dataclass
class RAGConfig:
    """
    Configuration for RAG system.
    
    Attributes:
        mode: Operation mode (DISABLED, BASIC, HYBRID, FULL)
        embedding_model: Sentence transformer model to use
        vector_store_dir: Directory for storing vector indices
        chunk_size: Size of text chunks for indexing
        chunk_overlap: Overlap between consecutive chunks
        top_k: Number of similar chunks to retrieve
        similarity_threshold: Minimum similarity score for retrieval
        use_ivf: Enable IVF indexing for large documents
        ivf_nlist: Number of clusters for IVF
        ivf_nprobe: Number of clusters to search
        use_gpu: Try to use GPU if available
        gpu_batch_size: Batch size for GPU operations
        cache_size: Number of cached query results
        cache_ttl: Cache time-to-live in seconds
        core_context_ratio: Proportion of tokens for core context
        rag_context_ratio: Proportion of tokens for RAG content
        summary_context_ratio: Proportion of tokens for summaries
        enable_caching: Enable result caching
        enable_compression: Enable content compression
        enable_async_indexing: Enable async indexing (future)
        enable_batch_processing: Enable batch query processing
    """
    
    mode: RAGMode = RAGMode.HYBRID
    embedding_model: str = "all-MiniLM-L6-v2"  # Good balance of quality and size
    vector_store_dir: str = ".rag"
    chunk_size: int = 512  # Characters per chunk for indexing
    chunk_overlap: int = 128  # Overlap between chunks
    top_k: int = 10  # Number of similar chunks to retrieve
    similarity_threshold: float = 0.5  # Minimum similarity score
    
    # IVF indexing parameters
    use_ivf: bool = True  # Use IVF indexing for large documents
    ivf_nlist: int = 100  # Number of clusters for IVF
    ivf_nprobe: int = 10  # Number of clusters to search
    
    # GPU configuration
    use_gpu: bool = True  # Try to use GPU if available
    gpu_batch_size: int = 64  # Batch size for GPU operations
    
    # Caching configuration
    cache_size: int = 1000  # Number of cached query results
    cache_ttl: int = 3600  # Cache TTL in seconds

    # Token distribution (must sum to 1.0)
    core_context_ratio: float = 0.4  # 40% for title + recent chapters
    rag_context_ratio: float = 0.4  # 40% for RAG-retrieved content
    summary_context_ratio: float = 0.2  # 20% for summaries

    # Feature flags
    enable_caching: bool = True
    enable_compression: bool = True
    enable_async_indexing: bool = False
    enable_batch_processing: bool = True

    def __post_init__(self):
        """Validate configuration."""
        total_ratio = (
            self.core_context_ratio + 
            self.rag_context_ratio + 
            self.summary_context_ratio
        )
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"Token distribution ratios must sum to 1.0, got {total_ratio}")


class HybridContextManager(HybridManagerBase):
    """
    Enhanced hybrid context manager with full RAG integration.
    
    This class extends the base manager with RAG-specific initialization logic,
    creating and configuring the semantic retriever based on the provided configuration.
    
    Features:
    - Automatic retriever initialization for HYBRID and FULL modes
    - Graceful degradation when dependencies are unavailable
    - Full compatibility with base manager interface
    """
    
    def __init__(self,
                 config: Optional[RAGConfig] = None,
                 provider: Optional[LLMProvider] = None,
                 cache_manager: Optional[CacheManager] = None,
                 max_tokens: int = 128000):
        """
        Initialize the hybrid context manager with RAG components.
        
        Args:
            config: RAG configuration (uses defaults if None)
            provider: LLM provider for summarization
            cache_manager: Cache manager for storing results
            max_tokens: Maximum context window size
        """
        self.config = config or RAGConfig()
        
        # Initialize retriever if in appropriate mode and dependencies available
        retriever = None
        if (self.config.mode in [RAGMode.HYBRID, RAGMode.FULL] and 
            SENTENCE_TRANSFORMERS_AVAILABLE and FAISS_AVAILABLE):
            try:
                retriever = SemanticContextRetriever(
                    config=self.config,
                    cache_manager=cache_manager
                )
                logging.info("Semantic retriever initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize retriever: {e}")
                # Downgrade to BASIC mode if retriever fails
                self.config.mode = RAGMode.BASIC
        
        # Call parent constructor
        super().__init__(
            config=self.config,
            provider=provider,
            cache_manager=cache_manager,
            retriever=retriever,
            max_tokens=max_tokens
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics.
        
        Returns:
            Dictionary containing performance metrics from all components
        """
        stats = self.get_stats()  # Get base stats
        
        # Add GPU and optimization status
        stats["optimization"] = {
            "gpu_available": CUDA_AVAILABLE,
            "gpu_enabled": self.config.use_gpu and CUDA_AVAILABLE,
            "ivf_enabled": self.config.use_ivf,
            "batch_processing": self.config.enable_batch_processing,
            "caching_enabled": self.config.enable_caching,
            "cache_size": self.config.cache_size,
            "cache_ttl": f"{self.config.cache_ttl}s"
        }
        
        return stats


def create_hybrid_manager(provider: Optional[LLMProvider] = None,
                         cache_manager: Optional[CacheManager] = None,
                         config: Optional[RAGConfig] = None,
                         max_tokens: int = 128000,
                         use_enhanced: bool = True) -> HybridContextManager:
    """
    Create a hybrid context manager with sensible defaults.
    
    This factory function automatically configures the manager based on
    available dependencies and system capabilities. It can create either
    the basic hybrid manager or the enhanced RAG system with all advanced features.
    
    Args:
        provider: LLM provider for summarization
        cache_manager: Cache manager for storing results
        config: Optional RAG configuration (auto-configured if None)
        max_tokens: Maximum context window size
        use_enhanced: Whether to use the enhanced RAG system with all features
        
    Returns:
        Configured HybridContextManager instance (or EnhancedRAGSystem wrapper)
        
    Example:
        >>> from providers.anthropic import AnthropicProvider
        >>> provider = AnthropicProvider(api_key="...")
        >>> # Use enhanced system with all features
        >>> manager = create_hybrid_manager(provider=provider, use_enhanced=True)
        >>> # Or use basic hybrid manager
        >>> manager = create_hybrid_manager(provider=provider, use_enhanced=False)
        >>> context = manager.prepare_context(book_data, current_chapter=5)
    """
    # Try to use enhanced system if requested
    if use_enhanced:
        try:
            from rag_enhanced_system import create_enhanced_rag_system, EnhancedRAGConfig
            
            # Create enhanced config based on provided RAG config
            enhanced_config = EnhancedRAGConfig()
            if config:
                enhanced_config.base_config = config
            
            # Auto-detect capabilities
            enhanced_config.enable_hybrid_search = SENTENCE_TRANSFORMERS_AVAILABLE and FAISS_AVAILABLE
            
            try:
                import spacy
                enhanced_config.enable_knowledge_graph = True
            except ImportError:
                enhanced_config.enable_knowledge_graph = False
            
            enhanced_config.enable_incremental_indexing = FAISS_AVAILABLE
            enhanced_config.enable_semantic_cache = SENTENCE_TRANSFORMERS_AVAILABLE
            enhanced_config.enable_metrics = True
            
            # Create enhanced system
            enhanced_system = create_enhanced_rag_system(
                provider=provider,
                cache_manager=cache_manager,
                config=enhanced_config
            )
            
            logging.info(
                f"Created Enhanced RAG System with features: "
                f"Hybrid Search={enhanced_config.enable_hybrid_search}, "
                f"Knowledge Graph={enhanced_config.enable_knowledge_graph}, "
                f"Incremental Indexing={enhanced_config.enable_incremental_indexing}, "
                f"Semantic Cache={enhanced_config.enable_semantic_cache}, "
                f"Metrics={enhanced_config.enable_metrics}"
            )
            
            # Return wrapped enhanced system that's compatible with HybridContextManager interface
            return _EnhancedRAGWrapper(enhanced_system)
            
        except ImportError as e:
            logging.warning(f"Enhanced RAG system not available: {e}, falling back to basic")
    
    # Fallback to basic hybrid manager
    if config is None:
        # Auto-configure based on available dependencies
        if SENTENCE_TRANSFORMERS_AVAILABLE and FAISS_AVAILABLE:
            config = RAGConfig(
                mode=RAGMode.HYBRID,
                use_gpu=CUDA_AVAILABLE,
                use_ivf=True,
                enable_batch_processing=True
            )
            logging.info(
                f"Auto-configured RAG mode: HYBRID "
                f"(GPU: {CUDA_AVAILABLE}, IVF: True, Batch: True)"
            )
        else:
            # Fallback to basic mode without vector search
            config = RAGConfig(mode=RAGMode.BASIC)
            missing = []
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                missing.append("sentence-transformers")
            if not FAISS_AVAILABLE:
                missing.append("faiss")
            logging.warning(
                f"RAG dependencies not available ({', '.join(missing)}), "
                f"using BASIC mode"
            )

    return HybridContextManager(
        config=config,
        provider=provider,
        cache_manager=cache_manager,
        max_tokens=max_tokens
    )

class _EnhancedRAGWrapper:
    """
    Wrapper to make EnhancedRAGSystem compatible with HybridContextManager interface.
    
    This allows seamless integration of the enhanced system with existing code
    that expects a HybridContextManager.
    """
    
    def __init__(self, enhanced_system):
        """Initialize wrapper with enhanced RAG system."""
        self.enhanced_system = enhanced_system
        self.config = enhanced_system.config.base_config
        self.provider = enhanced_system.provider
        self.cache_manager = enhanced_system.cache_manager
        
    def index_book(self, book_data: Dict[str, Any]):
        """Index book using enhanced system."""
        return self.enhanced_system.index_book(book_data)
    
    def prepare_context(self, book_data: Dict[str, Any], 
                       current_chapter: int = 0,
                       instructions: str = "",
                       style: str = "",
                       **kwargs) -> str:
        """
        Prepare context using enhanced RAG system.
        
        This method translates the HybridContextManager interface to
        the enhanced system's retrieve_context method.
        """
        # Build query from instructions and current context
        query_parts = []
        
        if instructions:
            query_parts.append(f"Instructions: {instructions}")
        
        if style:
            query_parts.append(f"Style: {style}")
        
        if current_chapter > 0 and book_data.get("chapters"):
            chapters = book_data["chapters"]
            if current_chapter < len(chapters):
                chapter = chapters[current_chapter]
                query_parts.append(f"Current chapter: {chapter.get('title', f'Chapter {current_chapter + 1}')}")
        
        query = "\n".join(query_parts) if query_parts else "Retrieve relevant context for book generation"
        
        # Index book if needed
        if not self.enhanced_system.indexed_chapters:
            self.enhanced_system.index_book(book_data)
        
        # Retrieve context
        result = self.enhanced_system.retrieve_context(
            query=query,
            max_tokens=kwargs.get("max_tokens", 4000)
        )
        
        # Return context string
        return result.get("context", "")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from enhanced system."""
        return self.enhanced_system.get_stats()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.enhanced_system.get_stats()
        # Format to match expected interface
        return {
            "queries_processed": stats.get("queries_processed", 0),
            "cache_hits": stats.get("cache_hits", 0),
            "avg_latency_ms": stats.get("avg_latency_ms", 0.0),
            "optimization": {
                "gpu_available": CUDA_AVAILABLE,
                "gpu_enabled": self.config.use_gpu and CUDA_AVAILABLE,
                "enhanced_features": {
                    "hybrid_search": self.enhanced_system.hybrid_search is not None,
                    "knowledge_graph": self.enhanced_system.knowledge_graph is not None,
                    "incremental_indexing": self.enhanced_system.incremental_indexer is not None,
                    "semantic_cache": self.enhanced_system.semantic_cache is not None,
                    "metrics": self.enhanced_system.metrics_collector is not None
                }
            }
        }
    
    def save_state(self, path: Optional[str] = None):
        """Save enhanced system state."""
        self.enhanced_system.save_state(path)
    
    def load_state(self, path: Optional[str] = None) -> bool:
        """Load enhanced system state."""
        return self.enhanced_system.load_state(path)
    
    def optimize(self):
        """Optimize the enhanced system."""
        self.enhanced_system.optimize()
    
    # Delegate other methods to base manager for compatibility
    def __getattr__(self, name):
        """Delegate unknown attributes to base manager."""
        if hasattr(self.enhanced_system.base_manager, name):
            return getattr(self.enhanced_system.base_manager, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


# Export main components
__all__ = [
    "create_hybrid_manager",
    "HybridContextManager",
    "RAGConfig",
    "RAGMode",
    "ChunkMetadata",
    "SENTENCE_TRANSFORMERS_AVAILABLE",
    "FAISS_AVAILABLE",
    "CUDA_AVAILABLE"
]