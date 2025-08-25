"""
Enhanced RAG System Integration.

This module integrates all advanced RAG components into a unified system
with hybrid search, knowledge graphs, incremental indexing, semantic caching,
and quality metrics.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from datetime import datetime

# Import existing RAG components
from token_optimizer_rag import RAGConfig, RAGMode, HybridContextManager
from rag_retriever import SemanticContextRetriever

# Import new enhanced components
from rag_hybrid_search import HybridSearchEngine, HybridSearchConfig
from rag_knowledge_graph import KnowledgeGraphBuilder, KnowledgeGraphConfig, EntityType
from rag_incremental_indexing import IncrementalIndexer, IncrementalIndexConfig
from rag_semantic_cache import SemanticCache, SemanticCacheConfig
from rag_metrics import RAGMetricsCollector, MetricsConfig, QueryFeedback

# Optional imports
try:
    from providers.base import LLMProvider
    PROVIDERS_AVAILABLE = True
except ImportError:
    PROVIDERS_AVAILABLE = False

try:
    from cache_manager import CacheManager
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False


@dataclass
class EnhancedRAGConfig:
    """Configuration for the enhanced RAG system."""
    # Base RAG settings
    base_config: RAGConfig = field(default_factory=RAGConfig)
    
    # Component configurations
    hybrid_search_config: HybridSearchConfig = field(default_factory=HybridSearchConfig)
    knowledge_graph_config: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    incremental_index_config: IncrementalIndexConfig = field(default_factory=IncrementalIndexConfig)
    semantic_cache_config: SemanticCacheConfig = field(default_factory=SemanticCacheConfig)
    metrics_config: MetricsConfig = field(default_factory=MetricsConfig)
    
    # System settings
    enable_hybrid_search: bool = True
    enable_knowledge_graph: bool = True
    enable_incremental_indexing: bool = True
    enable_semantic_cache: bool = True
    enable_metrics: bool = True
    
    # Integration settings
    auto_index_chapters: bool = True
    update_graph_on_generation: bool = True
    cache_rag_results: bool = True
    
    # Storage
    storage_path: str = ".rag/enhanced"


class EnhancedRAGSystem:
    """
    Unified enhanced RAG system with all advanced features.
    
    This class orchestrates:
    - Hybrid dense/sparse search
    - Knowledge graph for entity relationships
    - Incremental real-time indexing
    - Semantic query caching
    - Quality metrics and feedback loops
    """
    
    def __init__(self,
                 config: Optional[EnhancedRAGConfig] = None,
                 provider: Optional['LLMProvider'] = None,
                 cache_manager: Optional['CacheManager'] = None):
        """
        Initialize the enhanced RAG system.
        
        Args:
            config: Enhanced RAG configuration
            provider: LLM provider for generation
            cache_manager: Cache manager for caching
        """
        self.config = config or EnhancedRAGConfig()
        self.provider = provider
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize base components
        self.base_manager = HybridContextManager(
            config=self.config.base_config,
            provider=provider,
            cache_manager=cache_manager
        )
        
        # Initialize enhanced components
        self.hybrid_search = None
        self.knowledge_graph = None
        self.incremental_indexer = None
        self.semantic_cache = None
        self.metrics_collector = None
        
        self._initialize_components()
        
        # Book context
        self.current_book = None
        self.indexed_chapters = set()
        
        # Statistics
        self.stats = {
            "queries_processed": 0,
            "cache_hits": 0,
            "graph_queries": 0,
            "index_updates": 0,
            "avg_latency_ms": 0.0
        }
    
    def _initialize_components(self):
        """Initialize all enhanced RAG components."""
        # Hybrid search engine
        if self.config.enable_hybrid_search:
            try:
                self.hybrid_search = HybridSearchEngine(self.config.hybrid_search_config)
                self.logger.info("Initialized hybrid search engine")
            except Exception as e:
                self.logger.warning(f"Failed to initialize hybrid search: {e}")
        
        # Knowledge graph
        if self.config.enable_knowledge_graph:
            try:
                self.knowledge_graph = KnowledgeGraphBuilder(self.config.knowledge_graph_config)
                self.logger.info("Initialized knowledge graph")
            except Exception as e:
                self.logger.warning(f"Failed to initialize knowledge graph: {e}")
        
        # Incremental indexer
        if self.config.enable_incremental_indexing and self.hybrid_search:
            try:
                self.incremental_indexer = IncrementalIndexer(
                    base_index=self.hybrid_search.dense_index,
                    config=self.config.incremental_index_config
                )
                self.logger.info("Initialized incremental indexer")
            except Exception as e:
                self.logger.warning(f"Failed to initialize incremental indexer: {e}")
        
        # Semantic cache
        if self.config.enable_semantic_cache:
            try:
                self.semantic_cache = SemanticCache(self.config.semantic_cache_config)
                self.logger.info("Initialized semantic cache")
            except Exception as e:
                self.logger.warning(f"Failed to initialize semantic cache: {e}")
        
        # Metrics collector
        if self.config.enable_metrics:
            try:
                self.metrics_collector = RAGMetricsCollector(self.config.metrics_config)
                self.logger.info("Initialized metrics collector")
            except Exception as e:
                self.logger.warning(f"Failed to initialize metrics collector: {e}")
    
    def index_book(self, book_data: Dict[str, Any], force_reindex: bool = False):
        """
        Index a complete book for RAG retrieval.
        
        Args:
            book_data: Book data including chapters and metadata
            force_reindex: Whether to force complete reindexing
        """
        self.current_book = book_data
        book_title = book_data.get("title", "Unknown")
        
        self.logger.info(f"Indexing book: {book_title}")
        
        # Extract chapters
        chapters = book_data.get("chapters", [])
        if not chapters:
            self.logger.warning("No chapters found in book data")
            return
        
        # Prepare documents for indexing
        documents = []
        metadata_list = []
        
        for i, chapter in enumerate(chapters):
            chapter_text = self._extract_chapter_text(chapter)
            documents.append(chapter_text)
            metadata_list.append({
                "chapter_index": i,
                "chapter_title": chapter.get("title", f"Chapter {i+1}"),
                "chapter_number": chapter.get("number", i+1)
            })
        
        # Index with hybrid search
        if self.hybrid_search:
            self.hybrid_search.index_documents(documents, metadata_list)
            self.logger.info(f"Indexed {len(documents)} chapters with hybrid search")
        
        # Build knowledge graph
        if self.knowledge_graph:
            for doc, meta in zip(documents, metadata_list):
                context = {"chapter": meta["chapter_title"]}
                entities = self.knowledge_graph.extract_entities_from_text(doc, context)
                relationships = self.knowledge_graph.extract_relationships(doc, entities)
                self.knowledge_graph.add_to_graph(entities, relationships)
            
            self.logger.info("Built knowledge graph from book content")
        
        # Index with base manager
        self.base_manager.index_book(book_data)
        
        self.indexed_chapters = set(range(len(chapters)))
    
    def index_chapter(self, chapter_data: Dict[str, Any], chapter_index: int):
        """
        Index a single chapter incrementally.
        
        Args:
            chapter_data: Chapter data
            chapter_index: Chapter index in the book
        """
        if chapter_index in self.indexed_chapters:
            return  # Already indexed
        
        chapter_text = self._extract_chapter_text(chapter_data)
        metadata = {
            "chapter_index": chapter_index,
            "chapter_title": chapter_data.get("title", f"Chapter {chapter_index + 1}"),
            "chapter_number": chapter_data.get("number", chapter_index + 1)
        }
        
        # Incremental index update
        if self.incremental_indexer:
            doc_id = f"chapter_{chapter_index}"
            self.incremental_indexer.add_document(doc_id, chapter_text, metadata)
            self.stats["index_updates"] += 1
        
        # Update knowledge graph
        if self.knowledge_graph and self.config.update_graph_on_generation:
            context = {"chapter": metadata["chapter_title"]}
            entities = self.knowledge_graph.extract_entities_from_text(chapter_text, context)
            relationships = self.knowledge_graph.extract_relationships(chapter_text, entities)
            self.knowledge_graph.add_to_graph(entities, relationships)
        
        self.indexed_chapters.add(chapter_index)
        self.logger.debug(f"Indexed chapter {chapter_index}")
    
    def retrieve_context(self, query: str, max_tokens: int = 4000,
                        use_cache: bool = True) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query using all enhanced features.
        
        Args:
            query: Query text
            max_tokens: Maximum tokens for context
            use_cache: Whether to use semantic cache
        
        Returns:
            Dictionary containing retrieved context and metadata
        """
        start_time = time.time()
        query_id = f"q_{int(time.time() * 1000)}"
        
        # Check semantic cache first
        if use_cache and self.semantic_cache:
            cached_results = self.semantic_cache.get(query)
            if cached_results:
                self.stats["cache_hits"] += 1
                self.logger.debug(f"Cache hit for query: {query[:50]}...")
                return {
                    "context": cached_results,
                    "source": "cache",
                    "latency_ms": 0
                }
        
        # Perform hybrid search
        search_results = []
        if self.hybrid_search:
            search_results = self.hybrid_search.search(query, top_k=10, hybrid=True)
            self.logger.debug(f"Hybrid search returned {len(search_results)} results")
        
        # Query knowledge graph for additional context
        graph_context = {}
        if self.knowledge_graph:
            graph_context = self.knowledge_graph.query_graph(query, max_hops=2)
            self.stats["graph_queries"] += 1
            self.logger.debug(f"Knowledge graph returned {len(graph_context.get('entities', {}))} entities")
        
        # Combine results
        combined_context = self._combine_contexts(
            search_results,
            graph_context,
            max_tokens
        )
        
        # Cache results if enabled
        if use_cache and self.semantic_cache and self.config.cache_rag_results:
            self.semantic_cache.put(query, combined_context, ttl=3600)
        
        # Record metrics
        latency_ms = (time.time() - start_time) * 1000
        self.stats["queries_processed"] += 1
        self.stats["avg_latency_ms"] = (
            (self.stats["avg_latency_ms"] * (self.stats["queries_processed"] - 1) + latency_ms) /
            self.stats["queries_processed"]
        )
        
        if self.metrics_collector:
            self.metrics_collector.record_query(
                query_id,
                query,
                search_results,
                latency_ms
            )
        
        return {
            "context": combined_context,
            "search_results": search_results,
            "graph_context": graph_context,
            "source": "enhanced_rag",
            "latency_ms": latency_ms,
            "query_id": query_id
        }
    
    def provide_feedback(self, query_id: str, relevance_scores: List[float],
                        selected_docs: List[int], quality_score: float):
        """
        Provide feedback for a query to improve the system.
        
        Args:
            query_id: Query identifier
            relevance_scores: Relevance scores for retrieved documents
            selected_docs: Indices of actually useful documents
            quality_score: Overall quality score (0-1)
        """
        if self.metrics_collector:
            self.metrics_collector.record_feedback(
                query_id,
                selected_docs,
                relevance_scores,
                quality_score
            )
            self.logger.debug(f"Recorded feedback for query {query_id}")
    
    def _extract_chapter_text(self, chapter: Dict[str, Any]) -> str:
        """Extract text content from a chapter."""
        sections = chapter.get("sections", [])
        if sections:
            return "\n\n".join([
                f"{s.get('title', '')}\n{s.get('content', '')}"
                for s in sections
            ])
        return chapter.get("content", "")
    
    def _combine_contexts(self, search_results: List[Dict],
                         graph_context: Dict,
                         max_tokens: int) -> str:
        """Combine search results and graph context into a unified context."""
        context_parts = []
        current_tokens = 0
        
        # Add search results
        for result in search_results:
            text = result.get("text", "")
            # Estimate tokens (rough approximation)
            tokens = len(text.split()) * 1.3
            if current_tokens + tokens <= max_tokens * 0.7:  # Reserve 30% for graph context
                context_parts.append(text)
                current_tokens += tokens
            else:
                break
        
        # Add graph context
        if graph_context and "entities" in graph_context:
            graph_summary = self._summarize_graph_context(graph_context)
            tokens = len(graph_summary.split()) * 1.3
            if current_tokens + tokens <= max_tokens:
                context_parts.append(f"\n[Entity Relationships]\n{graph_summary}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _summarize_graph_context(self, graph_context: Dict) -> str:
        """Summarize knowledge graph context into text."""
        summary_parts = []
        
        # Summarize key entities
        entities = graph_context.get("entities", {})
        if entities:
            entity_list = []
            for entity_id, entity in list(entities.items())[:5]:  # Top 5 entities
                entity_list.append(f"- {entity.name} ({entity.type.value})")
            if entity_list:
                summary_parts.append("Key Entities:\n" + "\n".join(entity_list))
        
        # Summarize relationships
        relationships = graph_context.get("relationships", [])
        if relationships:
            rel_list = []
            for rel in relationships[:5]:  # Top 5 relationships
                source = entities.get(rel["source"], {})
                target = entities.get(rel["target"], {})
                if source and target:
                    rel_list.append(
                        f"- {source.name} {rel['type']} {target.name}"
                    )
            if rel_list:
                summary_parts.append("Relationships:\n" + "\n".join(rel_list))
        
        return "\n\n".join(summary_parts)
    
    def optimize(self):
        """Optimize all system components."""
        # Optimize indices
        if self.incremental_indexer:
            self.incremental_indexer.optimize()
        
        # Save knowledge graph
        if self.knowledge_graph:
            self.knowledge_graph.save()
        
        # Save metrics
        if self.metrics_collector:
            self.metrics_collector.save_metrics()
        
        self.logger.info("System optimization completed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = self.stats.copy()
        
        # Add component stats
        if self.hybrid_search:
            stats["hybrid_search"] = self.hybrid_search.get_stats()
        
        if self.knowledge_graph:
            stats["knowledge_graph"] = self.knowledge_graph.get_stats()
        
        if self.incremental_indexer:
            stats["incremental_indexer"] = self.incremental_indexer.get_stats()
        
        if self.semantic_cache:
            stats["semantic_cache"] = self.semantic_cache.get_stats()
        
        if self.metrics_collector:
            stats["metrics"] = self.metrics_collector.get_metrics_summary()
        
        return stats
    
    def save_state(self, path: Optional[str] = None):
        """Save the complete system state."""
        save_path = Path(path or self.config.storage_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save configurations
        config_data = {
            "enhanced_config": self.config.__dict__,
            "stats": self.stats,
            "indexed_chapters": list(self.indexed_chapters)
        }
        
        with open(save_path / "config.json", "w") as f:
            json.dump(config_data, f, indent=2, default=str)
        
        # Save component states
        if self.hybrid_search:
            self.hybrid_search.save_index(str(save_path / "hybrid_search"))
        
        if self.knowledge_graph:
            self.knowledge_graph.save(str(save_path / "knowledge_graph"))
        
        if self.incremental_indexer:
            self.incremental_indexer._save_checkpoint()
        
        if self.semantic_cache:
            self.semantic_cache._save_cache()
        
        if self.metrics_collector:
            self.metrics_collector.save_metrics()
        
        self.logger.info(f"System state saved to {save_path}")
    
    def load_state(self, path: Optional[str] = None) -> bool:
        """Load system state from disk."""
        load_path = Path(path or self.config.storage_path)
        if not load_path.exists():
            return False
        
        try:
            # Load configuration
            config_file = load_path / "config.json"
            if config_file.exists():
                with open(config_file, "r") as f:
                    config_data = json.load(f)
                    self.stats = config_data.get("stats", {})
                    self.indexed_chapters = set(config_data.get("indexed_chapters", []))
            
            # Load component states
            if self.hybrid_search:
                self.hybrid_search.load_index(str(load_path / "hybrid_search"))
            
            if self.knowledge_graph:
                self.knowledge_graph.load(str(load_path / "knowledge_graph"))
            
            if self.incremental_indexer:
                self.incremental_indexer._load_checkpoint()
            
            if self.semantic_cache:
                self.semantic_cache._load_cache()
            
            self.logger.info(f"System state loaded from {load_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load system state: {e}")
            return False


def create_enhanced_rag_system(
    provider: Optional['LLMProvider'] = None,
    cache_manager: Optional['CacheManager'] = None,
    config: Optional[EnhancedRAGConfig] = None
) -> EnhancedRAGSystem:
    """
    Factory function to create an enhanced RAG system.
    
    Args:
        provider: LLM provider for generation
        cache_manager: Cache manager for caching
        config: Enhanced RAG configuration
    
    Returns:
        Configured EnhancedRAGSystem instance
    """
    if config is None:
        config = EnhancedRAGConfig()
        
        # Auto-detect and configure based on available components
        try:
            from sentence_transformers import SentenceTransformer
            config.enable_hybrid_search = True
            config.enable_semantic_cache = True
        except ImportError:
            config.enable_hybrid_search = False
            config.enable_semantic_cache = False
        
        try:
            import spacy
            config.enable_knowledge_graph = True
        except ImportError:
            config.enable_knowledge_graph = False
    
    return EnhancedRAGSystem(config, provider, cache_manager)