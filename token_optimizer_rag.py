"""
Hybrid RAG-enhanced token optimization system for Ghostwriter AI.

This module implements a sophisticated context management system that combines:
1. Smart LLM-based summarization for chapter content
2. Semantic search using FAISS and sentence-transformers for relevant content retrieval
3. Intelligent token allocation between core context, RAG-retrieved content, and summaries

Features:
- Automatic chapter summarization with caching
- Vector-based semantic similarity search
- Hybrid context optimization with configurable token distribution
- Backward compatibility with existing books
- Thread-safe implementation with proper resource management
"""

import json
import logging
import os
import pickle
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Handle optional dependencies gracefully
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

from cache_manager import CacheManager
from providers.base import LLMProvider
from token_optimizer import BookContextManager, ContextElement, ContextPriority, SlidingWindowManager


class RAGMode(Enum):
    """RAG operation modes"""
    DISABLED = "disabled"  # RAG completely disabled (backward compatibility)
    BASIC = "basic"  # Basic RAG with semantic search only
    HYBRID = "hybrid"  # Full hybrid mode with summarization + RAG
    FULL = "full"  # Maximum RAG features enabled


@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    mode: RAGMode = RAGMode.HYBRID
    embedding_model: str = "all-MiniLM-L6-v2"  # Good balance of quality and size
    vector_store_dir: str = ".rag"
    chunk_size: int = 512  # Characters per chunk for indexing
    chunk_overlap: int = 128  # Overlap between chunks
    top_k: int = 10  # Number of similar chunks to retrieve
    similarity_threshold: float = 0.5  # Minimum similarity score
    
    # Token distribution (must sum to 1.0)
    core_context_ratio: float = 0.4  # 40% for title + recent chapters
    rag_context_ratio: float = 0.4  # 40% for RAG-retrieved content
    summary_context_ratio: float = 0.2  # 20% for summaries
    
    # Feature flags
    enable_caching: bool = True
    enable_compression: bool = True
    enable_async_indexing: bool = False
    
    def __post_init__(self):
        """Validate configuration"""
        total_ratio = self.core_context_ratio + self.rag_context_ratio + self.summary_context_ratio
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"Token distribution ratios must sum to 1.0, got {total_ratio}")


@dataclass
class ChunkMetadata:
    """Metadata for indexed text chunks"""
    chapter_number: int
    section_number: Optional[int] = None
    chunk_index: int = 0
    chapter_title: str = ""
    section_title: str = ""
    timestamp: float = field(default_factory=time.time)
    tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "chapter_number": self.chapter_number,
            "section_number": self.section_number,
            "chunk_index": self.chunk_index,
            "chapter_title": self.chapter_title,
            "section_title": self.section_title,
            "timestamp": self.timestamp,
            "tokens": self.tokens
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMetadata":
        """Create from dictionary"""
        return cls(**data)


class SmartSummarizer:
    """
    Intelligent summarization system using LLM for structured content summarization.
    
    Features:
    - Generates structured summaries capturing key narrative elements
    - Caches summaries for efficiency
    - Handles different content types (chapters, sections)
    - Thread-safe operation
    """
    
    def __init__(self, 
                 provider: Optional[LLMProvider] = None,
                 cache_manager: Optional[CacheManager] = None,
                 max_summary_tokens: int = 200):
        """
        Initialize the smart summarizer.
        
        Args:
            provider: LLM provider for generating summaries
            cache_manager: Cache manager for storing summaries
            max_summary_tokens: Maximum tokens per summary
        """
        self.provider = provider
        self.cache_manager = cache_manager or CacheManager(backend="memory")
        self.max_summary_tokens = max_summary_tokens
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
    
    def summarize_chapter(self, 
                          chapter: Dict[str, Any],
                          book_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate or retrieve a structured summary for a chapter.
        
        Args:
            chapter: Chapter data including content and metadata
            book_context: Optional book context for better summarization
            
        Returns:
            Structured summary string
        """
        if not chapter.get("content"):
            return ""
        
        # Create cache key
        cache_key = self._create_cache_key("chapter_summary", chapter)
        
        # Check cache
        if self.cache_manager:
            cached_summary = self.cache_manager.get(cache_key)
            if cached_summary:
                self.logger.debug(f"Using cached summary for chapter {chapter.get('number')}")
                return cached_summary
        
        # Generate summary
        summary = self._generate_summary(chapter, book_context)
        
        # Cache the summary
        if self.cache_manager and summary:
            self.cache_manager.set(cache_key, summary, expire=86400)  # Cache for 24 hours
        
        return summary
    
    def _generate_summary(self, 
                         chapter: Dict[str, Any],
                         book_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a summary using LLM or fallback to extraction.
        
        Args:
            chapter: Chapter to summarize
            book_context: Optional book context
            
        Returns:
            Generated summary
        """
        if self.provider:
            try:
                # Prepare the summarization prompt
                prompt = self._create_summary_prompt(chapter, book_context)
                
                # Generate summary using LLM
                with self._lock:
                    response = self.provider.generate(
                        prompt=prompt,
                        max_tokens=self.max_summary_tokens,
                        temperature=0.3  # Lower temperature for more focused summaries
                    )
                
                if response and response.content:
                    self.logger.info(f"Generated LLM summary for chapter {chapter.get('number')}")
                    return response.content.strip()
                    
            except Exception as e:
                self.logger.warning(f"LLM summarization failed: {e}, falling back to extraction")
        
        # Fallback to simple extraction
        return self._extract_summary(chapter)
    
    def _create_summary_prompt(self, 
                               chapter: Dict[str, Any],
                               book_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a structured prompt for chapter summarization.
        
        Args:
            chapter: Chapter to summarize
            book_context: Optional book context
            
        Returns:
            Formatted prompt string
        """
        chapter_num = chapter.get("number", "?")
        chapter_title = chapter.get("title", "Untitled")
        content = chapter.get("content", "")[:3000]  # Limit content length
        
        prompt = f"""Summarize this book chapter in a structured format. Focus on key narrative elements.

Chapter {chapter_num}: {chapter_title}

Content:
{content}

Provide a concise summary (max 150 words) that includes:
1. Main plot developments
2. Character actions and developments
3. Important world-building details
4. Unresolved questions or cliffhangers

Format as a single paragraph focusing on the most important elements."""
        
        if book_context:
            book_title = book_context.get("title", "")
            if book_title:
                prompt = f"Book: {book_title}\n\n" + prompt
        
        return prompt
    
    def _extract_summary(self, chapter: Dict[str, Any]) -> str:
        """
        Extract a simple summary from chapter content.
        
        Args:
            chapter: Chapter to extract from
            
        Returns:
            Extracted summary
        """
        content = chapter.get("content", "")
        topics = chapter.get("topics", "")
        
        summary_parts = []
        
        if topics:
            summary_parts.append(f"Topics: {topics}")
        
        # Extract first paragraph or sentences
        if content:
            # Try to get first meaningful paragraph
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            if paragraphs:
                first_para = paragraphs[0][:300]
                if len(first_para) < len(paragraphs[0]):
                    first_para += "..."
                summary_parts.append(first_para)
        
        return " ".join(summary_parts) if summary_parts else "No summary available."
    
    def _create_cache_key(self, prefix: str, chapter: Dict[str, Any]) -> str:
        """
        Create a unique cache key for a chapter summary.
        
        Args:
            prefix: Cache key prefix
            chapter: Chapter data
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Use content hash for cache key to handle content changes
        content = chapter.get("content", "")
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        chapter_num = chapter.get("number", 0)
        
        return f"{prefix}:ch{chapter_num}:{content_hash}"
    
    def batch_summarize(self, 
                        chapters: List[Dict[str, Any]],
                        book_context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Summarize multiple chapters efficiently.
        
        Args:
            chapters: List of chapters to summarize
            book_context: Optional book context
            
        Returns:
            List of summaries
        """
        summaries = []
        
        for chapter in chapters:
            summary = self.summarize_chapter(chapter, book_context)
            summaries.append(summary)
        
        return summaries


class SemanticContextRetriever:
    """
    Semantic search and retrieval system using sentence-transformers and FAISS.
    
    Features:
    - Efficient vector indexing with FAISS
    - Semantic similarity search for relevant content
    - Persistent vector stores
    - Incremental indexing support
    """
    
    def __init__(self, 
                 config: RAGConfig,
                 cache_manager: Optional[CacheManager] = None):
        """
        Initialize the semantic context retriever.
        
        Args:
            config: RAG configuration
            cache_manager: Optional cache manager
        """
        self.config = config
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.encoder = None
        self.index = None
        self.metadata = []
        self.chunks = []
        
        self._lock = threading.RLock()
        self._initialized = False
        
        # Initialize if dependencies are available
        if SENTENCE_TRANSFORMERS_AVAILABLE and FAISS_AVAILABLE:
            self._initialize()
    
    def _initialize(self):
        """Initialize the retriever components"""
        with self._lock:
            if self._initialized:
                return
            
            try:
                # Initialize sentence transformer
                self.logger.info(f"Loading embedding model: {self.config.embedding_model}")
                self.encoder = SentenceTransformer(self.config.embedding_model)
                
                # Get embedding dimension
                dummy_embedding = self.encoder.encode(["test"], show_progress_bar=False)
                self.embedding_dim = dummy_embedding.shape[1]
                
                # Initialize FAISS index
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
                
                self._initialized = True
                self.logger.info("Semantic retriever initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize semantic retriever: {e}")
                self._initialized = False
    
    def index_book(self, book: Dict[str, Any], book_dir: str) -> bool:
        """
        Index or update the vector store for a book.
        
        Args:
            book: Book data to index
            book_dir: Directory to store vector index
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            return False
        
        try:
            vector_store_path = Path(book_dir) / self.config.vector_store_dir
            vector_store_path.mkdir(exist_ok=True)
            
            # Check if index already exists and is up-to-date
            index_file = vector_store_path / "index.faiss"
            metadata_file = vector_store_path / "metadata.pkl"
            
            if self._is_index_current(book, vector_store_path):
                self.logger.info("Vector index is up-to-date, loading existing index")
                return self._load_index(vector_store_path)
            
            # Create new index
            self.logger.info("Creating new vector index for book")
            chunks_with_metadata = self._create_chunks(book)
            
            if not chunks_with_metadata:
                self.logger.warning("No content to index")
                return False
            
            # Encode chunks
            texts = [chunk for chunk, _ in chunks_with_metadata]
            metadata = [meta for _, meta in chunks_with_metadata]
            
            self.logger.info(f"Encoding {len(texts)} chunks...")
            with self._lock:
                embeddings = self.encoder.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    normalize_embeddings=True  # For cosine similarity
                )
                
                # Clear and rebuild index
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.index.add(embeddings.astype(np.float32))
                
                self.chunks = texts
                self.metadata = metadata
            
            # Save index
            self._save_index(vector_store_path)
            
            self.logger.info(f"Successfully indexed {len(texts)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to index book: {e}")
            return False
    
    def retrieve_similar(self, 
                        query: str,
                        top_k: Optional[int] = None,
                        threshold: Optional[float] = None) -> List[Tuple[str, ChunkMetadata, float]]:
        """
        Retrieve semantically similar chunks for a query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (chunk_text, metadata, similarity_score) tuples
        """
        if not self._initialized or self.index is None or self.index.ntotal == 0:
            return []
        
        top_k = top_k or self.config.top_k
        threshold = threshold or self.config.similarity_threshold
        
        try:
            # Encode query
            with self._lock:
                query_embedding = self.encoder.encode(
                    [query],
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                
                # Search index
                scores, indices = self.index.search(
                    query_embedding.astype(np.float32),
                    min(top_k, self.index.ntotal)
                )
            
            # Filter and format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= threshold and 0 <= idx < len(self.chunks):
                    results.append((
                        self.chunks[idx],
                        self.metadata[idx],
                        float(score)
                    ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve similar chunks: {e}")
            return []
    
    def _create_chunks(self, book: Dict[str, Any]) -> List[Tuple[str, ChunkMetadata]]:
        """
        Create indexed chunks from book content.
        
        Args:
            book: Book data
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        chunks_with_metadata = []
        
        # Process each chapter
        for chapter in book.get("toc", {}).get("chapters", []):
            if not chapter.get("content"):
                continue
            
            chapter_num = chapter.get("number", 0)
            chapter_title = chapter.get("title", "")
            content = chapter.get("content", "")
            
            # Split content into chunks
            chunks = self._split_text(content)
            
            for i, chunk in enumerate(chunks):
                metadata = ChunkMetadata(
                    chapter_number=chapter_num,
                    chunk_index=i,
                    chapter_title=chapter_title,
                    tokens=self._estimate_tokens(chunk)
                )
                chunks_with_metadata.append((chunk, metadata))
            
            # Also index section content if available
            for section in chapter.get("sections", []):
                if section.get("content"):
                    section_chunks = self._split_text(section["content"])
                    for i, chunk in enumerate(section_chunks):
                        metadata = ChunkMetadata(
                            chapter_number=chapter_num,
                            section_number=section.get("number"),
                            chunk_index=i,
                            chapter_title=chapter_title,
                            section_title=section.get("title", ""),
                            tokens=self._estimate_tokens(chunk)
                        )
                        chunks_with_metadata.append((chunk, metadata))
        
        return chunks_with_metadata
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        # Split by paragraphs first for better coherence
        paragraphs = text.split("\n\n")
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph fits in current chunk, add it
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk = (current_chunk + "\n\n" + para).strip()
            else:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Start new chunk with overlap
                if chunks and overlap > 0:
                    # Take last part of previous chunk as overlap
                    overlap_text = chunks[-1][-overlap:] if len(chunks[-1]) > overlap else chunks[-1]
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _save_index(self, vector_store_path: Path):
        """Save index and metadata to disk"""
        try:
            with self._lock:
                # Save FAISS index
                index_file = vector_store_path / "index.faiss"
                faiss.write_index(self.index, str(index_file))
                
                # Save metadata and chunks
                metadata_file = vector_store_path / "metadata.pkl"
                with open(metadata_file, "wb") as f:
                    pickle.dump({
                        "metadata": self.metadata,
                        "chunks": self.chunks,
                        "config": self.config,
                        "timestamp": time.time()
                    }, f)
                
                self.logger.info(f"Saved vector index to {vector_store_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
    
    def _load_index(self, vector_store_path: Path) -> bool:
        """Load index and metadata from disk"""
        try:
            index_file = vector_store_path / "index.faiss"
            metadata_file = vector_store_path / "metadata.pkl"
            
            if not index_file.exists() or not metadata_file.exists():
                return False
            
            with self._lock:
                # Load FAISS index
                self.index = faiss.read_index(str(index_file))
                
                # Load metadata
                with open(metadata_file, "rb") as f:
                    data = pickle.load(f)
                    self.metadata = data["metadata"]
                    self.chunks = data["chunks"]
                
                self.logger.info(f"Loaded vector index with {len(self.chunks)} chunks")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            return False
    
    def _is_index_current(self, book: Dict[str, Any], vector_store_path: Path) -> bool:
        """Check if existing index is up-to-date"""
        metadata_file = vector_store_path / "metadata.pkl"
        
        if not metadata_file.exists():
            return False
        
        try:
            # Check file modification time
            # Simple heuristic: if book has been modified after index, rebuild
            with open(metadata_file, "rb") as f:
                data = pickle.load(f)
                index_timestamp = data.get("timestamp", 0)
            
            # Check if all chapters are indexed
            indexed_chapters = set()
            for meta in data.get("metadata", []):
                if isinstance(meta, ChunkMetadata):
                    indexed_chapters.add(meta.chapter_number)
            
            book_chapters = set()
            for chapter in book.get("toc", {}).get("chapters", []):
                if chapter.get("content"):
                    book_chapters.add(chapter.get("number", 0))
            
            # If chapter count differs, rebuild
            if indexed_chapters != book_chapters:
                self.logger.info("Chapter count changed, rebuilding index")
                return False
            
            return True
            
        except Exception:
            return False
    
    @lru_cache(maxsize=1024)
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        if not text:
            return 0
        words = len(text.split())
        return int(words * 1.3)


class HybridContextManager:
    """
    Advanced hybrid context manager combining multiple strategies.
    
    Features:
    - Dynamic token allocation between different context types
    - Intelligent content selection based on relevance
    - Backward compatibility with non-RAG systems
    - Configurable operation modes
    """
    
    def __init__(self,
                 config: Optional[RAGConfig] = None,
                 provider: Optional[LLMProvider] = None,
                 cache_manager: Optional[CacheManager] = None,
                 max_tokens: int = 128000):
        """
        Initialize the hybrid context manager.
        
        Args:
            config: RAG configuration
            provider: LLM provider for summarization
            cache_manager: Cache manager
            max_tokens: Maximum context window size
        """
        self.config = config or RAGConfig()
        self.provider = provider
        self.cache_manager = cache_manager
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)
        
        # Initialize components based on mode
        if self.config.mode == RAGMode.DISABLED:
            # Use legacy context manager for backward compatibility
            self.legacy_manager = BookContextManager(max_tokens)
            self.summarizer = None
            self.retriever = None
        else:
            self.legacy_manager = BookContextManager(max_tokens)
            self.window_manager = SlidingWindowManager(max_tokens)
            
            # Initialize summarizer if provider is available
            self.summarizer = SmartSummarizer(
                provider=provider,
                cache_manager=cache_manager
            ) if provider else None
            
            # Initialize retriever if in hybrid or full mode
            if self.config.mode in [RAGMode.HYBRID, RAGMode.FULL]:
                self.retriever = SemanticContextRetriever(
                    config=self.config,
                    cache_manager=cache_manager
                )
            else:
                self.retriever = None
    
    def prepare_context(self,
                       book: Dict[str, Any],
                       current_chapter: Optional[int] = None,
                       book_dir: Optional[str] = None,
                       query: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Prepare optimized context using hybrid approach.
        
        Args:
            book: Book data
            current_chapter: Current chapter being generated
            book_dir: Book directory for vector store
            query: Optional query for semantic search
            
        Returns:
            Optimized context as message list
        """
        # Use legacy manager if RAG is disabled
        if self.config.mode == RAGMode.DISABLED:
            return self.legacy_manager.prepare_context(book, current_chapter)
        
        # Calculate token budgets
        available_tokens = self.max_tokens - 4096  # Reserve for response
        core_budget = int(available_tokens * self.config.core_context_ratio)
        rag_budget = int(available_tokens * self.config.rag_context_ratio)
        summary_budget = int(available_tokens * self.config.summary_context_ratio)
        
        elements = []
        
        # 1. Core Context (title, TOC, recent chapters)
        core_elements = self._prepare_core_context(book, current_chapter, core_budget)
        elements.extend(core_elements)
        
        # 2. RAG-Retrieved Context (if available)
        if self.retriever and book_dir:
            # Index book if needed
            self.retriever.index_book(book, book_dir)
            
            # Prepare query for retrieval
            if not query and current_chapter is not None:
                # Use current chapter info as query
                chapters = book.get("toc", {}).get("chapters", [])
                if 0 <= current_chapter < len(chapters):
                    chapter = chapters[current_chapter]
                    query = f"{chapter.get('title', '')} {chapter.get('topics', '')}"
            
            if query:
                rag_elements = self._prepare_rag_context(query, rag_budget)
                elements.extend(rag_elements)
        
        # 3. Summary Context (chapter summaries)
        if self.summarizer:
            summary_elements = self._prepare_summary_context(book, current_chapter, summary_budget)
            elements.extend(summary_elements)
        
        # Optimize all elements to fit
        optimized = self.window_manager.optimize_context(elements)
        
        # Convert to message format
        return self._format_as_messages(optimized, book, current_chapter)
    
    def _prepare_core_context(self,
                             book: Dict[str, Any],
                             current_chapter: Optional[int],
                             budget: int) -> List[ContextElement]:
        """
        Prepare core context elements (title, TOC, recent chapters).
        
        Args:
            book: Book data
            current_chapter: Current chapter number
            budget: Token budget for core context
            
        Returns:
            List of context elements
        """
        elements = []
        used_tokens = 0
        
        # Essential: Book title
        title = book.get("title", "Untitled")
        title_text = f"Book Title: {title}"
        title_tokens = self._estimate_tokens(title_text)
        
        elements.append(ContextElement(
            content=title_text,
            tokens=title_tokens,
            priority=ContextPriority.ESSENTIAL,
            metadata={"type": "title"}
        ))
        used_tokens += title_tokens
        
        # High priority: Table of contents
        if book.get("toc"):
            toc_text = self._format_toc_compact(book["toc"], current_chapter)
            toc_tokens = self._estimate_tokens(toc_text)
            
            if used_tokens + toc_tokens <= budget:
                elements.append(ContextElement(
                    content=f"Table of Contents:\n{toc_text}",
                    tokens=toc_tokens,
                    priority=ContextPriority.HIGH,
                    metadata={"type": "toc"}
                ))
                used_tokens += toc_tokens
        
        # Recent chapters (sliding window)
        if current_chapter is not None and book.get("toc", {}).get("chapters"):
            chapters = book["toc"]["chapters"]
            
            # Prioritize chapters around current
            window_size = 3  # Chapters before current to include
            start_idx = max(0, current_chapter - window_size)
            end_idx = min(len(chapters), current_chapter + 1)
            
            for i in range(start_idx, end_idx):
                if i >= len(chapters) or not chapters[i].get("content"):
                    continue
                
                chapter = chapters[i]
                recency = window_size - abs(i - current_chapter)
                
                # Full content for immediately previous chapter
                if i == current_chapter - 1:
                    content = chapter["content"]
                    content_tokens = self._estimate_tokens(content)
                    
                    # Truncate if needed
                    if used_tokens + content_tokens > budget:
                        available = budget - used_tokens
                        content = self._truncate_to_tokens(content, available)
                        content_tokens = available
                    
                    if content_tokens > 0:
                        elements.append(ContextElement(
                            content=f"Chapter {chapter['number']}: {chapter['title']}\n{content}",
                            tokens=content_tokens,
                            priority=ContextPriority.HIGH,
                            metadata={"type": "chapter", "number": i, "recency": recency}
                        ))
                        used_tokens += content_tokens
                
                # Title and brief for other recent chapters
                else:
                    brief = f"Chapter {chapter['number']}: {chapter['title']}"
                    brief_tokens = self._estimate_tokens(brief)
                    
                    if used_tokens + brief_tokens <= budget:
                        elements.append(ContextElement(
                            content=brief,
                            tokens=brief_tokens,
                            priority=ContextPriority.MEDIUM,
                            metadata={"type": "chapter_brief", "number": i, "recency": recency}
                        ))
                        used_tokens += brief_tokens
        
        return elements
    
    def _prepare_rag_context(self,
                            query: str,
                            budget: int) -> List[ContextElement]:
        """
        Prepare RAG-retrieved context elements.
        
        Args:
            query: Query for semantic search
            budget: Token budget for RAG context
            
        Returns:
            List of context elements
        """
        if not self.retriever:
            return []
        
        elements = []
        used_tokens = 0
        
        # Retrieve similar chunks
        similar_chunks = self.retriever.retrieve_similar(query)
        
        for chunk_text, metadata, score in similar_chunks:
            chunk_tokens = metadata.tokens or self._estimate_tokens(chunk_text)
            
            if used_tokens + chunk_tokens > budget:
                # Try to fit partial chunk
                available = budget - used_tokens
                if available > 100:  # Minimum useful chunk size
                    chunk_text = self._truncate_to_tokens(chunk_text, available)
                    chunk_tokens = available
                else:
                    break
            
            # Add chunk as context element
            elements.append(ContextElement(
                content=f"[Relevant content from Chapter {metadata.chapter_number}]:\n{chunk_text}",
                tokens=chunk_tokens,
                priority=ContextPriority.MEDIUM,
                metadata={
                    "type": "rag_chunk",
                    "chapter": metadata.chapter_number,
                    "score": score
                }
            ))
            used_tokens += chunk_tokens
            
            if used_tokens >= budget:
                break
        
        return elements
    
    def _prepare_summary_context(self,
                                 book: Dict[str, Any],
                                 current_chapter: Optional[int],
                                 budget: int) -> List[ContextElement]:
        """
        Prepare summary context elements.
        
        Args:
            book: Book data
            current_chapter: Current chapter number
            budget: Token budget for summaries
            
        Returns:
            List of context elements
        """
        if not self.summarizer:
            return []
        
        elements = []
        used_tokens = 0
        
        chapters = book.get("toc", {}).get("chapters", [])
        
        # Summarize previous chapters (prioritize recent ones)
        if current_chapter is not None and current_chapter > 0:
            # Start from most recent and work backward
            for i in range(current_chapter - 1, -1, -1):
                if i >= len(chapters) or not chapters[i].get("content"):
                    continue
                
                chapter = chapters[i]
                
                # Skip if we already have full content in core context
                if i == current_chapter - 1:
                    continue
                
                # Generate or retrieve summary
                summary = self.summarizer.summarize_chapter(chapter, book)
                summary_tokens = self._estimate_tokens(summary)
                
                if used_tokens + summary_tokens > budget:
                    # Try to fit partial summary
                    available = budget - used_tokens
                    if available > 50:
                        summary = self._truncate_to_tokens(summary, available)
                        summary_tokens = available
                    else:
                        break
                
                elements.append(ContextElement(
                    content=f"Chapter {chapter['number']} Summary:\n{summary}",
                    tokens=summary_tokens,
                    priority=ContextPriority.LOW,
                    metadata={"type": "summary", "chapter": i}
                ))
                used_tokens += summary_tokens
                
                if used_tokens >= budget:
                    break
        
        return elements
    
    def _format_as_messages(self,
                           elements: List[ContextElement],
                           book: Dict[str, Any],
                           current_chapter: Optional[int]) -> List[Dict[str, str]]:
        """
        Format context elements as conversation messages.
        
        Args:
            elements: Optimized context elements
            book: Book data
            current_chapter: Current chapter number
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # System message with essential context
        system_parts = ["You are writing a book."]
        
        # Add essential elements to system message
        for elem in elements:
            if elem.priority == ContextPriority.ESSENTIAL:
                system_parts.append(elem.content)
        
        messages.append({
            "role": "system",
            "content": " ".join(system_parts)
        })
        
        # Group other elements by type for better organization
        toc_elements = []
        chapter_elements = []
        rag_elements = []
        summary_elements = []
        
        for elem in elements:
            if elem.priority == ContextPriority.ESSENTIAL:
                continue  # Already in system message
            
            elem_type = elem.metadata.get("type", "")
            if "toc" in elem_type:
                toc_elements.append(elem.content)
            elif "rag" in elem_type:
                rag_elements.append(elem.content)
            elif "summary" in elem_type:
                summary_elements.append(elem.content)
            else:
                chapter_elements.append(elem.content)
        
        # Add grouped context as system messages
        if toc_elements:
            messages.append({
                "role": "system",
                "content": "\n".join(toc_elements)
            })
        
        if chapter_elements:
            messages.append({
                "role": "system",
                "content": "Previous content:\n" + "\n\n".join(chapter_elements)
            })
        
        if rag_elements:
            messages.append({
                "role": "system",
                "content": "Related content:\n" + "\n\n".join(rag_elements)
            })
        
        if summary_elements:
            messages.append({
                "role": "system",
                "content": "Chapter summaries:\n" + "\n\n".join(summary_elements)
            })
        
        # Add instruction for current chapter
        if current_chapter is not None:
            chapters = book.get("toc", {}).get("chapters", [])
            if 0 <= current_chapter < len(chapters):
                chapter = chapters[current_chapter]
                messages.append({
                    "role": "user",
                    "content": f"Write Chapter {chapter['number']}: {chapter['title']}"
                })
        
        return messages
    
    def _format_toc_compact(self, toc: Dict[str, Any], highlight_chapter: Optional[int]) -> str:
        """Format table of contents in compact form"""
        lines = []
        
        for i, chapter in enumerate(toc.get("chapters", [])):
            marker = ">>>" if i == highlight_chapter else ""
            lines.append(f"{marker} {chapter['number']}. {chapter['title']}")
            
            # Include section count for context
            sections = chapter.get("sections", [])
            if sections and (highlight_chapter is None or abs(i - (highlight_chapter or 0)) <= 2):
                lines.append(f"  ({len(sections)} sections)")
        
        return "\n".join(lines)
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens"""
        if not text:
            return ""
        
        # Rough approximation: 1.3 tokens per word
        max_words = int(max_tokens / 1.3)
        words = text.split()
        
        if len(words) <= max_words:
            return text
        
        truncated = " ".join(words[:max_words])
        return truncated + "..."
    
    @lru_cache(maxsize=2048)
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        if not text:
            return 0
        
        # Use provider's tokenizer if available
        if self.provider and hasattr(self.provider, "count_tokens"):
            try:
                return self.provider.count_tokens(text)
            except Exception:
                pass
        
        # Fallback to estimation
        words = len(text.split())
        punctuation = sum(1 for char in text if char in '.,;:!?"\'()[]{}')
        return int(words * 1.3 + punctuation * 0.1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about context management"""
        stats = {
            "mode": self.config.mode.value,
            "max_tokens": self.max_tokens,
            "token_distribution": {
                "core": f"{self.config.core_context_ratio:.0%}",
                "rag": f"{self.config.rag_context_ratio:.0%}",
                "summary": f"{self.config.summary_context_ratio:.0%}"
            }
        }
        
        if self.retriever and self.retriever.index:
            stats["vector_index"] = {
                "chunks": self.retriever.index.ntotal,
                "embedding_dim": self.retriever.embedding_dim
            }
        
        return stats


# Convenience function for backward compatibility
def create_hybrid_manager(provider: Optional[LLMProvider] = None,
                         cache_manager: Optional[CacheManager] = None,
                         config: Optional[RAGConfig] = None,
                         max_tokens: int = 128000) -> HybridContextManager:
    """
    Create a hybrid context manager with sensible defaults.
    
    Args:
        provider: LLM provider for summarization
        cache_manager: Cache manager
        config: Optional RAG configuration
        max_tokens: Maximum context window
        
    Returns:
        Configured HybridContextManager instance
    """
    if config is None:
        # Check if RAG dependencies are available
        if SENTENCE_TRANSFORMERS_AVAILABLE and FAISS_AVAILABLE:
            config = RAGConfig(mode=RAGMode.HYBRID)
        else:
            # Fallback to basic mode without vector search
            config = RAGConfig(mode=RAGMode.BASIC)
            logging.warning("RAG dependencies not available, using basic mode")
    
    return HybridContextManager(
        config=config,
        provider=provider,
        cache_manager=cache_manager,
        max_tokens=max_tokens
    )