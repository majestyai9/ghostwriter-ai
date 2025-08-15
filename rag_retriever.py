"""
Semantic context retrieval module for RAG system.

Provides high-performance semantic search with GPU acceleration.
"""

import logging
import pickle
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

from rag_models import ChunkMetadata
from rag_vector_cache import VectorCache
from rag_indexer import OptimizedFAISSIndexer
from rag_chunker import OptimizedTextChunker, ChunkingConfig


class SemanticContextRetriever:
    """
    Semantic search and retrieval system using sentence-transformers and FAISS.
    
    Features:
    - IVF indexing for large-scale search
    - GPU acceleration support
    - Batch processing capabilities
    - Vector caching for frequent queries
    - Performance metrics and logging
    """

    def __init__(self, config: Any, cache_manager: Any = None):
        """Initialize the semantic context retriever."""
        self.config = config
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.encoder = None
        self.indexer = None
        self.chunker = None
        self.metadata = []
        self.chunks = []
        self.embedding_dim = 0
        self.use_gpu = False
        
        # Vector cache for frequent queries
        self.vector_cache = VectorCache(
            max_size=config.cache_size,
            ttl=config.cache_ttl
        )
        
        # Performance metrics
        self.metrics = {
            "index_time": 0,
            "search_time": 0,
            "total_searches": 0,
            "total_indexed": 0
        }

        self._lock = threading.RLock()
        self._initialized = False

        # Initialize if dependencies are available
        if SENTENCE_TRANSFORMERS_AVAILABLE and FAISS_AVAILABLE:
            self._initialize()

    def _initialize(self):
        """Initialize the retriever components with GPU support."""
        with self._lock:
            if self._initialized:
                return

            try:
                device = "cuda" if (self.config.use_gpu and CUDA_AVAILABLE) else "cpu"
                self.use_gpu = device == "cuda"
                self.logger.info(f"Using {device.upper()} for embeddings")
                
                self.logger.info(f"Loading embedding model: {self.config.embedding_model}")
                self.encoder = SentenceTransformer(
                    self.config.embedding_model,
                    device=device
                )

                # Get embedding dimension
                dummy_embedding = self.encoder.encode(["test"], show_progress_bar=False)
                self.embedding_dim = dummy_embedding.shape[1]

                self.indexer = OptimizedFAISSIndexer(
                    embedding_dim=self.embedding_dim,
                    use_gpu=self.use_gpu,
                    use_ivf=self.config.use_ivf,
                    ivf_nlist=self.config.ivf_nlist,
                    ivf_nprobe=self.config.ivf_nprobe
                )
                
                self.chunker = OptimizedTextChunker(ChunkingConfig(
                    strategy="semantic",
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                    preserve_sentences=True
                ))

                self._initialized = True
                self.logger.info("Semantic retriever initialized successfully")

            except Exception as e:
                self.logger.error(f"Failed to initialize semantic retriever: {e}")
                self._initialized = False

    def index_book(self, book: Dict[str, Any], book_dir: str) -> bool:
        """Index or update the vector store for a book."""
        if not self._initialized:
            return False

        try:
            vector_store_path = Path(book_dir) / self.config.vector_store_dir
            vector_store_path.mkdir(exist_ok=True)

            # Check if index already exists and is up-to-date
            if self._is_index_current(book, vector_store_path):
                self.logger.info("Vector index is up-to-date, loading existing index")
                return self._load_index(vector_store_path)

            # Create new index
            self.logger.info("Creating new vector index for book")
            chunks_with_metadata = self._create_chunks(book)

            if not chunks_with_metadata:
                self.logger.warning("No content to index")
                return False

            # Encode chunks with optimized batching
            texts = [chunk for chunk, _ in chunks_with_metadata]
            metadata = [meta for _, meta in chunks_with_metadata]

            self.logger.info(f"Encoding {len(texts)} chunks...")
            start_time = time.time()
            
            with self._lock:
                # Use larger batch size for GPU
                batch_size = self.config.gpu_batch_size if self.use_gpu else 32
                
                embeddings = self.encoder.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                
                # Build index using optimized indexer
                self.indexer.build_index(embeddings)
                
                self.chunks = texts
                self.metadata = metadata
                
                # Update metrics
                self.metrics["index_time"] = time.time() - start_time
                self.metrics["total_indexed"] = len(texts)
                
                self.logger.info(
                    f"Indexed {len(texts)} chunks in {self.metrics['index_time']:.2f}s "
                    f"({len(texts)/self.metrics['index_time']:.0f} chunks/s)"
                )

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
        """Retrieve semantically similar chunks for a query."""
        if not self._initialized or self.indexer is None or self.indexer.index is None:
            return []

        top_k = top_k or self.config.top_k
        threshold = threshold or self.config.similarity_threshold
        
        # Check cache first
        cache_key = f"{query}:{top_k}:{threshold}"
        cached_result = self.vector_cache.get(cache_key)
        if cached_result:
            self.logger.debug(f"Cache hit for query: {query[:50]}...")
            return cached_result

        try:
            start_time = time.time()
            
            with self._lock:
                query_embedding = self.encoder.encode(
                    [query],
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                scores, indices = self.indexer.search(
                    query_embedding,
                    min(top_k, self.indexer.index.ntotal)
                )
                search_time = time.time() - start_time
                self.metrics["search_time"] += search_time
                self.metrics["total_searches"] += 1

            # Filter and format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= threshold and 0 <= idx < len(self.chunks):
                    results.append((
                        self.chunks[idx],
                        self.metadata[idx],
                        float(score)
                    ))
            
            # Cache results
            if results and self.config.enable_caching:
                self.vector_cache.set(cache_key, results, 1.0)
            
            self.logger.debug(
                f"Search completed in {search_time:.3f}s, found {len(results)} results"
            )
            
            return results

        except Exception as e:
            self.logger.error(f"Failed to retrieve similar chunks: {e}")
            return []

    def batch_retrieve(self,
                       queries: List[str],
                       top_k: Optional[int] = None,
                       threshold: Optional[float] = None) -> List[List[Tuple[str, ChunkMetadata, float]]]:
        """Retrieve similar chunks for multiple queries in batch."""
        if not self._initialized or self.indexer is None or self.indexer.index is None:
            return [[] for _ in queries]
        
        if not self.config.enable_batch_processing:
            # Fall back to sequential processing
            return [self.retrieve_similar(q, top_k, threshold) for q in queries]
        
        top_k = top_k or self.config.top_k
        threshold = threshold or self.config.similarity_threshold
        
        try:
            start_time = time.time()
            
            with self._lock:
                batch_size = self.config.gpu_batch_size if self.use_gpu else 32
                query_embeddings = self.encoder.encode(
                    queries,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                scores, indices = self.indexer.search(
                    query_embeddings,
                    min(top_k, self.indexer.index.ntotal)
                )
                search_time = time.time() - start_time
                self.metrics["search_time"] += search_time
                self.metrics["total_searches"] += len(queries)
            
            all_results = []
            for q_idx, query in enumerate(queries):
                results = []
                for score, idx in zip(scores[q_idx], indices[q_idx]):
                    if score >= threshold and 0 <= idx < len(self.chunks):
                        results.append((self.chunks[idx], self.metadata[idx], float(score)))
                all_results.append(results)
            
            self.logger.info(
                f"Batch search for {len(queries)} queries completed in {search_time:.3f}s "
                f"({len(queries)/search_time:.1f} queries/s)"
            )
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Batch retrieval failed: {e}")
            return [self.retrieve_similar(q, top_k, threshold) for q in queries]

    def _create_chunks(self, book: Dict[str, Any]) -> List[Tuple[str, ChunkMetadata]]:
        """Create indexed chunks from book content."""
        chunks_with_metadata = []

        # Process each chapter
        for chapter in book.get("toc", {}).get("chapters", []):
            if not chapter.get("content"):
                continue

            chapter_num = chapter.get("number", 0)
            chapter_title = chapter.get("title", "")
            content = chapter.get("content", "")

            # Split content into chunks using optimized chunker
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
        """Split text into chunks using optimized chunker."""
        if not text:
            return []
        
        if self.chunker:
            return self.chunker.chunk_text(text)
        
        # Fallback to simple splitting
        chunks = []
        chunk_size = self.config.chunk_size
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _save_index(self, vector_store_path: Path):
        """Save index and metadata to disk."""
        try:
            with self._lock:
                # Save index using optimized indexer
                index_file = vector_store_path / "index.faiss"
                if self.indexer and self.indexer.index:
                    self.indexer.save_index(str(index_file))

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
        """Load index and metadata from disk."""
        try:
            index_file = vector_store_path / "index.faiss"
            metadata_file = vector_store_path / "metadata.pkl"

            if not index_file.exists() or not metadata_file.exists():
                return False

            with self._lock:
                # Load index using optimized indexer
                if self.indexer:
                    self.indexer.load_index(str(index_file))

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
        """Check if existing index is up-to-date."""
        metadata_file = vector_store_path / "metadata.pkl"

        if not metadata_file.exists():
            return False

        try:
            with open(metadata_file, "rb") as f:
                data = pickle.load(f)

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
        """Estimate token count for text."""
        if not text:
            return 0
        words = len(text.split())
        return int(words * 1.3)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        avg_search_time = (
            self.metrics["search_time"] / self.metrics["total_searches"]
            if self.metrics["total_searches"] > 0 else 0
        )
        
        return {
            "index_stats": {
                "total_chunks": len(self.chunks) if self.chunks else 0,
                "index_size": self.indexer.index.ntotal if self.indexer and self.indexer.index else 0,
                "embedding_dim": self.embedding_dim,
                "using_gpu": self.use_gpu,
                "using_ivf": self.indexer.metrics["using_ivf"] if self.indexer else False
            },
            "performance": {
                "index_time": f"{self.metrics['index_time']:.2f}s",
                "total_indexed": self.metrics["total_indexed"],
                "total_searches": self.metrics["total_searches"],
                "avg_search_time": f"{avg_search_time:.3f}s",
                "searches_per_second": f"{1/avg_search_time:.1f}" if avg_search_time > 0 else "N/A"
            },
            "cache": self.vector_cache.get_stats()
        }