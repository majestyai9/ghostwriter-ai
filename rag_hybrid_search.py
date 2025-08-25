"""
Hybrid Search System for Enhanced RAG.

This module implements hybrid search combining dense and sparse retrieval methods
for improved retrieval accuracy and relevance in the RAG system.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import math
import re
from functools import lru_cache
import pickle
import json
from pathlib import Path

# Optional imports
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
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search system."""
    # Dense retrieval settings
    dense_weight: float = 0.6
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Sparse retrieval settings  
    sparse_weight: float = 0.4
    max_features: int = 10000
    min_df: int = 2
    max_df: float = 0.95
    use_idf: bool = True
    sublinear_tf: bool = True
    
    # BM25 settings
    k1: float = 1.2
    b: float = 0.75
    
    # Retrieval settings
    top_k: int = 20
    rerank: bool = True
    min_score: float = 0.3
    
    # Caching
    enable_cache: bool = True
    cache_size: int = 1000
    
    # Performance
    batch_size: int = 32
    use_gpu: bool = True


class BM25Index:
    """BM25 sparse retrieval index for text search."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avgdl = 0
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = []
        self.idf = {}
        self.tokenized_docs = []
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def fit(self, documents: List[str]):
        """Build BM25 index from documents."""
        self.doc_count = len(documents)
        self.tokenized_docs = []
        self.doc_lengths = []
        
        # Tokenize and compute statistics
        for doc in documents:
            tokens = self._tokenize(doc)
            self.tokenized_docs.append(tokens)
            self.doc_lengths.append(len(tokens))
            
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
        
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Compute IDF scores
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1)
    
    def score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score for a query-document pair."""
        query_tokens = self._tokenize(query)
        doc_tokens = self.tokenized_docs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        score = 0.0
        doc_freqs = defaultdict(int)
        for token in doc_tokens:
            doc_freqs[token] += 1
        
        for token in query_tokens:
            if token not in self.idf:
                continue
                
            freq = doc_freqs[token]
            if freq == 0:
                continue
                
            idf_score = self.idf[token]
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf_score * numerator / denominator
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search for top-k documents matching the query."""
        scores = []
        for idx in range(len(self.tokenized_docs)):
            score = self.score(query, idx)
            if score > 0:
                scores.append((idx, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridSearchEngine:
    """
    Hybrid search engine combining dense and sparse retrieval.
    
    Features:
    - Dense retrieval using semantic embeddings (FAISS)
    - Sparse retrieval using BM25 and TF-IDF
    - Score fusion with configurable weights
    - Reciprocal Rank Fusion (RRF) for combining results
    - Query expansion and reranking
    """
    
    def __init__(self, config: Optional[HybridSearchConfig] = None):
        self.config = config or HybridSearchConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.dense_index = None
        self.sparse_index = None
        self.bm25_index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Embeddings
        self.encoder = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer(self.config.embedding_model)
                if self.config.use_gpu and hasattr(self.encoder, 'cuda'):
                    self.encoder = self.encoder.cuda()
            except Exception as e:
                self.logger.warning(f"Failed to initialize encoder: {e}")
        
        # Documents and metadata
        self.documents = []
        self.doc_embeddings = None
        self.doc_metadata = []
        
        # Caching
        self._cache = {} if self.config.enable_cache else None
        self._cache_order = []
        
        # Statistics
        self.stats = {
            "dense_searches": 0,
            "sparse_searches": 0,
            "hybrid_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def index_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Index documents for hybrid search.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        self.documents = documents
        self.doc_metadata = metadata or [{} for _ in documents]
        
        # Build dense index
        if self.encoder and FAISS_AVAILABLE:
            self._build_dense_index(documents)
        
        # Build sparse indices
        self._build_sparse_indices(documents)
        
        # Clear cache after reindexing
        if self._cache:
            self._cache.clear()
            self._cache_order.clear()
    
    def _build_dense_index(self, documents: List[str]):
        """Build FAISS index for dense retrieval."""
        try:
            # Generate embeddings
            self.logger.info(f"Generating embeddings for {len(documents)} documents")
            embeddings = []
            
            for i in range(0, len(documents), self.config.batch_size):
                batch = documents[i:i + self.config.batch_size]
                batch_embeddings = self.encoder.encode(batch, convert_to_numpy=True)
                embeddings.append(batch_embeddings)
            
            self.doc_embeddings = np.vstack(embeddings).astype('float32')
            
            # Build FAISS index
            dimension = self.doc_embeddings.shape[1]
            
            if len(documents) > 10000:
                # Use IVF index for large collections
                nlist = int(np.sqrt(len(documents)))
                quantizer = faiss.IndexFlatL2(dimension)
                self.dense_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                self.dense_index.train(self.doc_embeddings)
            else:
                # Use flat index for small collections
                self.dense_index = faiss.IndexFlatL2(dimension)
            
            self.dense_index.add(self.doc_embeddings)
            
            if self.config.use_gpu and faiss.get_num_gpus() > 0:
                self.dense_index = faiss.index_cpu_to_gpu(
                    faiss.StandardGpuResources(), 0, self.dense_index
                )
            
            self.logger.info(f"Dense index built with {self.dense_index.ntotal} vectors")
            
        except Exception as e:
            self.logger.error(f"Failed to build dense index: {e}")
            self.dense_index = None
    
    def _build_sparse_indices(self, documents: List[str]):
        """Build sparse retrieval indices (BM25 and TF-IDF)."""
        try:
            # Build BM25 index
            self.bm25_index = BM25Index(k1=self.config.k1, b=self.config.b)
            self.bm25_index.fit(documents)
            
            # Build TF-IDF index if sklearn available
            if SKLEARN_AVAILABLE:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=self.config.max_features,
                    min_df=self.config.min_df,
                    max_df=self.config.max_df,
                    use_idf=self.config.use_idf,
                    sublinear_tf=self.config.sublinear_tf,
                    lowercase=True,
                    stop_words='english'
                )
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
                self.logger.info(f"TF-IDF index built with shape {self.tfidf_matrix.shape}")
            
            self.logger.info("Sparse indices built successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to build sparse indices: {e}")
    
    def search(self, query: str, top_k: Optional[int] = None, 
               hybrid: bool = True) -> List[Dict[str, Any]]:
        """
        Perform hybrid search on indexed documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            hybrid: Whether to use hybrid search (True) or dense-only (False)
        
        Returns:
            List of search results with scores and metadata
        """
        top_k = top_k or self.config.top_k
        
        # Check cache
        cache_key = (query, top_k, hybrid)
        if self._cache and cache_key in self._cache:
            self.stats["cache_hits"] += 1
            return self._cache[cache_key]
        
        self.stats["cache_misses"] += 1
        
        if hybrid:
            results = self._hybrid_search(query, top_k)
            self.stats["hybrid_searches"] += 1
        else:
            results = self._dense_search(query, top_k)
            self.stats["dense_searches"] += 1
        
        # Update cache
        if self._cache is not None:
            self._update_cache(cache_key, results)
        
        return results
    
    def _dense_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform dense semantic search."""
        if not self.dense_index or not self.encoder:
            return []
        
        try:
            # Encode query
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding.astype('float32')
            
            # Search
            scores, indices = self.dense_index.search(query_embedding, top_k)
            
            # Format results
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0 and idx < len(self.documents):
                    # Convert L2 distance to similarity score
                    similarity = 1 / (1 + score)
                    results.append({
                        "index": int(idx),
                        "text": self.documents[idx],
                        "score": float(similarity),
                        "metadata": self.doc_metadata[idx],
                        "method": "dense"
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Dense search failed: {e}")
            return []
    
    def _sparse_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform sparse keyword search."""
        results = []
        
        # BM25 search
        if self.bm25_index:
            bm25_results = self.bm25_index.search(query, top_k)
            for idx, score in bm25_results:
                results.append({
                    "index": idx,
                    "text": self.documents[idx],
                    "score": score,
                    "metadata": self.doc_metadata[idx],
                    "method": "bm25"
                })
        
        # TF-IDF search
        if self.tfidf_vectorizer and self.tfidf_matrix is not None:
            try:
                query_vec = self.tfidf_vectorizer.transform([query])
                similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                for idx in top_indices:
                    if similarities[idx] > 0:
                        results.append({
                            "index": int(idx),
                            "text": self.documents[idx],
                            "score": float(similarities[idx]),
                            "metadata": self.doc_metadata[idx],
                            "method": "tfidf"
                        })
            except Exception as e:
                self.logger.error(f"TF-IDF search failed: {e}")
        
        self.stats["sparse_searches"] += 1
        return results
    
    def _hybrid_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse results.
        
        Uses Reciprocal Rank Fusion (RRF) to combine rankings.
        """
        # Get results from both methods
        dense_results = self._dense_search(query, top_k * 2)
        sparse_results = self._sparse_search(query, top_k * 2)
        
        # Reciprocal Rank Fusion
        k = 60  # RRF constant
        doc_scores = defaultdict(float)
        doc_data = {}
        
        # Process dense results
        for rank, result in enumerate(dense_results):
            idx = result["index"]
            doc_scores[idx] += self.config.dense_weight / (k + rank + 1)
            doc_data[idx] = result
        
        # Process sparse results
        for rank, result in enumerate(sparse_results):
            idx = result["index"]
            doc_scores[idx] += self.config.sparse_weight / (k + rank + 1)
            if idx not in doc_data:
                doc_data[idx] = result
        
        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format results
        results = []
        for idx, score in sorted_docs[:top_k]:
            if score >= self.config.min_score:
                result = doc_data[idx].copy()
                result["score"] = score
                result["method"] = "hybrid"
                results.append(result)
        
        # Optional reranking
        if self.config.rerank and self.encoder:
            results = self._rerank_results(query, results)
        
        return results
    
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rerank results using cross-encoder or more sophisticated scoring."""
        # For now, just ensure score normalization
        # In future, could use cross-encoder models for better reranking
        if not results:
            return results
        
        max_score = max(r["score"] for r in results)
        if max_score > 0:
            for result in results:
                result["score"] = result["score"] / max_score
        
        return results
    
    def _update_cache(self, key: Any, value: Any):
        """Update LRU cache with size limit."""
        if key in self._cache:
            self._cache_order.remove(key)
        
        self._cache[key] = value
        self._cache_order.append(key)
        
        # Enforce cache size limit
        while len(self._cache) > self.config.cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        return self.stats.copy()
    
    def save_index(self, path: str):
        """Save hybrid search index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save dense index
        if self.dense_index and FAISS_AVAILABLE:
            faiss.write_index(self.dense_index, str(path / "dense.index"))
        
        # Save sparse indices
        if self.bm25_index:
            with open(path / "bm25.pkl", "wb") as f:
                pickle.dump(self.bm25_index, f)
        
        if self.tfidf_vectorizer:
            with open(path / "tfidf_vectorizer.pkl", "wb") as f:
                pickle.dump(self.tfidf_vectorizer, f)
            
            with open(path / "tfidf_matrix.pkl", "wb") as f:
                pickle.dump(self.tfidf_matrix, f)
        
        # Save documents and metadata
        with open(path / "documents.json", "w", encoding="utf-8") as f:
            json.dump({
                "documents": self.documents,
                "metadata": self.doc_metadata
            }, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Hybrid search index saved to {path}")
    
    def load_index(self, path: str) -> bool:
        """Load hybrid search index from disk."""
        path = Path(path)
        if not path.exists():
            return False
        
        try:
            # Load dense index
            dense_path = path / "dense.index"
            if dense_path.exists() and FAISS_AVAILABLE:
                self.dense_index = faiss.read_index(str(dense_path))
            
            # Load BM25 index
            bm25_path = path / "bm25.pkl"
            if bm25_path.exists():
                with open(bm25_path, "rb") as f:
                    self.bm25_index = pickle.load(f)
            
            # Load TF-IDF
            tfidf_vec_path = path / "tfidf_vectorizer.pkl"
            tfidf_mat_path = path / "tfidf_matrix.pkl"
            if tfidf_vec_path.exists() and tfidf_mat_path.exists():
                with open(tfidf_vec_path, "rb") as f:
                    self.tfidf_vectorizer = pickle.load(f)
                with open(tfidf_mat_path, "rb") as f:
                    self.tfidf_matrix = pickle.load(f)
            
            # Load documents
            docs_path = path / "documents.json"
            if docs_path.exists():
                with open(docs_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents = data["documents"]
                    self.doc_metadata = data["metadata"]
            
            self.logger.info(f"Hybrid search index loaded from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            return False