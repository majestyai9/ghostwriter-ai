"""
Optimized FAISS indexing module with IVF and GPU support.

Provides high-performance vector indexing for large document collections.
"""

import logging
import time
from typing import Optional

import numpy as np

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


class OptimizedFAISSIndexer:
    """
    High-performance FAISS indexer with IVF and GPU support.
    
    Features:
    - IVF (Inverted File Index) for large-scale search
    - GPU acceleration when available
    - Automatic index type selection based on data size
    - Performance metrics tracking
    """
    
    def __init__(self, 
                 embedding_dim: int,
                 use_gpu: bool = True,
                 use_ivf: bool = True,
                 ivf_nlist: int = 100,
                 ivf_nprobe: int = 10):
        """
        Initialize the indexer.
        
        Args:
            embedding_dim: Dimension of embeddings
            use_gpu: Whether to try using GPU
            use_ivf: Whether to use IVF for large datasets
            ivf_nlist: Number of clusters for IVF
            ivf_nprobe: Number of clusters to search
        """
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.use_ivf = use_ivf
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe
        self.logger = logging.getLogger(__name__)
        
        self.index = None
        self.gpu_resources = None
        self.is_trained = False
        
        # Performance metrics
        self.metrics = {
            "build_time": 0,
            "search_time": 0,
            "num_vectors": 0,
            "using_gpu": False,
            "using_ivf": False
        }
    
    def create_index(self, num_vectors: int) -> faiss.Index:
        """
        Create appropriate FAISS index based on data size.
        
        Args:
            num_vectors: Number of vectors to index
            
        Returns:
            Configured FAISS index
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS is not available")
        
        # Determine if IVF should be used
        use_ivf = self.use_ivf and num_vectors > 1000
        
        if use_ivf:
            # Calculate appropriate nlist
            nlist = min(self.ivf_nlist, int(np.sqrt(num_vectors)))
            
            self.logger.info(
                f"Creating IVF index: {nlist} clusters for {num_vectors} vectors"
            )
            
            # Create quantizer
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            
            # Create IVF index
            index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            
            self.metrics["using_ivf"] = True
            self.is_trained = False  # IVF needs training
        else:
            # Use flat index for small datasets
            self.logger.info(f"Creating flat index for {num_vectors} vectors")
            index = faiss.IndexFlatIP(self.embedding_dim)
            self.metrics["using_ivf"] = False
            self.is_trained = True  # Flat index doesn't need training
        
        # Try to move to GPU if requested
        if self.use_gpu and CUDA_AVAILABLE:
            index = self._move_to_gpu(index)
        
        self.index = index
        self.metrics["num_vectors"] = num_vectors
        
        return index
    
    def _move_to_gpu(self, index: faiss.Index) -> faiss.Index:
        """
        Move index to GPU if possible.
        
        Args:
            index: CPU index
            
        Returns:
            GPU index or original if GPU transfer fails
        """
        try:
            if self.gpu_resources is None:
                self.gpu_resources = faiss.StandardGpuResources()
            
            # Configure GPU options
            gpu_options = faiss.GpuIndexFlatConfig()
            gpu_options.device = 0  # Use first GPU
            
            # Move to GPU
            gpu_index = faiss.index_cpu_to_gpu(
                self.gpu_resources, 0, index
            )
            
            self.logger.info("Successfully moved index to GPU")
            self.metrics["using_gpu"] = True
            
            return gpu_index
            
        except Exception as e:
            self.logger.warning(f"Failed to use GPU for FAISS: {e}")
            self.metrics["using_gpu"] = False
            return index
    
    def build_index(self, embeddings: np.ndarray) -> float:
        """
        Build index from embeddings.
        
        Args:
            embeddings: Vector embeddings to index
            
        Returns:
            Build time in seconds
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS is not available")
        
        start_time = time.time()
        
        # Create index if not exists
        if self.index is None:
            self.create_index(len(embeddings))
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)
        
        # Train if needed (for IVF)
        if hasattr(self.index, 'train') and not self.is_trained:
            self.logger.info("Training IVF index...")
            
            # Use subset for training if dataset is very large
            if len(embeddings) > 100000:
                train_size = min(50000, len(embeddings))
                train_indices = np.random.choice(
                    len(embeddings), train_size, replace=False
                )
                train_data = embeddings[train_indices]
            else:
                train_data = embeddings
            
            self.index.train(train_data)
            self.is_trained = True
            
            # Set search parameters
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.ivf_nprobe
        
        # Add vectors
        self.logger.info(f"Adding {len(embeddings)} vectors to index...")
        self.index.add(embeddings)
        
        build_time = time.time() - start_time
        self.metrics["build_time"] = build_time
        self.metrics["num_vectors"] = len(embeddings)
        
        self.logger.info(
            f"Index built in {build_time:.2f}s "
            f"({len(embeddings)/build_time:.0f} vectors/s)"
        )
        
        return build_time
    
    def search(self, 
               queries: np.ndarray, 
               k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Search index for nearest neighbors.
        
        Args:
            queries: Query vectors
            k: Number of neighbors to retrieve
            
        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None or self.index.ntotal == 0:
            return np.array([[]]), np.array([[]])
        
        start_time = time.time()
        
        # Ensure queries are float32
        queries = queries.astype(np.float32)
        
        # Ensure k doesn't exceed index size
        k = min(k, self.index.ntotal)
        
        # Search
        distances, indices = self.index.search(queries, k)
        
        search_time = time.time() - start_time
        self.metrics["search_time"] += search_time
        
        return distances, indices
    
    def save_index(self, path: str):
        """
        Save index to disk.
        
        Args:
            path: Path to save index
        """
        if self.index is None:
            raise ValueError("No index to save")
        
        # Move to CPU before saving if on GPU
        if self.metrics["using_gpu"]:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, path)
        else:
            faiss.write_index(self.index, path)
        
        self.logger.info(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """
        Load index from disk.
        
        Args:
            path: Path to load index from
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS is not available")
        
        self.index = faiss.read_index(path)
        
        # Move to GPU if requested
        if self.use_gpu and CUDA_AVAILABLE:
            self.index = self._move_to_gpu(self.index)
        
        # Set search parameters for IVF
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.ivf_nprobe
            self.metrics["using_ivf"] = True
        
        self.metrics["num_vectors"] = self.index.ntotal
        self.is_trained = True
        
        self.logger.info(f"Index loaded from {path}")
    
    def get_metrics(self) -> dict:
        """
        Get performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "num_vectors": self.metrics["num_vectors"],
            "embedding_dim": self.embedding_dim,
            "using_gpu": self.metrics["using_gpu"],
            "using_ivf": self.metrics["using_ivf"],
            "build_time": f"{self.metrics['build_time']:.2f}s",
            "search_time": f"{self.metrics['search_time']:.3f}s",
            "index_type": self._get_index_type()
        }
    
    def _get_index_type(self) -> str:
        """Get descriptive name of index type."""
        if self.index is None:
            return "None"
        
        type_str = ""
        
        if self.metrics["using_ivf"]:
            type_str += "IVF"
        else:
            type_str += "Flat"
        
        if self.metrics["using_gpu"]:
            type_str += "_GPU"
        else:
            type_str += "_CPU"
        
        return type_str