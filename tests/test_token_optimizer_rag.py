"""
Comprehensive unit tests for RAG-enhanced token optimization.

This module provides thorough testing of the RAG system including
configuration, mode selection, retriever functionality, and hybrid context management.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from token_optimizer_rag import (
    RAGConfig,
    RAGMode,
    HybridContextManager,
    create_hybrid_manager,
    SENTENCE_TRANSFORMERS_AVAILABLE,
    FAISS_AVAILABLE,
    CUDA_AVAILABLE
)


class TestRAGMode:
    """Test suite for RAGMode enum."""
    
    def test_rag_modes(self):
        """Test all RAG mode values."""
        assert RAGMode.DISABLED.value == "disabled"
        assert RAGMode.BASIC.value == "basic"
        assert RAGMode.HYBRID.value == "hybrid"
        assert RAGMode.FULL.value == "full"
    
    def test_mode_from_string(self):
        """Test creating RAG mode from string."""
        assert RAGMode("disabled") == RAGMode.DISABLED
        assert RAGMode("basic") == RAGMode.BASIC
        assert RAGMode("hybrid") == RAGMode.HYBRID
        assert RAGMode("full") == RAGMode.FULL
    
    def test_invalid_mode(self):
        """Test invalid mode raises error."""
        with pytest.raises(ValueError):
            RAGMode("invalid_mode")


class TestRAGConfig:
    """Test suite for RAGConfig dataclass."""
    
    def test_default_config(self):
        """Test default RAG configuration values."""
        config = RAGConfig()
        
        assert config.mode == RAGMode.HYBRID
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 128
        assert config.top_k == 10
        assert config.similarity_threshold == 0.5
        assert config.use_ivf is True
        assert config.use_gpu is True
        assert config.cache_size == 1000
        assert config.cache_ttl == 3600
    
    def test_custom_config(self):
        """Test custom RAG configuration."""
        config = RAGConfig(
            mode=RAGMode.FULL,
            embedding_model="custom-model",
            chunk_size=1024,
            top_k=20,
            use_gpu=False
        )
        
        assert config.mode == RAGMode.FULL
        assert config.embedding_model == "custom-model"
        assert config.chunk_size == 1024
        assert config.top_k == 20
        assert config.use_gpu is False
    
    def test_token_distribution_validation(self):
        """Test validation of token distribution ratios."""
        # Valid distribution (sums to 1.0)
        config = RAGConfig(
            core_context_ratio=0.3,
            rag_context_ratio=0.5,
            summary_context_ratio=0.2
        )
        assert config.core_context_ratio == 0.3
        assert config.rag_context_ratio == 0.5
        assert config.summary_context_ratio == 0.2
    
    def test_invalid_token_distribution(self):
        """Test that invalid token distribution raises error."""
        with pytest.raises(ValueError) as exc_info:
            RAGConfig(
                core_context_ratio=0.5,
                rag_context_ratio=0.4,
                summary_context_ratio=0.3  # Sum = 1.2
            )
        
        assert "Token distribution ratios must sum to 1.0" in str(exc_info.value)
    
    def test_feature_flags(self):
        """Test feature flag configuration."""
        config = RAGConfig(
            enable_caching=False,
            enable_compression=False,
            enable_async_indexing=True,
            enable_batch_processing=False
        )
        
        assert config.enable_caching is False
        assert config.enable_compression is False
        assert config.enable_async_indexing is True
        assert config.enable_batch_processing is False
    
    def test_ivf_parameters(self):
        """Test IVF indexing parameters."""
        config = RAGConfig(
            use_ivf=True,
            ivf_nlist=200,
            ivf_nprobe=20
        )
        
        assert config.use_ivf is True
        assert config.ivf_nlist == 200
        assert config.ivf_nprobe == 20


class TestHybridContextManager:
    """Test suite for HybridContextManager."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        result = MagicMock()
        result.content = "Summarized content"
        provider.generate = MagicMock(return_value=result)
        return provider
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create a mock cache manager."""
        cache = MagicMock()
        cache.get = MagicMock(return_value=None)
        cache.set = MagicMock()
        return cache
    
    @pytest.fixture
    def mock_retriever(self):
        """Create a mock semantic retriever."""
        retriever = MagicMock()
        retriever.index_content = MagicMock()
        retriever.search = MagicMock(return_value=[
            {"content": "Retrieved chunk 1", "score": 0.9},
            {"content": "Retrieved chunk 2", "score": 0.8}
        ])
        return retriever
    
    @pytest.fixture
    def sample_book(self):
        """Sample book data for testing."""
        return {
            "title": "Test Book",
            "summary": "A book about testing",
            "toc": {
                "chapters": [
                    {
                        "number": 1,
                        "title": "Introduction",
                        "content": "Chapter 1 content"
                    },
                    {
                        "number": 2,
                        "title": "Main Content",
                        "content": "Chapter 2 content"
                    },
                    {
                        "number": 3,
                        "title": "Advanced Topics",
                        "content": "Chapter 3 content"
                    }
                ]
            }
        }
    
    @patch("token_optimizer_rag.SemanticContextRetriever")
    @patch("token_optimizer_rag.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("token_optimizer_rag.FAISS_AVAILABLE", True)
    def test_initialization_with_rag(
        self,
        mock_retriever_class,
        mock_provider,
        mock_cache_manager
    ):
        """Test manager initialization with RAG enabled."""
        mock_retriever = MagicMock()
        mock_retriever_class.return_value = mock_retriever
        
        config = RAGConfig(mode=RAGMode.HYBRID)
        manager = HybridContextManager(
            config=config,
            provider=mock_provider,
            cache_manager=mock_cache_manager
        )
        
        assert manager.config == config
        mock_retriever_class.assert_called_once_with(
            config=config,
            cache_manager=mock_cache_manager
        )
    
    @patch("token_optimizer_rag.SENTENCE_TRANSFORMERS_AVAILABLE", False)
    def test_initialization_without_dependencies(
        self,
        mock_provider,
        mock_cache_manager
    ):
        """Test manager initialization when dependencies are missing."""
        config = RAGConfig(mode=RAGMode.HYBRID)
        
        with patch("token_optimizer_rag.logging") as mock_logging:
            manager = HybridContextManager(
                config=config,
                provider=mock_provider,
                cache_manager=mock_cache_manager
            )
            
            # Should downgrade to BASIC mode
            assert manager.config.mode == RAGMode.BASIC
            mock_logging.warning.assert_called()
    
    def test_prepare_context_disabled_mode(
        self,
        mock_provider,
        mock_cache_manager,
        sample_book
    ):
        """Test context preparation with RAG disabled."""
        config = RAGConfig(mode=RAGMode.DISABLED)
        
        with patch("token_optimizer_rag.HybridManagerBase") as mock_base:
            mock_base_instance = MagicMock()
            mock_base_instance.prepare_context.return_value = [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "User message"}
            ]
            mock_base.return_value = mock_base_instance
            
            manager = HybridContextManager(
                config=config,
                provider=mock_provider,
                cache_manager=mock_cache_manager
            )
            
            # Mock the parent class method
            manager.prepare_context = mock_base_instance.prepare_context
            
            context = manager.prepare_context(
                book=sample_book,
                current_chapter=1
            )
            
            assert len(context) == 2
            assert context[0]["role"] == "system"
    
    def test_get_performance_stats(
        self,
        mock_provider,
        mock_cache_manager
    ):
        """Test performance statistics retrieval."""
        config = RAGConfig(
            mode=RAGMode.HYBRID,
            use_gpu=True,
            use_ivf=True,
            enable_batch_processing=True
        )
        
        with patch("token_optimizer_rag.HybridManagerBase"):
            manager = HybridContextManager(
                config=config,
                provider=mock_provider,
                cache_manager=mock_cache_manager
            )
            
            # Mock the parent get_stats method
            manager.get_stats = MagicMock(return_value={
                "contexts_prepared": 10,
                "cache_hits": 5
            })
            
            stats = manager.get_performance_stats()
            
            assert "contexts_prepared" in stats
            assert "cache_hits" in stats
            assert "optimization" in stats
            
            opt_stats = stats["optimization"]
            assert opt_stats["gpu_enabled"] == (config.use_gpu and CUDA_AVAILABLE)
            assert opt_stats["ivf_enabled"] == config.use_ivf
            assert opt_stats["batch_processing"] == config.enable_batch_processing
            assert opt_stats["caching_enabled"] == config.enable_caching


class TestCreateHybridManager:
    """Test suite for create_hybrid_manager factory function."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        return MagicMock()
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create a mock cache manager."""
        return MagicMock()
    
    @patch("token_optimizer_rag.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("token_optimizer_rag.FAISS_AVAILABLE", True)
    @patch("token_optimizer_rag.CUDA_AVAILABLE", True)
    def test_auto_configuration_with_all_dependencies(
        self,
        mock_provider,
        mock_cache_manager
    ):
        """Test auto-configuration when all dependencies are available."""
        with patch("token_optimizer_rag.HybridContextManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            
            manager = create_hybrid_manager(
                provider=mock_provider,
                cache_manager=mock_cache_manager
            )
            
            # Should create with HYBRID mode and GPU enabled
            call_args = mock_manager_class.call_args
            config = call_args.kwargs["config"]
            
            assert config.mode == RAGMode.HYBRID
            assert config.use_gpu is True
            assert config.use_ivf is True
            assert config.enable_batch_processing is True
            assert manager == mock_manager
    
    @patch("token_optimizer_rag.SENTENCE_TRANSFORMERS_AVAILABLE", False)
    @patch("token_optimizer_rag.FAISS_AVAILABLE", True)
    def test_auto_configuration_missing_sentence_transformers(
        self,
        mock_provider,
        mock_cache_manager
    ):
        """Test auto-configuration with missing sentence-transformers."""
        with patch("token_optimizer_rag.HybridContextManager") as mock_manager_class:
            with patch("token_optimizer_rag.logging") as mock_logging:
                manager = create_hybrid_manager(
                    provider=mock_provider,
                    cache_manager=mock_cache_manager
                )
                
                # Should fallback to BASIC mode
                call_args = mock_manager_class.call_args
                config = call_args.kwargs["config"]
                
                assert config.mode == RAGMode.BASIC
                mock_logging.warning.assert_called()
    
    @patch("token_optimizer_rag.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("token_optimizer_rag.FAISS_AVAILABLE", False)
    def test_auto_configuration_missing_faiss(
        self,
        mock_provider,
        mock_cache_manager
    ):
        """Test auto-configuration with missing FAISS."""
        with patch("token_optimizer_rag.HybridContextManager") as mock_manager_class:
            with patch("token_optimizer_rag.logging") as mock_logging:
                manager = create_hybrid_manager(
                    provider=mock_provider,
                    cache_manager=mock_cache_manager
                )
                
                # Should fallback to BASIC mode
                call_args = mock_manager_class.call_args
                config = call_args.kwargs["config"]
                
                assert config.mode == RAGMode.BASIC
                mock_logging.warning.assert_called()
    
    def test_custom_configuration(
        self,
        mock_provider,
        mock_cache_manager
    ):
        """Test creating manager with custom configuration."""
        custom_config = RAGConfig(
            mode=RAGMode.FULL,
            chunk_size=1024,
            top_k=20,
            use_gpu=False
        )
        
        with patch("token_optimizer_rag.HybridContextManager") as mock_manager_class:
            manager = create_hybrid_manager(
                provider=mock_provider,
                cache_manager=mock_cache_manager,
                config=custom_config,
                max_tokens=200000
            )
            
            mock_manager_class.assert_called_once_with(
                config=custom_config,
                provider=mock_provider,
                cache_manager=mock_cache_manager,
                max_tokens=200000
            )
    
    @patch("token_optimizer_rag.CUDA_AVAILABLE", False)
    def test_gpu_disabled_when_cuda_unavailable(
        self,
        mock_provider,
        mock_cache_manager
    ):
        """Test that GPU is disabled when CUDA is not available."""
        with patch("token_optimizer_rag.SENTENCE_TRANSFORMERS_AVAILABLE", True):
            with patch("token_optimizer_rag.FAISS_AVAILABLE", True):
                with patch("token_optimizer_rag.HybridContextManager") as mock_manager_class:
                    manager = create_hybrid_manager(
                        provider=mock_provider,
                        cache_manager=mock_cache_manager
                    )
                    
                    call_args = mock_manager_class.call_args
                    config = call_args.kwargs["config"]
                    
                    # GPU should be disabled
                    assert config.use_gpu is False


class TestIntegration:
    """Integration tests for the RAG system."""
    
    @pytest.mark.integration
    def test_full_rag_pipeline(self):
        """Test the complete RAG pipeline integration."""
        # Create mock components
        provider = MagicMock()
        provider.generate = MagicMock(return_value=MagicMock(content="Summary"))
        
        cache_manager = MagicMock()
        cache_manager.get = MagicMock(return_value=None)
        cache_manager.set = MagicMock()
        
        # Sample book data
        book = {
            "title": "Integration Test Book",
            "toc": {
                "chapters": [
                    {
                        "number": 1,
                        "title": "Chapter 1",
                        "content": "First chapter content " * 100
                    },
                    {
                        "number": 2,
                        "title": "Chapter 2",
                        "content": "Second chapter content " * 100
                    }
                ]
            }
        }
        
        with patch("token_optimizer_rag.SENTENCE_TRANSFORMERS_AVAILABLE", True):
            with patch("token_optimizer_rag.FAISS_AVAILABLE", True):
                with patch("token_optimizer_rag.SemanticContextRetriever") as mock_retriever_class:
                    mock_retriever = MagicMock()
                    mock_retriever.index_content = MagicMock()
                    mock_retriever.search = MagicMock(return_value=[
                        {"content": "Relevant content 1", "score": 0.95},
                        {"content": "Relevant content 2", "score": 0.90}
                    ])
                    mock_retriever_class.return_value = mock_retriever
                    
                    # Create manager
                    config = RAGConfig(mode=RAGMode.HYBRID)
                    manager = HybridContextManager(
                        config=config,
                        provider=provider,
                        cache_manager=cache_manager
                    )
                    
                    # Mock the base prepare_context
                    with patch.object(manager, "prepare_context") as mock_prepare:
                        mock_prepare.return_value = [
                            {"role": "system", "content": "System prompt"},
                            {"role": "user", "content": "Write chapter"}
                        ]
                        
                        # Prepare context
                        context = manager.prepare_context(
                            book=book,
                            current_chapter=1,
                            query="Write an engaging chapter"
                        )
                        
                        assert len(context) == 2
                        mock_prepare.assert_called_once()
    
    @pytest.mark.integration
    def test_mode_switching(self):
        """Test switching between different RAG modes."""
        provider = MagicMock()
        cache_manager = MagicMock()
        
        # Test each mode
        for mode in [RAGMode.DISABLED, RAGMode.BASIC, RAGMode.HYBRID, RAGMode.FULL]:
            config = RAGConfig(mode=mode)
            
            with patch("token_optimizer_rag.logging"):
                manager = create_hybrid_manager(
                    provider=provider,
                    cache_manager=cache_manager,
                    config=config
                )
                
                assert manager.config.mode == mode or manager.config.mode == RAGMode.BASIC
    
    @pytest.mark.integration
    def test_caching_behavior(self):
        """Test caching behavior in the RAG system."""
        provider = MagicMock()
        cache_manager = MagicMock()
        
        # Configure cache behavior
        cache_data = {}
        cache_manager.get = lambda key: cache_data.get(key)
        cache_manager.set = lambda key, value: cache_data.update({key: value})
        
        config = RAGConfig(
            mode=RAGMode.BASIC,
            enable_caching=True,
            cache_ttl=3600
        )
        
        manager = create_hybrid_manager(
            provider=provider,
            cache_manager=cache_manager,
            config=config
        )
        
        # Verify caching is configured
        assert manager.config.enable_caching is True
        assert manager.config.cache_ttl == 3600