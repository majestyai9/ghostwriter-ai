"""
Comprehensive unit tests for GenerationService.

This module provides thorough testing of the GenerationService class,
including success cases, error handling, caching behavior, and RAG integration.
"""

import json
from unittest.mock import MagicMock, Mock, patch
from typing import Any, Dict

import pytest

from services.generation_service import GenerationService


class TestGenerationService:
    """Test suite for GenerationService class."""
    
    @pytest.fixture
    def mock_provider_factory(self):
        """Create a mock provider factory."""
        factory = MagicMock()
        return factory
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create a mock cache manager."""
        cache = MagicMock()
        cache.create_key = MagicMock(return_value="test_cache_key")
        cache.get = MagicMock(return_value=None)
        cache.set = MagicMock()
        return cache
    
    @pytest.fixture
    def mock_token_optimizer(self):
        """Create a mock token optimizer."""
        optimizer = MagicMock()
        optimizer.optimize_messages = MagicMock(
            return_value=[{"role": "user", "content": "optimized prompt"}]
        )
        return optimizer
    
    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        result = MagicMock()
        result.content = "Generated text response"
        provider.generate = MagicMock(return_value=result)
        provider.generate_stream = MagicMock(
            return_value=iter(["chunk1", "chunk2", "chunk3"])
        )
        return provider
    
    @pytest.fixture
    def generation_service(
        self,
        mock_provider_factory,
        mock_cache_manager,
        mock_token_optimizer
    ):
        """Create a GenerationService instance with mocked dependencies."""
        service = GenerationService(
            provider_factory=mock_provider_factory,
            cache_manager=mock_cache_manager,
            token_optimizer=mock_token_optimizer,
            enable_rag=False
        )
        return service
    
    def test_initialization_without_rag(
        self,
        mock_provider_factory,
        mock_cache_manager,
        mock_token_optimizer
    ):
        """Test service initialization without RAG enabled."""
        service = GenerationService(
            provider_factory=mock_provider_factory,
            cache_manager=mock_cache_manager,
            token_optimizer=mock_token_optimizer,
            enable_rag=False
        )
        
        assert service._provider_factory == mock_provider_factory
        assert service._cache_manager == mock_cache_manager
        assert service._token_optimizer == mock_token_optimizer
        assert service._hybrid_context_manager is None
        assert hasattr(service, "_book_context_manager")
    
    @patch("services.generation_service.settings")
    @patch("services.generation_service.RAG_AVAILABLE", True)
    @patch("services.generation_service.create_hybrid_manager")
    def test_initialization_with_rag(
        self,
        mock_create_hybrid,
        mock_settings,
        mock_provider_factory,
        mock_cache_manager,
        mock_token_optimizer
    ):
        """Test service initialization with RAG enabled."""
        # Configure settings
        mock_settings.ENABLE_RAG = True
        mock_settings.RAG_MODE = "hybrid"
        mock_settings.RAG_EMBEDDING_MODEL = "test-model"
        mock_settings.RAG_CHUNK_SIZE = 512
        mock_settings.RAG_TOP_K = 10
        mock_settings.RAG_SIMILARITY_THRESHOLD = 0.5
        mock_settings.RAG_CORE_CONTEXT_RATIO = 0.4
        mock_settings.RAG_RETRIEVED_CONTEXT_RATIO = 0.4
        mock_settings.RAG_SUMMARY_CONTEXT_RATIO = 0.2
        mock_settings.LLM_PROVIDER = "openai"
        mock_settings.OPENAI_API_KEY = "test-key"
        
        # Create mock hybrid manager
        mock_hybrid_manager = MagicMock()
        mock_create_hybrid.return_value = mock_hybrid_manager
        
        service = GenerationService(
            provider_factory=mock_provider_factory,
            cache_manager=mock_cache_manager,
            token_optimizer=mock_token_optimizer,
            enable_rag=True
        )
        
        assert service._hybrid_context_manager is not None
        mock_create_hybrid.assert_called_once()
    
    def test_get_provider(self, generation_service, mock_provider_factory, mock_provider):
        """Test provider retrieval from factory."""
        mock_provider_factory.create_provider.return_value = mock_provider
        
        with patch("services.generation_service.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-openai-key"
            mock_settings.ANTHROPIC_API_KEY = "test-anthropic-key"
            
            provider = generation_service._get_provider("openai")
            
            mock_provider_factory.create_provider.assert_called_once_with(
                "openai",
                {
                    "provider": "openai",
                    "api_key": "test-openai-key"
                }
            )
            assert provider == mock_provider
    
    def test_generate_text_without_cache(
        self,
        generation_service,
        mock_provider_factory,
        mock_provider,
        mock_cache_manager,
        mock_token_optimizer
    ):
        """Test text generation without using cache."""
        mock_provider_factory.create_provider.return_value = mock_provider
        
        with patch("services.generation_service.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            
            result = generation_service.generate_text(
                provider_name="openai",
                prompt="Test prompt",
                use_cache=False,
                temperature=0.7
            )
            
            # Verify optimization
            mock_token_optimizer.optimize_messages.assert_called_once_with(
                [{"role": "user", "content": "Test prompt"}],
                4096
            )
            
            # Verify generation
            mock_provider.generate.assert_called_once_with(
                "optimized prompt",
                temperature=0.7
            )
            
            # Verify cache was not used
            mock_cache_manager.get.assert_not_called()
            mock_cache_manager.set.assert_not_called()
            
            assert result == "Generated text response"
    
    def test_generate_text_with_cache_miss(
        self,
        generation_service,
        mock_provider_factory,
        mock_provider,
        mock_cache_manager,
        mock_token_optimizer
    ):
        """Test text generation with cache miss."""
        mock_provider_factory.create_provider.return_value = mock_provider
        mock_cache_manager.get.return_value = None  # Cache miss
        
        with patch("services.generation_service.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            
            result = generation_service.generate_text(
                provider_name="openai",
                prompt="Test prompt",
                use_cache=True
            )
            
            # Verify cache check
            mock_cache_manager.create_key.assert_called_once()
            mock_cache_manager.get.assert_called_once_with("test_cache_key")
            
            # Verify generation after cache miss
            mock_provider.generate.assert_called_once()
            
            # Verify result was cached
            mock_cache_manager.set.assert_called_once_with(
                "test_cache_key",
                "Generated text response"
            )
            
            assert result == "Generated text response"
    
    def test_generate_text_with_cache_hit(
        self,
        generation_service,
        mock_cache_manager,
        mock_provider
    ):
        """Test text generation with cache hit."""
        cached_response = "Cached response"
        mock_cache_manager.get.return_value = cached_response
        
        result = generation_service.generate_text(
            provider_name="openai",
            prompt="Test prompt",
            use_cache=True
        )
        
        # Verify cache was used
        mock_cache_manager.get.assert_called_once()
        
        # Verify no generation occurred
        mock_provider.generate.assert_not_called()
        
        assert result == cached_response
    
    def test_generate_text_stream(
        self,
        generation_service,
        mock_provider_factory,
        mock_provider,
        mock_token_optimizer
    ):
        """Test streaming text generation."""
        mock_provider_factory.create_provider.return_value = mock_provider
        
        with patch("services.generation_service.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            
            result = list(generation_service.generate_text_stream(
                provider_name="openai",
                prompt="Test prompt",
                temperature=0.5
            ))
            
            # Verify optimization
            mock_token_optimizer.optimize_messages.assert_called_once()
            
            # Verify streaming generation
            mock_provider.generate_stream.assert_called_once_with(
                "optimized prompt",
                temperature=0.5
            )
            
            assert result == ["chunk1", "chunk2", "chunk3"]
    
    def test_generate_book_chapter_without_rag(
        self,
        generation_service,
        mock_provider_factory,
        mock_provider
    ):
        """Test book chapter generation without RAG context."""
        mock_provider_factory.create_provider.return_value = mock_provider
        
        # Disable RAG
        generation_service._hybrid_context_manager = None
        
        book_data = {
            "title": "Test Book",
            "toc": {
                "chapters": [
                    {"title": "Chapter 1", "topics": "Introduction"},
                    {"title": "Chapter 2", "topics": "Main Content"}
                ]
            }
        }
        
        with patch("services.generation_service.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            
            result = generation_service.generate_book_chapter(
                provider_name="openai",
                book=book_data,
                chapter_number=0
            )
            
            # Verify context preparation
            generation_service._book_context_manager.prepare_context.assert_called_once_with(
                book_data,
                current_chapter=0
            )
            
            # Verify generation with history
            mock_provider.generate.assert_called_once_with(
                "",
                history=mock_context
            )
            
            assert result == "Generated text response"
    
    @patch("services.generation_service.HybridContextManager")
    def test_generate_book_chapter_with_hybrid_context(
        self,
        mock_hybrid_class,
        generation_service,
        mock_provider_factory,
        mock_provider
    ):
        """Test book chapter generation with hybrid RAG context."""
        mock_provider_factory.create_provider.return_value = mock_provider
        
        # Setup hybrid context manager
        mock_hybrid_manager = MagicMock()
        mock_context = [
            {"role": "system", "content": "Enhanced context"},
            {"role": "user", "content": "Write chapter with RAG"}
        ]
        mock_hybrid_manager.prepare_context.return_value = mock_context
        generation_service._hybrid_context_manager = mock_hybrid_manager
        
        book_data = {
            "title": "Test Book",
            "toc": {
                "chapters": [
                    {"title": "Chapter 1", "topics": "Introduction"},
                    {"title": "Chapter 2", "topics": "Main Content"}
                ]
            }
        }
        
        with patch("services.generation_service.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            
            result = generation_service.generate_book_chapter(
                provider_name="openai",
                book=book_data,
                chapter_number=1,
                book_dir="/path/to/book"
            )
            
            # Verify hybrid context preparation
            mock_hybrid_manager.prepare_context.assert_called_once_with(
                book=book_data,
                current_chapter=1,
                book_dir="/path/to/book",
                query="Chapter 2 Main Content"
            )
            
            # Verify generation
            mock_provider.generate.assert_called_once_with(
                "",
                history=mock_context
            )
            
            assert result == "Generated text response"
    
    def test_generate_text_error_handling(
        self,
        generation_service,
        mock_provider_factory,
        mock_provider
    ):
        """Test error handling in text generation."""
        mock_provider_factory.create_provider.return_value = mock_provider
        mock_provider.generate.side_effect = Exception("API Error")
        
        with patch("services.generation_service.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            
            with pytest.raises(Exception) as exc_info:
                generation_service.generate_text(
                    provider_name="openai",
                    prompt="Test prompt"
                )
            
            assert "API Error" in str(exc_info.value)
    
    def test_provider_kwargs_forwarding(
        self,
        generation_service,
        mock_provider_factory,
        mock_provider,
        mock_token_optimizer
    ):
        """Test that additional kwargs are properly forwarded to provider."""
        mock_provider_factory.create_provider.return_value = mock_provider
        
        with patch("services.generation_service.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            
            generation_service.generate_text(
                provider_name="openai",
                prompt="Test prompt",
                use_cache=False,
                temperature=0.9,
                max_tokens=2048,
                top_p=0.95,
                frequency_penalty=0.5
            )
            
            # Verify all kwargs were forwarded
            mock_provider.generate.assert_called_once_with(
                "optimized prompt",
                temperature=0.9,
                max_tokens=2048,
                top_p=0.95,
                frequency_penalty=0.5
            )


class TestGenerationServiceIntegration:
    """Integration tests for GenerationService with real-like scenarios."""
    
    @pytest.mark.integration
    def test_full_generation_flow_with_caching(self):
        """Test complete generation flow with caching behavior."""
        # Create real-like mocks
        provider_factory = MagicMock()
        cache_manager = MagicMock()
        token_optimizer = MagicMock()
        
        # Configure cache behavior
        cache_manager.create_key = lambda provider, **kwargs: f"{provider}_{kwargs.get('prompt', '')[:10]}"
        cache_manager.get = MagicMock(side_effect=[None, "Cached result"])
        cache_manager.set = MagicMock()
        
        # Configure token optimizer
        token_optimizer.optimize_messages = lambda msgs, _: msgs
        
        # Configure provider
        provider = MagicMock()
        result = MagicMock()
        result.content = "Fresh generation"
        provider.generate = MagicMock(return_value=result)
        provider_factory.create_provider = MagicMock(return_value=provider)
        
        service = GenerationService(
            provider_factory=provider_factory,
            cache_manager=cache_manager,
            token_optimizer=token_optimizer,
            enable_rag=False
        )
        
        with patch("services.generation_service.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            
            # First call - cache miss
            result1 = service.generate_text(
                "openai",
                "Test prompt for caching",
                use_cache=True
            )
            assert result1 == "Fresh generation"
            cache_manager.set.assert_called_once()
            
            # Second call - cache hit
            result2 = service.generate_text(
                "openai",
                "Test prompt for caching",
                use_cache=True
            )
            assert result2 == "Cached result"
            assert provider.generate.call_count == 1  # No new generation
    
    @pytest.mark.integration
    def test_provider_switching(self):
        """Test switching between different providers."""
        provider_factory = MagicMock()
        cache_manager = MagicMock()
        token_optimizer = MagicMock()
        
        # Create different providers
        openai_provider = MagicMock()
        openai_result = MagicMock()
        openai_result.content = "OpenAI response"
        openai_provider.generate = MagicMock(return_value=openai_result)
        
        anthropic_provider = MagicMock()
        anthropic_result = MagicMock()
        anthropic_result.content = "Anthropic response"
        anthropic_provider.generate = MagicMock(return_value=anthropic_result)
        
        def create_provider(name, config):
            if name == "openai":
                return openai_provider
            elif name == "anthropic":
                return anthropic_provider
            raise ValueError(f"Unknown provider: {name}")
        
        provider_factory.create_provider = create_provider
        token_optimizer.optimize_messages = lambda msgs, _: msgs
        
        service = GenerationService(
            provider_factory=provider_factory,
            cache_manager=cache_manager,
            token_optimizer=token_optimizer,
            enable_rag=False
        )
        
        with patch("services.generation_service.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "openai-key"
            mock_settings.ANTHROPIC_API_KEY = "anthropic-key"
            
            # Test OpenAI provider
            result1 = service.generate_text(
                "openai",
                "Test prompt",
                use_cache=False
            )
            assert result1 == "OpenAI response"
            
            # Test Anthropic provider
            result2 = service.generate_text(
                "anthropic",
                "Test prompt",
                use_cache=False
            )
            assert result2 == "Anthropic response"