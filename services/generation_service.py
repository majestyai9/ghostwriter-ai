"""
Unified service for orchestrating AI content generation.
"""

import logging
from collections.abc import Generator
from typing import Any, Dict, Optional

from app_config import settings
from cache_manager import CacheManager
from providers.base import LLMProvider
from providers.factory import ProviderFactory

# Import legacy and new token optimizers
from token_optimizer import BookContextManager, TokenOptimizer

# Try to import RAG-enhanced optimizer
try:
    from token_optimizer_rag import HybridContextManager, RAGConfig, RAGMode, create_hybrid_manager
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logging.warning("RAG-enhanced optimizer not available, using legacy system")

class GenerationService:
    """
    A unified service for orchestrating AI content generation with optional RAG support.
    """
    def __init__(
        self,
        provider_factory: ProviderFactory,
        cache_manager: CacheManager,
        token_optimizer: TokenOptimizer,
        enable_rag: bool = True,
        rag_config: Optional[Dict[str, Any]] = None,
    ):
        self._provider_factory = provider_factory
        self._cache_manager = cache_manager
        self._token_optimizer = token_optimizer
        self.logger = logging.getLogger(__name__)

        # Initialize context manager based on RAG availability and settings
        self._hybrid_context_manager = None
        if enable_rag and RAG_AVAILABLE and settings.ENABLE_RAG:
            try:
                # Create RAG configuration from settings
                if rag_config is None:
                    rag_config = RAGConfig(
                        mode=RAGMode(settings.RAG_MODE.lower()),
                        embedding_model=settings.RAG_EMBEDDING_MODEL,
                        chunk_size=settings.RAG_CHUNK_SIZE,
                        top_k=settings.RAG_TOP_K,
                        similarity_threshold=settings.RAG_SIMILARITY_THRESHOLD,
                        core_context_ratio=settings.RAG_CORE_CONTEXT_RATIO,
                        rag_context_ratio=settings.RAG_RETRIEVED_CONTEXT_RATIO,
                        summary_context_ratio=settings.RAG_SUMMARY_CONTEXT_RATIO,
                    )

                # Get a provider for summarization
                provider = self._get_provider(settings.LLM_PROVIDER)

                # Create hybrid context manager
                self._hybrid_context_manager = create_hybrid_manager(
                    provider=provider,
                    cache_manager=cache_manager,
                    config=rag_config
                )
                self.logger.info(f"RAG-enhanced context manager initialized in {rag_config.mode.value} mode")
            except Exception as e:
                self.logger.warning(f"Failed to initialize RAG context manager: {e}, falling back to legacy")
                self._hybrid_context_manager = None

        # Fallback to legacy context manager
        if self._hybrid_context_manager is None:
            self._book_context_manager = BookContextManager()
            self.logger.info("Using legacy context manager")

    def _get_provider(self, provider_name: str) -> LLMProvider:
        """Retrieves a provider instance from the factory."""
        provider_config = {
            "provider": provider_name,
            "api_key": settings.OPENAI_API_KEY if provider_name == "openai" else settings.ANTHROPIC_API_KEY,
        }
        return self._provider_factory.create_provider(provider_name, provider_config)

    def generate_text(
        self,
        provider_name: str,
        prompt: str,
        use_cache: bool = True,
        **provider_kwargs: Any,
    ) -> str:
        """
        Generates text using a specified provider, with caching and optimization.

        Args:
            provider_name: The name of the LLM provider to use (e.g., 'openai').
            prompt: The input prompt for the LLM.
            use_cache: Whether to use the cache for this request.
            **provider_kwargs: Additional keyword arguments for the provider.

        Returns:
            The generated text as a string.
        """
        if use_cache:
            cache_key = self._cache_manager.create_key(provider_name, prompt=prompt, **provider_kwargs)
            cached_result = self._cache_manager.get(cache_key)
            if cached_result:
                return cached_result

        optimized_prompt = self._token_optimizer.optimize_messages([{"role": "user", "content": prompt}], 4096)
        provider = self._get_provider(provider_name)

        result = provider.generate(optimized_prompt[0]["content"], **provider_kwargs)

        if use_cache:
            self._cache_manager.set(cache_key, result.content)

        return result.content

    def generate_text_stream(
        self,
        provider_name: str,
        prompt: str,
        **provider_kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        Generates text as a stream from a specified provider.
        Note: Caching is typically not used for streaming responses.
        """
        optimized_prompt = self._token_optimizer.optimize_messages([{"role": "user", "content": prompt}], 4096)
        provider = self._get_provider(provider_name)

        yield from provider.generate_stream(optimized_prompt[0]["content"], **provider_kwargs)

    def generate_book_chapter(
        self,
        provider_name: str,
        book: Dict[str, Any],
        chapter_number: int,
        book_dir: Optional[str] = None,
        **provider_kwargs: Any,
    ) -> str:
        """
        Generates a book chapter with optimized context using hybrid RAG if available.

        Args:
            provider_name: The name of the LLM provider to use.
            book: The book data.
            chapter_number: The number of the chapter to generate.
            book_dir: Optional book directory for RAG vector store.
            **provider_kwargs: Additional keyword arguments for the provider.

        Returns:
            The generated chapter content.
        """
        # Prepare context using hybrid manager if available
        if self._hybrid_context_manager:
            # Get chapter info for query
            chapters = book.get("toc", {}).get("chapters", [])
            query = None
            if 0 <= chapter_number < len(chapters):
                chapter = chapters[chapter_number]
                query = f"{chapter.get('title', '')} {chapter.get('topics', '')}"

            context = self._hybrid_context_manager.prepare_context(
                book=book,
                current_chapter=chapter_number,
                book_dir=book_dir,
                query=query
            )
            self.logger.debug(f"Using hybrid context with {len(context)} messages")
        else:
            # Fallback to legacy context manager
            context = self._book_context_manager.prepare_context(
                book,
                current_chapter=chapter_number
            )
            self.logger.debug(f"Using legacy context with {len(context)} messages")

        provider = self._get_provider(provider_name)

        # The prompt is now the conversation history
        result = provider.generate("", history=context, **provider_kwargs)

        return result.content
