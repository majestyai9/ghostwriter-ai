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

# Import token optimizers
from token_optimizer import TokenOptimizer
from token_optimizer_rag import HybridContextManager, RAGConfig, RAGMode, create_hybrid_manager

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

        # Initialize RAG context manager
        if enable_rag and settings.ENABLE_RAG:
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
        else:
            self._hybrid_context_manager = None
            self.logger.info("RAG system disabled")

    def _get_provider(self, provider_name: str) -> LLMProvider:
        """Retrieves a provider instance from the factory."""
        # Get the appropriate API key based on provider
        if provider_name == "openai":
            api_key = settings.OPENAI_API_KEY
        elif provider_name == "anthropic":
            api_key = settings.ANTHROPIC_API_KEY
        elif provider_name == "gemini":
            api_key = settings.GEMINI_API_KEY
        elif provider_name == "cohere":
            api_key = settings.COHERE_API_KEY
        elif provider_name == "openrouter":
            api_key = settings.OPENROUTER_API_KEY
        else:
            api_key = None
            
        provider_config = {
            "provider": provider_name,
            "api_key": api_key,
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
        continuity_context: Optional[str] = None,
        quality_requirements: Optional[str] = None,
        originality_requirements: Optional[str] = None,
        **provider_kwargs: Any,
    ) -> str:
        """
        Generates a book chapter with optimized context using hybrid RAG if available.

        Args:
            provider_name: The name of the LLM provider to use.
            book: The book data.
            chapter_number: The number of the chapter to generate.
            book_dir: Optional book directory for RAG vector store.
            continuity_context: Optional narrative continuity context.
            quality_requirements: Optional quality enforcement requirements.
            originality_requirements: Optional plot originality requirements.
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
            # RAG is disabled, prepare minimal context
            context = [
                {"role": "system", "content": "You are a professional writer creating a book."},
            ]
            self.logger.debug("Using minimal context without RAG")

        # Add quality enhancement contexts
        enhancement_prompts = []
        
        if continuity_context:
            enhancement_prompts.append(f"NARRATIVE CONTINUITY:\n{continuity_context}")
        
        if quality_requirements:
            enhancement_prompts.append(f"QUALITY STANDARDS:\n{quality_requirements}")
        
        if originality_requirements:
            enhancement_prompts.append(f"ORIGINALITY GUIDELINES:\n{originality_requirements}")
        
        # Add enhancement prompts to context
        if enhancement_prompts:
            enhancement_message = "\n\n".join(enhancement_prompts)
            context.append({
                "role": "system",
                "content": f"IMPORTANT REQUIREMENTS FOR THIS CHAPTER:\n\n{enhancement_message}"
            })
        
        # Add the chapter generation prompt
        if 0 <= chapter_number < len(book.get("toc", {}).get("chapters", [])):
            chapter = book["toc"]["chapters"][chapter_number]
            prompt = f"Generate Chapter {chapter_number + 1}: {chapter.get('title', 'Untitled')}"
            if 'topics' in chapter:
                prompt += f"\n\nTopics to cover: {chapter['topics']}"
            context.append({"role": "user", "content": prompt})

        provider = self._get_provider(provider_name)

        # The prompt is now the conversation history
        result = provider.generate("", history=context, **provider_kwargs)

        return result.content
