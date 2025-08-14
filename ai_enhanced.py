"""
Enhanced AI module with streaming, caching, and optimization
"""
import logging
from collections.abc import Generator
from typing import Any, Dict, List

import config
from cache_manager import get_cache
from events import Event, EventType, event_manager
from providers import LLMProvider, get_provider
from streaming import streaming_manager
from token_optimizer import BookContextManager, get_optimizer

# Initialize components
provider: LLMProvider = None
cache = None
optimizer = None

def initialize_enhanced():
    """Initialize enhanced AI components"""
    global provider, cache, optimizer

    try:
        # Initialize provider
        provider = get_provider(config=config.PROVIDER_CONFIG)
        logging.info(f"Initialized {config.LLM_PROVIDER} provider")

        # Initialize cache
        cache_backend = config.__dict__.get('CACHE_BACKEND', 'memory')
        cache = get_cache()

        # Initialize optimizer
        optimizer = get_optimizer()
        optimizer.provider = provider

        logging.info("Enhanced AI components initialized")

    except Exception as e:
        logging.error(f"Failed to initialize enhanced AI: {e}")
        raise

# Initialize on module load
initialize_enhanced()

def generate_with_cache(prompt: str,
                        history: List[Dict[str, str]] = None,
                        max_tokens: int = 1024,
                        temperature: float = 0.7,
                        cache_key: str = None,
                        cache_expire: int = 3600,
                        **kwargs) -> str:
    """
    Generate text with caching support
    
    Args:
        prompt: Input prompt
        history: Conversation history
        max_tokens: Maximum tokens
        temperature: Sampling temperature
        cache_key: Custom cache key
        cache_expire: Cache expiration in seconds
        
    Returns:
        Generated text
    """
    # Create cache key if not provided
    if not cache_key:
        cache_key = cache.create_key(
            'generate',
            prompt=prompt[:100],  # First 100 chars
            max_tokens=max_tokens,
            temperature=temperature
        )

    # Try to get from cache
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        logging.info(f"Cache hit for generation: {cache_key}")
        event_manager.emit(Event(EventType.API_CALL_COMPLETED, {
            'from_cache': True,
            'cache_key': cache_key
        }))
        return cached_result

    # Generate new content
    logging.info(f"Cache miss, generating new content: {cache_key}")

    # Optimize context if needed
    if history:
        history = optimizer.optimize_messages(history, max_tokens)

    # Generate
    response = provider.generate(
        prompt=prompt,
        history=history,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )

    # Cache result
    cache.set(cache_key, response.content, cache_expire)

    return response.content

def generate_stream(prompt: str,
                   history: List[Dict[str, str]] = None,
                   max_tokens: int = 1024,
                   temperature: float = 0.7,
                   stream_id: str = None,
                   **kwargs) -> Generator[str, None, None]:
    """
    Generate text with streaming
    
    Args:
        prompt: Input prompt
        history: Conversation history
        max_tokens: Maximum tokens
        temperature: Sampling temperature
        stream_id: Optional stream ID for tracking
        
    Yields:
        Text chunks as generated
    """
    import uuid

    if not stream_id:
        stream_id = str(uuid.uuid4())

    # Optimize context if needed
    if history:
        history = optimizer.optimize_messages(history, max_tokens)

    # Check if provider supports streaming
    if hasattr(provider, 'generate_stream'):
        logging.info(f"Starting stream generation: {stream_id}")

        # Get raw stream from provider
        raw_stream = provider.generate_stream(
            prompt=prompt,
            history=history,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        # Wrap with streaming manager
        for chunk in streaming_manager.create_stream(stream_id, raw_stream):
            if not chunk.is_final:
                yield chunk.content
            else:
                # Final chunk - emit completion event
                event_manager.emit(Event(EventType.API_CALL_COMPLETED, {
                    'stream_id': stream_id,
                    'total_chunks': chunk.metadata.get('total_chunks', 0)
                }))

    else:
        # Fallback to non-streaming
        logging.info("Provider doesn't support streaming, using fallback")
        result = generate_with_cache(
            prompt, history, max_tokens, temperature, **kwargs
        )
        yield result

def generate_book_optimized(book: Dict[str, Any],
                           title: str,
                           instructions: str,
                           language: str,
                           use_cache: bool = True,
                           use_streaming: bool = False) -> Generator[Dict[str, Any], None, None]:
    """
    Generate book with all optimizations
    
    Args:
        book: Existing book data
        title: Book title
        instructions: Generation instructions
        language: Target language
        use_cache: Whether to use caching
        use_streaming: Whether to use streaming
        
    Yields:
        Updated book after each step
    """
    from generate import write_book

    # Initialize book context manager
    book_manager = BookContextManager()

    # Original system message
    original_message = {
        "role": "system",
        "content": f"You are a book writer, writing a new book in {language}."
    }

    # Track chapter for context optimization
    current_chapter_idx = 0

    # Custom generate function with optimizations
    def optimized_generate(prompt, history, short=False, force_max=False):
        # Prepare optimized context
        optimized_history = book_manager.prepare_context(
            book,
            current_chapter=current_chapter_idx,
            window_size=5
        )

        # Merge with provided history
        if history and len(history) > 1:
            optimized_history.extend(history[1:])  # Skip duplicate system message

        max_tokens = config.MAX_TOKENS_SHORT if short else config.MAX_TOKENS

        if use_cache:
            # Generate cache key based on book context
            cache_key = cache.create_key(
                'book_gen',
                title=title,
                chapter=current_chapter_idx,
                prompt_hash=hash(prompt[:100])
            )

            return generate_with_cache(
                prompt=prompt,
                history=optimized_history,
                max_tokens=max_tokens,
                temperature=config.TEMPERATURE,
                cache_key=cache_key,
                cache_expire=7200  # 2 hours for book content
            )
        elif use_streaming:
            # Stream generation
            content = ""
            for chunk in generate_stream(
                prompt=prompt,
                history=optimized_history,
                max_tokens=max_tokens,
                temperature=config.TEMPERATURE
            ):
                content += chunk
            return content
        else:
            # Standard generation with optimization
            response = provider.generate(
                prompt=prompt,
                history=optimized_history,
                max_tokens=max_tokens,
                temperature=config.TEMPERATURE
            )
            return response.content

    # Monkey-patch the generate module to use our optimized function
    import generate
    original_callLLM = generate.callLLM
    generate.callLLM = optimized_generate

    try:
        # Generate book with optimizations
        for updated_book in write_book(book, title, instructions, language):
            # Update chapter tracking
            if 'toc' in updated_book:
                completed_chapters = sum(
                    1 for ch in updated_book['toc']['chapters']
                    if ch.get('content')
                )
                current_chapter_idx = completed_chapters

            yield updated_book

    finally:
        # Restore original function
        generate.callLLM = original_callLLM

class EnhancedAI:
    """Enhanced AI interface with all optimizations"""

    def __init__(self):
        self.provider = provider
        self.cache = cache
        self.optimizer = optimizer
        self.book_manager = BookContextManager()

    def generate(self, *args, **kwargs):
        """Standard generation with caching"""
        return generate_with_cache(*args, **kwargs)

    def stream(self, *args, **kwargs):
        """Streaming generation"""
        return generate_stream(*args, **kwargs)

    def generate_book(self, *args, **kwargs):
        """Optimized book generation"""
        return generate_book_optimized(*args, **kwargs)

    def clear_cache(self, pattern: str = None):
        """Clear cache"""
        if pattern:
            self.cache.invalidate_pattern(pattern)
        else:
            self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            'cache': self.cache.get_stats(),
            'provider': self.provider.get_model_info()
        }

# Global enhanced AI instance
enhanced_ai = EnhancedAI()

# Convenience functions
def generate(*args, **kwargs):
    """Generate with enhancements"""
    return enhanced_ai.generate(*args, **kwargs)

def stream(*args, **kwargs):
    """Stream generation"""
    return enhanced_ai.stream(*args, **kwargs)

def generate_book(*args, **kwargs):
    """Generate book with optimizations"""
    return enhanced_ai.generate_book(*args, **kwargs)
