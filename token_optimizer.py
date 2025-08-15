"""
Token optimization with sliding window context management and budget tracking.
"""
import logging
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional


class ContextPriority(Enum):
    """Priority levels for context elements"""
    ESSENTIAL = 1  # Must always be included (title, summary)
    HIGH = 2       # Recent content
    MEDIUM = 3     # Related content
    LOW = 4        # Older content

@dataclass
class ContextElement:
    """Represents an element in the context window"""
    content: str
    tokens: int
    priority: ContextPriority
    metadata: Dict[str, Any]

class SlidingWindowManager:
    """Manages context with sliding window optimization"""

    def __init__(self, max_tokens: int = 128000, reserved_tokens: int = 4096):
        """
        Initialize sliding window manager
        
        Args:
            max_tokens: Maximum context window size
            reserved_tokens: Tokens reserved for response
        """
        self.max_tokens = max_tokens
        self.reserved_tokens = reserved_tokens
        self.available_tokens = max_tokens - reserved_tokens
        self.logger = logging.getLogger(__name__)

    def optimize_context(self,
                        elements: List[ContextElement],
                        current_tokens: int = 0) -> List[ContextElement]:
        """
        Optimize context to fit within token limits
        
        Args:
            elements: List of context elements
            current_tokens: Tokens already used
            
        Returns:
            Optimized list of context elements
        """
        available = self.available_tokens - current_tokens

        # Sort by priority and metadata
        sorted_elements = sorted(
            elements,
            key=lambda x: (x.priority.value, -x.metadata.get('recency', 0))
        )

        optimized = []
        used_tokens = 0

        for element in sorted_elements:
            if used_tokens + element.tokens <= available:
                optimized.append(element)
                used_tokens += element.tokens
            elif element.priority == ContextPriority.ESSENTIAL:
                # Essential elements must be included - compress if needed
                compressed = self._compress_element(element, available - used_tokens)
                if compressed:
                    optimized.append(compressed)
                    used_tokens += compressed.tokens

        self.logger.info(f"Context optimized: {len(elements)} -> {len(optimized)} elements, "
                        f"{used_tokens}/{available} tokens used")

        return optimized

    def _compress_element(self, element: ContextElement, max_tokens: int) -> Optional[ContextElement]:
        """
        Compress an element to fit within token limit
        
        Args:
            element: Element to compress
            max_tokens: Maximum tokens allowed
            
        Returns:
            Compressed element or None if cannot compress
        """
        if element.tokens <= max_tokens:
            return element

        # Simple truncation for now - could use summarization
        content = element.content
        while self._estimate_tokens(content) > max_tokens and len(content) > 100:
            content = content[:int(len(content) * 0.9)] + "..."

        if len(content) < 100:
            return None

        return ContextElement(
            content=content,
            tokens=self._estimate_tokens(content),
            priority=element.priority,
            metadata={**element.metadata, 'compressed': True}
        )

    @lru_cache(maxsize=1024)
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count with improved accuracy and caching
        Uses 1.3 tokens per word formula with punctuation factor
        """
        if not text:
            return 0

        # Count words and punctuation
        words = len(text.split())
        punctuation_count = sum(1 for char in text if char in '.,;:!?"\'()[]{}')

        # Improved estimation: ~1.3 tokens per word + punctuation factor
        estimated_tokens = int(words * 1.3 + punctuation_count * 0.1)

        return estimated_tokens

class BookContextManager:
    """Manages context specifically for book generation"""

    def __init__(self, max_tokens: int = 128000):
        self.window_manager = SlidingWindowManager(max_tokens)
        self.logger = logging.getLogger(__name__)

    def prepare_context(self,
                       book: Dict[str, Any],
                       current_chapter: int = None,
                       window_size: int = 5) -> List[Dict[str, str]]:
        """
        Prepare optimized context for book generation
        
        Args:
            book: Book data
            current_chapter: Current chapter being generated
            window_size: Number of recent chapters to include
            
        Returns:
            Optimized context as message list
        """
        elements = []

        # Essential elements (always included)
        elements.append(ContextElement(
            content=f"Title: {book.get('title', 'Untitled')}",
            tokens=self._estimate_tokens(book.get('title', '')),
            priority=ContextPriority.ESSENTIAL,
            metadata={'type': 'title'}
        ))

        if book.get('summary'):
            elements.append(ContextElement(
                content=f"Summary: {book['summary']}",
                tokens=self._estimate_tokens(book['summary']),
                priority=ContextPriority.ESSENTIAL,
                metadata={'type': 'summary'}
            ))

        # Table of contents (compressed)
        if book.get('toc'):
            toc_text = self._format_toc(book['toc'], highlight_chapter=current_chapter)
            elements.append(ContextElement(
                content=f"Table of Contents:\n{toc_text}",
                tokens=self._estimate_tokens(toc_text),
                priority=ContextPriority.HIGH,
                metadata={'type': 'toc'}
            ))

        # Recent chapters (sliding window)
        if book.get('toc', {}).get('chapters'):
            chapters = book['toc']['chapters']

            if current_chapter is not None:
                # Include chapters around current
                start_idx = max(0, current_chapter - window_size)
                end_idx = min(len(chapters), current_chapter + 1)

                for i in range(start_idx, end_idx):
                    chapter = chapters[i]
                    recency = window_size - abs(i - current_chapter)

                    # Full content for very recent chapters
                    if abs(i - current_chapter) <= 2 and chapter.get('content'):
                        elements.append(ContextElement(
                            content=f"Chapter {chapter['number']}: {chapter['title']}\n{chapter['content']}",
                            tokens=self._estimate_tokens(chapter.get('content', '')),
                            priority=ContextPriority.HIGH,
                            metadata={'type': 'chapter', 'number': i, 'recency': recency}
                        ))
                    # Summary for older chapters
                    elif chapter.get('content'):
                        summary = self._summarize_chapter(chapter)
                        elements.append(ContextElement(
                            content=f"Chapter {chapter['number']} Summary: {summary}",
                            tokens=self._estimate_tokens(summary),
                            priority=ContextPriority.MEDIUM,
                            metadata={'type': 'chapter_summary', 'number': i, 'recency': recency}
                        ))

        # Optimize and convert to messages
        optimized = self.window_manager.optimize_context(elements)

        # Convert to message format
        messages = []

        # System message with essential context
        system_content = "You are a book writer. "
        for elem in optimized:
            if elem.priority == ContextPriority.ESSENTIAL:
                system_content += elem.content + " "

        messages.append({"role": "system", "content": system_content.strip()})

        # Add other context as system messages
        for elem in optimized:
            if elem.priority != ContextPriority.ESSENTIAL:
                messages.append({"role": "system", "content": elem.content})

        return messages

    def _format_toc(self, toc: Dict[str, Any], highlight_chapter: int = None) -> str:
        """Format table of contents with optional highlighting"""
        lines = []
        for i, chapter in enumerate(toc.get('chapters', [])):
            if highlight_chapter is not None and i == highlight_chapter:
                lines.append(f">>> {chapter['number']}. {chapter['title']} <<<")
                for section in chapter.get('sections', []):
                    lines.append(f"    {chapter['number']}.{section['number']}. {section['title']}")
            else:
                lines.append(f"{chapter['number']}. {chapter['title']}")
                if highlight_chapter is None or abs(i - highlight_chapter) <= 2:
                    # Show sections for nearby chapters
                    section_count = len(chapter.get('sections', []))
                    lines.append(f"    ({section_count} sections)")

        return '\n'.join(lines)

    def _summarize_chapter(self, chapter: Dict[str, Any]) -> str:
        """Create a summary of a chapter"""
        # Simple truncation for now - could use AI summarization
        content = chapter.get('content', '')
        topics = chapter.get('topics', '')

        if topics:
            summary = f"Topics: {topics}. "
        else:
            summary = ""

        # Take first 200 characters
        if content:
            summary += content[:200] + "..."

        return summary

    @lru_cache(maxsize=1024)
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count with improved accuracy and caching
        Uses 1.3 tokens per word formula with punctuation factor
        """
        if not text:
            return 0

        # Count words and punctuation
        words = len(text.split())
        punctuation_count = sum(1 for char in text if char in '.,;:!?"\'()[]{}')

        # Improved estimation: ~1.3 tokens per word + punctuation factor
        estimated_tokens = int(words * 1.3 + punctuation_count * 0.1)

        return estimated_tokens

class TokenOptimizer:
    """
    Main token optimization interface with efficient caching and batch processing
    
    Features:
    - Token count caching with LRU cache
    - Batch token counting for efficiency
    - Provider-specific tokenizers when available
    - Improved estimation accuracy
    """

    def __init__(self, provider=None, cache_size: int = 2048):
        """
        Initialize token optimizer
        
        Args:
            provider: LLM provider for accurate token counting
            cache_size: Size of token count cache (default: 2048)
        """
        self.provider = provider
        self.book_manager = BookContextManager()
        self.logger = logging.getLogger(__name__)

        # Token cache dictionary for frequently accessed strings
        self.token_cache: Dict[int, int] = {}  # hash -> token count
        self.cache_size = cache_size

    @lru_cache(maxsize=2048)
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text with caching
        
        Args:
            text: Text to count
            
        Returns:
            Token count
        """
        if not text:
            return 0

        # Check provider's tokenizer first
        if self.provider and hasattr(self.provider, 'count_tokens'):
            try:
                return self.provider.count_tokens(text)
            except Exception as e:
                self.logger.warning(f"Provider token counting failed: {e}, using estimation")

        # Use improved estimation with caching
        return self._estimate_tokens_improved(text)

    @lru_cache(maxsize=2048)
    def _estimate_tokens_improved(self, text: str) -> int:
        """
        Improved token estimation with 1.3 tokens per word formula
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Count words and special characters
        words = len(text.split())

        # Count various punctuation and special characters
        punctuation_count = sum(1 for char in text if char in '.,;:!?"\'()[]{}')
        special_chars = sum(1 for char in text if char in '@#$%^&*+=<>/\\|`~')

        # Improved formula: ~1.3 tokens per word + factors for punctuation and special chars
        base_tokens = words * 1.3
        punctuation_factor = punctuation_count * 0.1
        special_factor = special_chars * 0.2

        estimated_tokens = int(base_tokens + punctuation_factor + special_factor)

        return max(1, estimated_tokens)  # At least 1 token for non-empty text

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """
        Count tokens for multiple texts efficiently
        
        Args:
            texts: List of texts to count
            
        Returns:
            List of token counts
        """
        results = []

        # If provider supports batch counting, use it
        if self.provider and hasattr(self.provider, 'count_tokens_batch'):
            try:
                return self.provider.count_tokens_batch(texts)
            except Exception as e:
                self.logger.warning(f"Batch token counting failed: {e}, using individual counting")

        # Otherwise, count individually with caching
        for text in texts:
            results.append(self.count_tokens(text))

        return results

    def estimate_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Estimate total tokens in a message list
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Total estimated token count
        """
        total = 0

        for message in messages:
            # Count role tokens (usually 1-2 tokens)
            total += 2

            # Count content tokens
            content = message.get('content', '')
            total += self.count_tokens(content)

            # Add separator tokens (usually 3-4 tokens between messages)
            total += 4

        return total

    def optimize_messages(self,
                         messages: List[Dict[str, str]],
                         max_tokens: int,
                         preserve_recent: int = 3) -> List[Dict[str, str]]:
        """
        Optimize message history to fit within token limits
        
        Args:
            messages: Message history
            max_tokens: Maximum tokens allowed
            preserve_recent: Number of recent messages to preserve
            
        Returns:
            Optimized message list
        """
        if not messages:
            return messages

        # Always keep system message
        optimized = [messages[0]] if messages[0]['role'] == 'system' else []

        # Calculate available tokens
        system_tokens = self.count_tokens(optimized[0]['content']) if optimized else 0
        available = max_tokens - system_tokens

        # Preserve recent messages
        recent = messages[-preserve_recent:] if len(messages) > preserve_recent else messages[1:]
        recent_tokens = sum(self.count_tokens(msg['content']) for msg in recent)

        if recent_tokens <= available:
            # Add all recent messages
            optimized.extend(recent)
            available -= recent_tokens

            # Try to add older messages
            older = messages[1:-preserve_recent] if len(messages) > preserve_recent + 1 else []
            for msg in reversed(older):
                msg_tokens = self.count_tokens(msg['content'])
                if msg_tokens <= available:
                    optimized.insert(1, msg)  # Insert after system message
                    available -= msg_tokens

        else:
            # Need to truncate even recent messages
            for msg in recent:
                msg_tokens = self.count_tokens(msg['content'])
                if msg_tokens <= available:
                    optimized.append(msg)
                    available -= msg_tokens

        self.logger.info(f"Messages optimized: {len(messages)} -> {len(optimized)}")
        return optimized

    def create_summary_message(self,
                               messages: List[Dict[str, str]],
                               max_tokens: int = 500) -> Dict[str, str]:
        """
        Create a summary of messages that don't fit
        
        Args:
            messages: Messages to summarize
            max_tokens: Maximum tokens for summary
            
        Returns:
            Summary message
        """
        # For now, just concatenate key points
        summary_points = []
        tokens_used = 0

        for msg in messages:
            if msg['role'] == 'assistant':
                # Take first sentence or 100 chars
                point = msg['content'].split('.')[0][:100]
                point_tokens = self.count_tokens(point)

                if tokens_used + point_tokens <= max_tokens:
                    summary_points.append(point)
                    tokens_used += point_tokens

        if summary_points:
            return {
                'role': 'system',
                'content': f"Previous context summary: {'; '.join(summary_points)}"
            }
        return None

# Global token optimizer with thread safety
import threading

_token_optimizer = None
_optimizer_lock = threading.Lock()

def initialize_optimizer(provider=None, max_tokens: int = 128000):
    """Initialize global token optimizer with thread safety."""
    global _token_optimizer
    with _optimizer_lock:
        _token_optimizer = TokenOptimizer(provider, max_tokens_per_request=max_tokens)
    return _token_optimizer

def get_optimizer() -> TokenOptimizer:
    """Get global token optimizer with double-checked locking."""
    global _token_optimizer

    # First check without locking for performance
    if _token_optimizer is None:
        with _optimizer_lock:
            # Double-check after acquiring lock
            if _token_optimizer is None:
                _token_optimizer = TokenOptimizer()

    return _token_optimizer
