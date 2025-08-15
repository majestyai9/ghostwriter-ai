"""
Hybrid context management module for RAG system.

Provides intelligent context preparation combining multiple strategies.
"""

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

from rag_summarizer import SmartSummarizer
from token_optimizer import (
    BookContextManager,
    ContextElement,
    ContextPriority,
    SlidingWindowManager,
)


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
                 config: Any,
                 provider: Any = None,
                 cache_manager: Any = None,
                 retriever: Any = None,
                 max_tokens: int = 128000):
        """
        Initialize the hybrid context manager.
        
        Args:
            config: RAG configuration
            provider: LLM provider for summarization
            cache_manager: Cache manager
            retriever: Semantic context retriever
            max_tokens: Maximum context window size
        """
        self.config = config
        self.provider = provider
        self.cache_manager = cache_manager
        self.retriever = retriever
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.legacy_manager = BookContextManager(max_tokens)
        self.window_manager = SlidingWindowManager(max_tokens)

        # Initialize summarizer if provider is available
        self.summarizer = SmartSummarizer(
            provider=provider,
            cache_manager=cache_manager
        ) if provider else None

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
        if self.config.mode.value == "disabled":
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
            summary_elements = self._prepare_summary_context(
                book, current_chapter, summary_budget
            )
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
        """Get comprehensive statistics about context management."""
        stats = {
            "mode": self.config.mode.value,
            "max_tokens": self.max_tokens,
            "token_distribution": {
                "core": f"{self.config.core_context_ratio:.0%}",
                "rag": f"{self.config.rag_context_ratio:.0%}",
                "summary": f"{self.config.summary_context_ratio:.0%}"
            },
            "features": {
                "gpu_enabled": self.config.use_gpu,
                "ivf_indexing": self.config.use_ivf,
                "batch_processing": self.config.enable_batch_processing,
                "caching": self.config.enable_caching
            }
        }

        if self.retriever:
            stats["retriever"] = self.retriever.get_performance_metrics()

        if self.summarizer:
            stats["summarizer"] = self.summarizer.get_metrics()

        return stats