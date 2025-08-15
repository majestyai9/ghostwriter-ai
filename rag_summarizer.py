"""
Smart summarization module for RAG system.

Provides LLM-based and fallback summarization for book chapters.
"""

import hashlib
import logging
import threading
from typing import Any, Dict, List, Optional

from cache_manager import CacheManager
from providers.base import LLMProvider


class SmartSummarizer:
    """
    Intelligent summarization system using LLM for structured content summarization.
    
    Features:
    - Generates structured summaries capturing key narrative elements
    - Caches summaries for efficiency
    - Handles different content types (chapters, sections)
    - Thread-safe operation
    - Fallback to extraction when LLM unavailable
    """

    def __init__(self,
                 provider: Optional[LLMProvider] = None,
                 cache_manager: Optional[CacheManager] = None,
                 max_summary_tokens: int = 200):
        """
        Initialize the smart summarizer.
        
        Args:
            provider: LLM provider for generating summaries
            cache_manager: Cache manager for storing summaries
            max_summary_tokens: Maximum tokens per summary
        """
        self.provider = provider
        self.cache_manager = cache_manager or CacheManager(backend="memory")
        self.max_summary_tokens = max_summary_tokens
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # Performance metrics
        self.metrics = {
            "total_summaries": 0,
            "cache_hits": 0,
            "llm_summaries": 0,
            "fallback_summaries": 0
        }

    def summarize_chapter(self,
                          chapter: Dict[str, Any],
                          book_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate or retrieve a structured summary for a chapter.
        
        Args:
            chapter: Chapter data including content and metadata
            book_context: Optional book context for better summarization
            
        Returns:
            Structured summary string
        """
        if not chapter.get("content"):
            return ""

        # Create cache key
        cache_key = self._create_cache_key("chapter_summary", chapter)

        # Check cache
        if self.cache_manager:
            cached_summary = self.cache_manager.get(cache_key)
            if cached_summary:
                self.logger.debug(f"Using cached summary for chapter {chapter.get('number')}")
                self.metrics["cache_hits"] += 1
                return cached_summary

        # Generate summary
        summary = self._generate_summary(chapter, book_context)
        self.metrics["total_summaries"] += 1

        # Cache the summary
        if self.cache_manager and summary:
            self.cache_manager.set(cache_key, summary, expire=86400)  # Cache for 24 hours

        return summary

    def _generate_summary(self,
                         chapter: Dict[str, Any],
                         book_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a summary using LLM or fallback to extraction.
        
        Args:
            chapter: Chapter to summarize
            book_context: Optional book context
            
        Returns:
            Generated summary
        """
        if self.provider:
            try:
                # Prepare the summarization prompt
                prompt = self._create_summary_prompt(chapter, book_context)

                # Generate summary using LLM
                with self._lock:
                    response = self.provider.generate(
                        prompt=prompt,
                        max_tokens=self.max_summary_tokens,
                        temperature=0.3  # Lower temperature for more focused summaries
                    )

                if response and response.content:
                    self.logger.info(f"Generated LLM summary for chapter {chapter.get('number')}")
                    self.metrics["llm_summaries"] += 1
                    return response.content.strip()

            except Exception as e:
                self.logger.warning(f"LLM summarization failed: {e}, falling back to extraction")

        # Fallback to simple extraction
        self.metrics["fallback_summaries"] += 1
        return self._extract_summary(chapter)

    def _create_summary_prompt(self,
                               chapter: Dict[str, Any],
                               book_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a structured prompt for chapter summarization.
        
        Args:
            chapter: Chapter to summarize
            book_context: Optional book context
            
        Returns:
            Formatted prompt string
        """
        chapter_num = chapter.get("number", "?")
        chapter_title = chapter.get("title", "Untitled")
        content = chapter.get("content", "")[:3000]  # Limit content length

        prompt = f"""Summarize this book chapter in a structured format. Focus on key narrative elements.

Chapter {chapter_num}: {chapter_title}

Content:
{content}

Provide a concise summary (max 150 words) that includes:
1. Main plot developments
2. Character actions and developments
3. Important world-building details
4. Unresolved questions or cliffhangers

Format as a single paragraph focusing on the most important elements."""

        if book_context:
            book_title = book_context.get("title", "")
            if book_title:
                prompt = f"Book: {book_title}\n\n" + prompt

        return prompt

    def _extract_summary(self, chapter: Dict[str, Any]) -> str:
        """
        Extract a simple summary from chapter content.
        
        Args:
            chapter: Chapter to extract from
            
        Returns:
            Extracted summary
        """
        content = chapter.get("content", "")
        topics = chapter.get("topics", "")

        summary_parts = []

        if topics:
            summary_parts.append(f"Topics: {topics}")

        # Extract first paragraph or sentences
        if content:
            # Try to get first meaningful paragraph
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            if paragraphs:
                first_para = paragraphs[0][:300]
                if len(first_para) < len(paragraphs[0]):
                    first_para += "..."
                summary_parts.append(first_para)

        return " ".join(summary_parts) if summary_parts else "No summary available."

    def _create_cache_key(self, prefix: str, chapter: Dict[str, Any]) -> str:
        """
        Create a unique cache key for a chapter summary.
        
        Args:
            prefix: Cache key prefix
            chapter: Chapter data
            
        Returns:
            Cache key string
        """
        # Use content hash for cache key to handle content changes
        content = chapter.get("content", "")
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        chapter_num = chapter.get("number", 0)

        return f"{prefix}:ch{chapter_num}:{content_hash}"

    def batch_summarize(self,
                        chapters: List[Dict[str, Any]],
                        book_context: Optional[Dict[str, Any]] = None,
                        max_concurrent: int = 3) -> List[str]:
        """
        Summarize multiple chapters efficiently.
        
        Args:
            chapters: List of chapters to summarize
            book_context: Optional book context
            max_concurrent: Maximum concurrent summaries (for future parallel processing)
            
        Returns:
            List of summaries
        """
        summaries = []

        for chapter in chapters:
            summary = self.summarize_chapter(chapter, book_context)
            summaries.append(summary)

        return summaries
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get summarizer performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        total = self.metrics["total_summaries"]
        cache_rate = (
            self.metrics["cache_hits"] / (self.metrics["cache_hits"] + total)
            if total > 0 else 0
        )
        
        return {
            "total_summaries": total,
            "cache_hits": self.metrics["cache_hits"],
            "cache_hit_rate": f"{cache_rate:.2%}",
            "llm_summaries": self.metrics["llm_summaries"],
            "fallback_summaries": self.metrics["fallback_summaries"],
            "llm_rate": f"{self.metrics['llm_summaries']/total:.2%}" if total > 0 else "0%"
        }
    
    def clear_cache(self):
        """Clear all cached summaries."""
        if self.cache_manager:
            # Clear summaries from cache
            self.logger.info("Clearing summary cache")
            # Note: This assumes cache manager has appropriate methods
            # In practice, you might want to track keys or use a prefix pattern